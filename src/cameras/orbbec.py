import os
import cv2
import time
import base64
import argparse
import threading
import numpy as np
import pyorbbecsdk as ob 
from typing import Optional
from datetime import datetime
from collections import deque
from termcolor import colored, cprint
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse

from google import genai
from google.genai import types
from lang_sam import LangSAM

from .utils import generate_bbox, parse_response, clean_bbox_dict, VisionDetector

DETECTION_PROMPT = """
          Point to no more than 10 items in the image. The label returned
          should be an identifying name for the object detected.
          The answer should follow the json format: [{"point": <point>,
          "label": <label1>}, ...]. The points are in [y, x] format
          normalized to 0-1000.
        """

camera_capture: Optional["OrbbecCamera"] = None

class TemporalFilter():
    """Temporal filter for depth frames to reduce noise"""

    def __init__(self, alpha):
        self.alpha = alpha
        self.previous_frame = None

    def process(self, frame):
        if self.previous_frame is None:
            result = frame
        else:
            result = cv2.addWeighted(frame, self.alpha, self.previous_frame, 1 - self.alpha, 0)
        self.previous_frame = result
        return result

class OrbbecCamera():
    def __init__(self, output_fps: int, max_frames: int):
        self.output_fps = output_fps
        self.max_frames = max_frames
        self.min_depth = 200
        self.max_depth = 20000
        self.device_found = False
        self.langsam_model = LangSAM() # Orbbec should have a sam model for object segmentation.
        print(colored("[Camera] ", "green") + "LangSAM model loaded successfully.")

    def check_for_device(self):
        self.device_found = False
        context = ob.Context()
        device_list = context.query_devices()
        if device_list.get_count() >= 1:
            self.device_found = True
        else:
            cprint("[Camera] Error: No Orbbec camera found.", "red")
        return self.device_found
    
    def get_stream_config(self):
        """
        Get matched color and depth stream profiles
        """
        try:
            profile_list = self.pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
            assert profile_list is not None
            
            for i in range(len(profile_list)):
                color_profile = profile_list[i]

                if color_profile.get_format() != ob.OBFormat.RGB:
                    continue

                d2c_profile_list = self.pipeline.get_d2c_depth_profile_list(color_profile, ob.OBAlignMode.HW_MODE)
                if len(d2c_profile_list) == 0:
                    continue

                d2c_profile = d2c_profile_list[0]
                print(colored("[Camera] ", "green") + f"Found compatible d2c color profile: {color_profile}, depth profile: {d2c_profile}")

                self.config.enable_stream(d2c_profile)
                self.config.enable_stream(color_profile)

                self.config.set_align_mode(ob.OBAlignMode.HW_MODE)
                return self.config
        except Exception as e:
            cprint("[Camera] Error while getting stream config: " + str(e), "red")
            return None
    
    def start(self):
        self.pipeline = ob.Pipeline()
        self.config = ob.Config()
        self.color_buffer = deque(maxlen=self.max_frames)
        self.depth_buffer = deque(maxlen=self.max_frames)
        self.aligned_buffer = deque(maxlen=self.max_frames)
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        self.temporal_filter = TemporalFilter(alpha=0.1)
        self.point_cloud_filter = ob.PointCloudFilter()

        # set stream config
        self.get_stream_config()
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join()
        if hasattr(self, "pipeline"):
            self.pipeline.stop()

    def _capture_loop(self):
        self.pipeline.enable_frame_sync()
        self.pipeline.start(self.config)
        param = self.pipeline.get_camera_param()
        print(colored("[Camera] ", "green") + f"Camera parameters: {param.depth_intrinsic}")
        self.fx, self.fy, self.cx, self.cy = param.depth_intrinsic.fx, param.depth_intrinsic.fy, param.depth_intrinsic.cx, param.depth_intrinsic.cy
        self.color_fx, self.color_fy, self.color_cx, self.color_cy = param.rgb_intrinsic.fx, param.rgb_intrinsic.fy, param.rgb_intrinsic.cx, param.rgb_intrinsic.cy
        dist = param.rgb_distortion
        self.color_distortion = [dist.k1, dist.k2, dist.p1, dist.p2, dist.k3, dist.k4, dist.k5, dist.k6]
        frame_counter = 0
        last_time = time.time_ns()
        while self.running:
            frames = self.pipeline.wait_for_frames(100)
            if frames is None:
                continue

            curr_time = time.time_ns()
            remaining_time = (1 / self.output_fps) - (curr_time - last_time) / 1e9
            if (0.0001 < remaining_time < 0.2):
                time.sleep(remaining_time)
            # print((time.time_ns() - last_time) / 1e9)
            last_time = time.time_ns()

            frame_counter += 1

            # Get RGB Frame
            rgb_frame = frames.get_color_frame()
            # Get Depth Frame
            depth_frame = frames.get_depth_frame()
            if rgb_frame is None or depth_frame is None:
                continue

            width, height = rgb_frame.get_width(), rgb_frame.get_height()
            assert rgb_frame.get_format() == ob.OBFormat.RGB
            rgb_image = np.resize(np.asanyarray(rgb_frame.get_data()), (height, width, 3))
            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            bgr_image = cv2.flip(bgr_image, 0)
            bgr_image = cv2.flip(bgr_image, 1)

            with self.lock:
                timestamp = datetime.now()
                self.color_buffer.append({
                    "frame": bgr_image,
                    "timestamp": timestamp
                })

            width, height = depth_frame.get_width(), depth_frame.get_height()
            scale = depth_frame.get_depth_scale()
            depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
            depth_data = depth_data.reshape((height, width))
            depth_data = depth_data.astype(np.float32) * scale
            depth_data = np.where((depth_data > self.min_depth) & (depth_data < self.max_depth), depth_data, 0)
            depth_data = depth_data.astype(np.uint16)
            depth_data = self.temporal_filter.process(depth_data)
            depth_data = cv2.flip(depth_data, 0)
            depth_data = cv2.flip(depth_data, 1)

            with self.lock:
                timestamp = datetime.now()
                self.depth_buffer.append({
                    "frame": depth_data,
                    "timestamp": timestamp
                })

    def get_latest_frames(self):
        latest_color = None
        latest_depth = None
        if len(self.color_buffer) > 0 and len(self.depth_buffer) > 0:
            with self.lock:
                latest_color = self.color_buffer[-1]
                latest_depth = self.depth_buffer[-1]
        return latest_color, latest_depth
    
    def get_image(self):
        return self.get_latest_frames()[0]
    
    def object_detection(self, image=None):
        """Run object detection on an in-memory frame payload from self.color_buffer."""
        if image is None:
            image = self.get_image()
        assert image is not None, "No image available for object detection."

        if isinstance(image, dict):
            if "frame" not in image:
                raise ValueError("Image dict must include a 'frame' key.")
            frame = image["frame"]
        else:
            frame = image
        assert isinstance(frame, np.ndarray), "Input image must be a numpy array."

        success, encoded = cv2.imencode(".png", frame)
        assert success, "Failed to encode image for object detection."
        image_bytes = encoded.tobytes()

        api_key = os.environ.get("GENAI_API_KEY", "GET_YOUR_OWN_KEY")
        client = genai.Client(api_key = api_key)
        image_response = client.models.generate_content(
            model="gemini-robotics-er-1.5-preview",
            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type='image/png',
                ),
                DETECTION_PROMPT
            ],
            config = types.GenerateContentConfig(
                temperature=0.5,
                thinking_config=types.ThinkingConfig(thinking_budget=0)
            )
        )
        client.close()
        del client

        print(colored("[Camera] ", "green") + f"Object detection response: {image_response.text}")
        return image_response.text

    def find_objects(self):
        last_frame, last_depth = self.get_latest_frames()
        assert last_frame is not None and last_depth is not None, "No frames available for object detection."

        llm_response = self.object_detection(last_frame)
        assert llm_response is not None and isinstance(llm_response, str), "LLM did not return a valid response for object detection."

        json_response = parse_response(llm_response)
        assert json_response is not None and isinstance(json_response, list), "Parsed LLM response is not a valid list of detections."

        obbs, annotated_bgr = generate_bbox(depth_mm=last_depth["frame"], 
                                            color_bgr=last_frame["frame"], 
                                            detections=json_response,
                                            langsam_model=self.langsam_model,
                                            fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy)
        
        success, buffer = cv2.imencode(".png", annotated_bgr)
        assert success, "Failed to encode annotated image for visualization."

        bbox = clean_bbox_dict(obbs)
        center = {item["label"]: item.get("centers", None) for item in obbs}
        content = {
            "image": base64.b64encode(buffer.tobytes()).decode("utf-8"),
            "objects": center,
            "bbox": bbox,
        }
        return content
    
    def read_ArUco(self, image: Optional[np.ndarray] = None):
        if image is None:
            image = self.get_image()
        assert image is not None, "No image available for object detection."

        if isinstance(image, dict):
            if "frame" not in image:
                raise ValueError("Image dict must include a 'frame' key.")
            frame = image["frame"]
        else:
            frame = image
        assert isinstance(frame, np.ndarray), "Input image must be a numpy array."
        aruco_detector = VisionDetector(
                                        fx=self.color_fx, fy=self.color_fy, cx=self.color_cx, cy=self.color_cy, 
                                        distortion=self.color_distortion,
                                        marker_size=0.06)

        rvecs, tvecs, corners, ids, frame_markers = aruco_detector.plot_aruco(frame)
        marker_poses_camera, marker_poses_robot = aruco_detector.pose_vectors_to_cart(rvecs, tvecs)
        print(colored("[Camera] ", "green") + f"Detected ArUco markers with IDs: {ids.flatten() if ids is not None else 'None'}")
        print(colored("[Camera] ", "green") + f"Marker Cartesian poses in camera frame: {marker_poses_camera}")
        print(colored("[Camera] ", "green") + f"Marker Cartesian poses in robot frame: {marker_poses_robot}")
        return marker_poses_camera, marker_poses_robot


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global camera_capture
    camera_capture = OrbbecCamera(
        output_fps=20,
        max_frames=20 * 60, # buffer for 60 seconds
    )
    if not camera_capture.check_for_device():
        return
    
    camera_capture.start()
    yield
    camera_capture.stop()




def main():
    global camera_capture
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--FastAPI",
        action="store_true",
        help="use FastAPI server to serve camera data",
    )
    args = parser.parse_args()

    if args.FastAPI:
        import uvicorn
        app = FastAPI(lifespan=lifespan)

        @app.get("/find_objects")
        async def find_objects():
            cam = camera_capture
            if cam is None:
                raise HTTPException(status_code=503, detail="Camera is not initialized")
            content = cam.find_objects()
            return JSONResponse(content = content)
        
        @app.get("/latest/color")
        async def get_latest_color_image():
            """Get the most recent color PNG image"""
            cam = camera_capture
            if cam is None:
                raise HTTPException(status_code=503, detail="Camera is not initialized")

            latest_frame, latest_depth = cam.get_latest_frames()

            if latest_frame is None:
                error_msg = f"No color frames available yet."
                print(f"[503] {error_msg}")
                raise HTTPException(status_code=503, detail=error_msg)

            # Encode frame to PNG in memory
            success, buffer = cv2.imencode('.png', latest_frame['frame'])
            if not success:
                raise HTTPException(status_code=500, detail="Failed to encode image")

            return Response(
                content=buffer.tobytes(),
                media_type="image/png")

        uvicorn.run(app, host="0.0.0.0", port=8010)

    else: 
        camera_capture = OrbbecCamera(
            output_fps=20,
            max_frames=20 * 60, # buffer for 60 seconds
        )
        camera_capture.start()
        time.sleep(8) # allow camera to warm up and fill buffer
        while True:
            try:
                test = input("****************************** ")
                camera_capture.read_ArUco()
                continue
            except KeyboardInterrupt:
                print("Exiting...")
                break


if __name__ == "__main__":
    main()





# def get_frame_data(color_frame, depth_frame):
#     color_frame = color_frame.as_video_frame()
#     depth_frame = depth_frame.as_video_frame()

#     depth_width = depth_frame.get_width()
#     depth_height = depth_frame.get_height()

#     color_profile = color_frame.get_stream_profile()
#     depth_profile = depth_frame.get_stream_profile()
#     color_intrinsics = color_profile.as_video_stream_profile().get_intrinsic()
#     color_distortion = color_profile.as_video_stream_profile().get_distortion()
#     depth_intrinsics = depth_profile.as_video_stream_profile().get_intrinsic()
#     depth_distortion = depth_profile.as_video_stream_profile().get_distortion()
#     extrinsic = depth_profile.get_extrinsic_to(color_profile)
#     depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(depth_height, depth_width)
#     return (color_intrinsics, color_distortion, depth_intrinsics, depth_distortion,
#             extrinsic, depth_data, depth_width, depth_height)
    

        