import re
import cv2
import json
import numpy as np
import open3d as o3d
from PIL import Image

C2R_TRANSFORM_MATRIX = np.array(
    [[-6.61005715e-07,  1.16412109e-04, -1.02513453e-03,  1.47849877e+00],
    [ 4.45458234e-04,  5.96769214e-06,  1.87086144e-04, -1.08528400e+00],
    [ 3.07633630e-05, -4.29136120e-04, -7.85084248e-04,  1.42743140e+00],
    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]
)

def load_c2r_matrix() -> np.ndarray | None:
    try:
        C2R = np.load("C2R.npy")
    except Exception:
        return None
    C2R = np.asarray(C2R, dtype=np.float64)
    if C2R.shape != (4, 4):
        return None
    return C2R

def _compare_transforms(tag: str, ref: np.ndarray, cand: np.ndarray, tol_m: float = 0.05) -> None:
    ref = np.asarray(ref, dtype=np.float64)
    cand = np.asarray(cand, dtype=np.float64)
    if ref.shape != cand.shape:
        print(f"[C2R compare] {tag}: shape mismatch ref={ref.shape} cand={cand.shape}")
        return
    err = np.linalg.norm(ref - cand, axis=-1) if ref.ndim > 1 else np.linalg.norm(ref - cand)
    max_err = float(np.max(err)) if np.ndim(err) > 0 else float(err)
    mean_err = float(np.mean(err)) if np.ndim(err) > 0 else float(err)
    if max_err > tol_m:
        print(f"[C2R compare] {tag}: mismatch mean={mean_err:.4f}m max={max_err:.4f}m (tol={tol_m:.4f}m)")

FILTER_PHRASES = [
    "robot", "arm", "surface", "frame", "gripper", "table",
    "tabletop", "background", "container", "box", "cable", "support",
    "curtain", "cloth", "cable", "power", "cord", "wall", "floor",
    "camera", "base", "light", "warning", "pole", "desk", 
    "button", "joint", "fabric", "control", "end"
]

def parse_response(response_text: str):
    json_match = re.search(r'\[\s*\{.*?\}\s*\]', response_text, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
        detections = json.loads(json_str)

        # Filter out objects with names in FILTER_PHRASES and flip X/Y coordinates
        filtered_detections = []
        for detection in detections:
            label = detection.get('label', '')

            # Check if any filter phrase appears in the label
            skip = False
            for word in label.lower().split():
                for filter_phrase in FILTER_PHRASES:
                    if filter_phrase in word:
                        skip = True
                        break

            if not skip:
                # Flip X and Y coordinates (swap positions in the point array)
                point = detection.get('point', [0, 0])
                fx = 3840.0 / 1000.0 * point[1]
                fy = 2160.0 / 1000.0 * point[0]
                detection['point'] = [fx, fy]  # Swap Y and X
                filtered_detections.append(detection)

        return filtered_detections
    return None

def uvz_to_xyz(uvz, fx, fy, cx, cy):
    uvz = np.asarray(uvz, dtype=np.float64)
    u = uvz[:, 0]
    v = uvz[:, 1]
    z = uvz[:, 2]
    X = (u - cx) * z / fx
    Y = (v - cy) * z / fy
    Z = z
    return np.stack([X, Y, Z], axis=1)


def xyz_to_duv(xyz: np.ndarray, fx: float, fy: float, cx: float, cy: float):
    xyz = np.asarray(xyz, dtype=np.float64)
    X = xyz[:, 0]
    Y = xyz[:, 1]
    Z = xyz[:, 2]
    Z = np.maximum(Z, 1e-6)
    u = fx * (X / Z) + cx
    v = fy * (Y / Z) + cy
    return u, v

def generate_bbox(depth_mm, color_bgr, detections, langsam_model, fx, fy, cx, cy):
    C2R_np = load_c2r_matrix()

    def draw_object_pc_on_color(img, pts_duvz, stride=3, radius=1, alpha=0.45):
        """
        pts_duvz: (N,3) in depth pixel coords + z(mm)
        stride:   subsample to keep it fast
        """
        if pts_duvz is None or len(pts_duvz) == 0:
            return

        # subsample
        pts = pts_duvz[::max(1, int(stride))]

        overlay = img.copy()
        # choose a visible color (BGR)
        col = (0, 255, 0) 

        for u_d, v_d, _z in pts:
            px, py = float(u_d), float(v_d)
            ix = int(np.clip(round(px), 0, Wc - 1))
            iy = int(np.clip(round(py), 0, Hc - 1))
            cv2.circle(overlay, (ix, iy), int(radius), col, -1, lineType=cv2.LINE_AA)

        cv2.addWeighted(overlay, float(alpha), img, 1.0 - float(alpha), 0, img)

    def draw_wireframe(img, uv, label):
        maroon = (0, 0, 128)   # BGR

        edges = [
            (0, 3), (3, 5), (5, 2), (2, 0),
            (1, 6), (6, 4), (4, 7), (7, 1),
            (4, 5), (6, 3), (7, 2), (1, 0)
        ]

        for a, b in edges:
            cv2.line(img, tuple(uv[a]), tuple(uv[b]),
                    maroon, 2, lineType=cv2.LINE_AA)

        top = int(np.argmin(uv[:, 1]))
        tx, ty = uv[top]

        cv2.putText(img, label, (tx, max(0, ty - 8)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, maroon, 3, cv2.LINE_AA)
    
    def draw_uv_indices(img, uv):
        for i, (x, y) in enumerate(uv):
            x = int(x)
            y = int(y)

            # draw the dot
            cv2.circle(img, (x, y), 6, (0, 0, 255), -1, cv2.LINE_AA)

            font_scale = 2.0      
            thickness  = 4       

            cv2.putText(img, str(i), (x+8, y-8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (0, 0, 0), thickness+2, cv2.LINE_AA)

            cv2.putText(img, str(i), (x+8, y-8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (0, 0, 255), thickness, cv2.LINE_AA)
            
    def mask_filter(mask_u8: np.ndarray, outline_px: int = 2) -> np.ndarray:
        m = (mask_u8 > 0).astype(np.uint8)
        if outline_px <= 0:
            return m

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        m_inner = cv2.erode(m, k, iterations=int(outline_px))
        return m_inner
    
    def pc_filter_xyz_fast(pts_xyz: np.ndarray,
                        *,
                        mad_k: float = 2.5,
                        voxel: float = 5.0,   # mm
                        radius: float = 12.0, # mm
                        min_nb: int = 10,
                        nb_neighbors: int = 40,
                        std_ratio: float = 1.0):
        pts = np.asarray(pts_xyz, dtype=np.float64)

        # MAD z-clip
        z = pts[:, 2]
        med = np.median(z)
        mad = np.median(np.abs(z - med)) + 1e-6
        robust_sigma = 1.4826 * mad
        pts = pts[np.abs(z - med) <= (mad_k * robust_sigma)]
        if pts.shape[0] < 200:
            return pts.astype(np.float32)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)

        # downsample
        if voxel and voxel > 0:
            pcd = pcd.voxel_down_sample(voxel_size=float(voxel))

        # radius outlier
        pcd, _ = pcd.remove_radius_outlier(nb_points=int(min_nb), radius=float(radius))
        if len(pcd.points) < 200:
            return np.asarray(pcd.points, dtype=np.float32)

        # statistical outlier (faster than DBSCAN)
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=int(nb_neighbors),
                                                std_ratio=float(std_ratio))

        return np.asarray(pcd.points, dtype=np.float32)

    Hc, Wc = color_bgr.shape[:2]
    Hd, Wd = depth_mm.shape[:2]
    annotated = color_bgr.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    image_pil = Image.fromarray(cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)) # LangSAM expects PIL RGB images

    results = []
    label_cache = {}
    for det in detections:
        label = det.get("label", "object")
        x, y = det.get("point", [0, 0])  # [x,y] in COLOR px
        x = float(x); y = float(y)
        # print("image_pil:", type(image_pil), getattr(image_pil, "size", None))
        # print("label:", type(label), label)
        if label not in label_cache:
            try:
                pred = langsam_model.predict([image_pil], [label])   # list of dicts
            except: continue
            res = pred[0]

            masks_l   = res["masks"]
            label_cache[label] = masks_l
        masks_l = label_cache[label]
        if masks_l is None or len(masks_l) == 0:
            continue

        # pick the mask that contains the Gemini point; fallback to first
        xi = int(np.clip(round(x), 0, Wc - 1))
        yi = int(np.clip(round(y), 0, Hc - 1))

        chosen = None
        for m in masks_l:
            m_u8 = (np.asarray(m) > 0).astype(np.uint8)  # (H,W) 0/1
            if m_u8[yi, xi] > 0:
                chosen = m_u8
                break
        if chosen is None:
            chosen = (np.asarray(masks_l[0]) > 0).astype(np.uint8)

        mask = chosen
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask = mask_filter(mask, outline_px=2)

        ys, xs = np.where(mask.astype(bool))
        if xs.size < 80:
            continue

        # map mask pixels to depth and form (u_d, v_d, z_mm)
        ud, vd = xs.astype(np.float32), ys.astype(np.float32)  

        ud_i = np.clip(np.round(ud).astype(np.int32), 0, Wd - 1)
        vd_i = np.clip(np.round(vd).astype(np.int32), 0, Hd - 1)

        z = depth_mm[vd_i, ud_i].astype(np.float32)
        valid = (z >= 200) & (z <= 20000)
        if not np.any(valid):
            continue

        pts_duvz = np.stack([ud_i[valid].astype(np.float32),
                            vd_i[valid].astype(np.float32),
                            z[valid]], axis=1)
        
        if pts_duvz.shape[0] < 200:
            continue
        draw_object_pc_on_color(annotated, pts_duvz, stride=4, radius=1, alpha=0.35)

        # --- OBB in true Euclidean XYZ (depth camera frame) ---
        # IMPORTANT: pts_duvz should use REAL depth pixel indices (ud_i, vd_i) + z, not float ud/vd from C2D (identity matrix now).
        pts_xyz = uvz_to_xyz(pts_duvz, fx, fy, cx, cy)  # mm units
        pts_xyz = pc_filter_xyz_fast(pts_xyz, mad_k=2.5,)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_xyz.astype(np.float64))
        obb = pcd.get_minimal_oriented_bounding_box()
        corners_xyz = np.asarray(obb.get_box_points(), dtype=np.float64)  # (8,3) in XYZ (mm)
        center_xyz = np.asarray(pcd.get_center(), dtype=np.float64)  # (3,) in depth camera XYZ (mm)
        # XYZ -> depth pixel (float)
        u_c, v_c = xyz_to_duv(center_xyz[None, :], fx, fy, cx, cy)
        u_d = float(u_c[0])
        v_d = float(v_c[0])

        # depth pixel -> color pixel
        ud_i = int(round(u_d))
        vd_i = int(round(v_d))
        px, py = float(ud_i), float(vd_i)
        px = float(np.clip(px, 0, Wc - 1))
        py = float(np.clip(py, 0, Hc - 1))

        # use Z from center_xyz
        z_mm = float(center_xyz[2])

        # color_px, color_py, z) -> robot
        center_robot_h = C2R_TRANSFORM_MATRIX @ np.array([px, py, z_mm, 1.0], dtype=np.float64)
        center_robot = (center_robot_h[:3] / center_robot_h[3]).tolist()

        if C2R_np is not None:
            center_xyz_m = center_xyz / 1000.0
            center_robot_np_h = C2R_np @ np.array([center_xyz_m[0], center_xyz_m[1], center_xyz_m[2], 1.0], dtype=np.float64)
            center_robot_np = center_robot_np_h[:3] / center_robot_np_h[3]
            _compare_transforms(f"center:{label}", np.asarray(center_robot, dtype=np.float64), center_robot_np)
            # convert center_robot_np to list
            center_robot_np = center_robot_np.tolist()

        # show_xyz_interactive(pts_xyz, corners_xyz, point_size=2.0)
        u_c, v_c = xyz_to_duv(corners_xyz, fx, fy, cx, cy)  # depth pixel coords (float)

        corners_pxpyz = []
        corners_uv = []
        for (u_d, v_d, z_corner) in zip(u_c, v_c, corners_xyz[:, 2]):
            ud_i = int(round(u_d)); vd_i = int(round(v_d))
            px, py = float(ud_i), float(vd_i)
            px = float(np.clip(px, 0, Wc - 1))
            py = float(np.clip(py, 0, Hc - 1))
            z_mm = float(z_corner) 
            corners_pxpyz.append([px, py, z_mm])
            corners_uv.append([int(round(px)), int(round(py))])
        corners_uv = np.asarray(corners_uv, dtype=np.int32)

        # draw prompt + box + label
        cv2.circle(annotated, (int(round(x)), int(round(y))), 6, (0, 0, 255), -1, lineType=cv2.LINE_AA)
        draw_wireframe(annotated, corners_uv, label)
        draw_uv_indices(annotated, corners_uv)

        results.append({"label": label, "corners": corners_xyz, "pixel_corners": corners_pxpyz, "centers": center_robot_np})
    return results, annotated

def clean_bbox_dict(obbs): 
    C2R_np = load_c2r_matrix()
    corner_dict = {}
    pixel_corner_dict = {}
    for item in obbs:
        label = item.get("label", "object")
        corner_dict[label] = item.get("corners", None)
        pixel_corner_dict[label] = item.get("pixel_corners", None)
    bbox = {}

    for obj, corners in corner_dict.items():
        corners = np.asarray(corners, dtype=np.float64) / 1000.0  # This corner_dict is in camera frame XYZ (mm)
        pixel_corners = np.asarray(pixel_corner_dict[obj], dtype=np.float64)  # (8,3) in dx,dy,dz in camera frame

        if corners.shape != (8, 3):
            continue

        # (px,py,z,1) -> robot
        homo = np.hstack([pixel_corners, np.ones((8, 1), dtype=np.float64)])
        corners_h = (C2R_TRANSFORM_MATRIX @ homo.T).T
        corners_robot = corners_h[:, :3] / corners_h[:, 3:4]  # (8,3) robot XYZ

        if C2R_np is not None:
            homo_np = np.hstack([corners, np.ones((8, 1), dtype=np.float64)])
            corners_np_h = (C2R_np @ homo_np.T).T
            corners_robot_np = corners_np_h[:, :3] / corners_np_h[:, 3:4]
            _compare_transforms(f"corners:{obj}", corners_robot, corners_robot_np)
        # print(obj)
        # print(corners_robot)

        # edges from corner 0 in pixel
        p0 = corners_robot[0]
        e01 = corners_robot[1] - p0
        e02 = corners_robot[2] - p0
        e03 = corners_robot[3] - p0

        # edges from corner 0 in XYZ
        cp0 = corners[0]
        ce01 = corners[1] - cp0
        ce02 = corners[2] - cp0
        ce03 = corners[3] - cp0

        E = np.stack([e01, e02, e03], axis=0)  # (3,3)
        CE = np.stack([ce01, ce02, ce03], axis=0)  # (3,3)
        L = np.linalg.norm(E, axis=1) + 1e-9   # (3,)
        CL = np.linalg.norm(CE, axis=1) + 1e-9   # (3,)

        # pick 2 edges most parallel to XY plane:
        horiz_score = np.abs(E[:, 2]) / L
        idx = np.argsort(horiz_score)  # first two are most horizontal
        i_a, i_b = int(idx[0]), int(idx[1])

        # two horizontal-ish edges
        va, vb = E[i_a], E[i_b]
        cla, clb = float(CL[i_a]), float(CL[i_b])

        # length/width: longer is length
        if cla >= clb:
            v_len, length, width = va, cla, clb
        else:
            v_len, length, width = vb, clb, cla

        # height:
        # the remaining edge length
        i_h = int(idx[2])
        height = float(CL[i_h])

        # angle in robot XY plane from +x
        vx, vy = float(v_len[0]), float(v_len[1])
        angle_ccw = float(np.arctan2(vy, vx))

        if angle_ccw > np.pi / 2:
            angle_ccw -= float(np.pi)
        if angle_ccw <= -np.pi / 2:
            angle_ccw += float(np.pi)

        bbox[obj] = {
            "height": height,
            "width": float(width),
            "length": float(length),
            "angle": float(angle_ccw),
            "corners": corners_robot_np  # corners of object in X,Y,Z
        }

    return bbox


class VisionDetector:
    def __init__(
            self,
            fx, fy, cx, cy, distortion,
            marker_size,
            inpainting_factor=0.1
    ):  
        
        self.camera_intrinsics = np.array([[fx, 0, cx],
                                      [0, fy, cy],
                                      [0, 0, 1]])
        
        self.distortion = np.array(distortion)
        
        self.marker_size = marker_size
        self.inpainting_factor = inpainting_factor
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)


    def estimatePoseSingleMarkers(self, corners):
        """
        Description: Aruco Marker pose estimation.
        Author: M lab
        Date: Jul 31, 2023 at 10:49
        URL: https://github.com/Menginventor/aruco_example_cv_4.8.0
        """
        '''
        This will estimate the rvec and tvec for each of the marker corners detected by:
        corners, ids, rejectedImgPoints = detector.detectMarkers(image)
        corners - is an array of detected corners for each detected marker in the image
        marker_size - is the size of the detected markers
        mtx - is the camera matrix
        distortion - is the camera distortion matrix
        RETURN list of rvecs, tvecs, and trash (so that it corresponds to the old estimatePoseSingleMarkers())
        '''
        marker_points = np.array([[-self.marker_size / 2, self.marker_size / 2, 0],
                                [self.marker_size / 2, self.marker_size / 2, 0],
                                [self.marker_size / 2, -self.marker_size / 2, 0],
                                [-self.marker_size / 2, -self.marker_size / 2, 0]], dtype=np.float32)
        trash = []
        rvecs = []
        tvecs = []
        for c in corners:
            nada, R, t = cv2.solvePnP(
                marker_points,
                c,
                self.camera_intrinsics,
                self.distortion,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
            rvecs.append(R)
            tvecs.append(t)
            trash.append(nada)
        return rvecs, tvecs, trash

    def pose_vectors_to_cart(self, rvecs, tvecs, ids=None):
        """
        Convert OpenCV pose vectors into Cartesian marker poses in the camera frame.
        Returns one pose dict per marker with translation and rotation matrix.
        """
        try:
            C2R = load_c2r_matrix()
        except Exception:
            C2R = None

        camera_poses = []
        robot_poses = []
        if rvecs is None or tvecs is None:
            return camera_poses, robot_poses

        marker_ids = None
        if ids is not None:
            marker_ids = np.asarray(ids).reshape(-1).tolist()

        for idx, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
            r = np.asarray(rvec, dtype=np.float64).reshape(-1, 3)[0]
            t = np.asarray(tvec, dtype=np.float64).reshape(-1, 3)[0]
            R, _ = cv2.Rodrigues(r.reshape(3, 1))

            cam_pose = {
                "x": float(t[0]),
                "y": float(t[1]),
                "z": float(t[2]),
                "rvec": r.tolist(),
                "R": R.tolist(),
            }
            if marker_ids is not None and idx < len(marker_ids):
                cam_pose["id"] = int(marker_ids[idx])
            camera_poses.append(cam_pose)
            if C2R is not None:
                T_c = np.eye(4, dtype=np.float64)
                T_c[:3, :3] = R
                T_c[:3, 3] = t

                T_r = C2R @ T_c
                robot_R = T_r[:3, :3]
                robot_t = T_r[:3, 3]
                robot_r, _ = cv2.Rodrigues(robot_R)

                robot_pose = {
                    "x": float(robot_t[0]),
                    "y": float(robot_t[1]),
                    "z": float(robot_t[2]),
                    "robot_rvec": robot_r.reshape(3).tolist(),
                    "robot_R": robot_R.tolist(),
                }
                if marker_ids is not None and idx < len(marker_ids):
                    robot_pose["id"] = int(marker_ids[idx])
                robot_poses.append(robot_pose)
        return camera_poses, robot_poses



    def plot_aruco(self, image, rotation_matrix = None): # rot in case you want AruCO frames to align otherwise
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(self.aruco_dict, parameters)

        # Detect the markers
        corners, ids, rejected = detector.detectMarkers(gray)
        frame_markers = cv2.aruco.drawDetectedMarkers(image.copy(), corners)

        rvecs = []
        tvecs = [] 
        for i in range(len(corners)):
            rvec_list, tvec_list, _ = self.estimatePoseSingleMarkers(corners[i])
            rvec = np.asarray(rvec_list[0], dtype=np.float64).reshape(3, 1)
            tvec = np.asarray(tvec_list[0], dtype=np.float64).reshape(3, 1)

            if rotation_matrix is not None:
                R, _ = cv2.Rodrigues(rvec)
                R_new = R @ rotation_matrix  
                rvec, _ = cv2.Rodrigues(R_new)

            frame_markers = cv2.drawFrameAxes(
                frame_markers.copy(),
                self.camera_intrinsics,
                np.zeros(5),
                rvec,
                tvec,
                0.025
            )
            rvecs.append(rvec)
            tvecs.append(tvec)

        return rvecs, tvecs, corners, ids, frame_markers

    def aruco_inpainting(self, image, base_image):
        # Get the base image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(self.aruco_dict, parameters)
        # Detect the markers
        corners, ids, rejected = detector.detectMarkers(gray)

        rvecs = []
        tvecs = []
        for i in range(len(corners)): 
            new_corner = self.mask_extension(corners[i][0], 0.3, 0.3)
            curr_points = np.array(new_corner, dtype = np.int32)
            mask = np.zeros_like(image)
            cv2.fillConvexPoly(mask, curr_points, (255, 255, 255))
            base_region = cv2.bitwise_and(base_image, mask)
            mask_inv = cv2.bitwise_not(mask)
            image_bg = cv2.bitwise_and(image, mask_inv)
            image = cv2.add(image_bg, base_region)

            rvec_list, tvec_list, _ = self.estimatePoseSingleMarkers(corners[i])
            rvec = np.asarray(rvec_list[0], dtype=np.float64).reshape(3, 1)
            tvec = np.asarray(tvec_list[0], dtype=np.float64).reshape(3, 1)
            rvecs.append(rvec)
            tvecs.append(tvec)

        return rvecs, tvecs, corners, ids, image

    def cv2_aruco_inpainting(self, image, inpainting_radius):
        # This method uses the inpainting method provided by the cv2 library
        # Get the base image
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(self.aruco_dict, parameters)
        # Detect the markers
        corners, ids, rejected = detector.detectMarkers(image)

        rvecs = []
        tvecs = []
        for i in range(len(corners)):
            new_corner = self.mask_extension(corners[i][0], self.inpainting_factor, self.inpainting_factor)
            curr_points = np.array(new_corner, dtype = np.int32)
            mask = np.zeros_like(image)
            cv2.fillConvexPoly(mask, curr_points, (255, 255, 255))
            mask = np.transpose(mask, (2, 0, 1))
            mask = mask[0]
            image = cv2.inpaint(image, mask, inpainting_radius, cv2.INPAINT_NS)

            rvec_list, tvec_list, _ = self.estimatePoseSingleMarkers(corners[i])
            rvec = np.asarray(rvec_list[0], dtype=np.float64).reshape(3, 1)
            tvec = np.asarray(tvec_list[0], dtype=np.float64).reshape(3, 1)
            rvecs.append(rvec)
            tvecs.append(tvec)

        return rvecs, tvecs, corners, ids, image


    def mask_extension(self, corner, extention_x, extention_y):
        vector_x = corner[0] - corner[1]
        vector_y = corner[0] - corner[3]
        new_corner = np.zeros_like(corner)
        new_corner[0] = corner[0] + vector_x* extention_x + vector_y* extention_y
        new_corner[1] = corner[1] - vector_x* extention_x + vector_y* extention_y
        new_corner[2] = corner[2] - vector_x* extention_x - vector_y* extention_y
        new_corner[3] = corner[3] + vector_x* extention_x - vector_y* extention_y

        return new_corner