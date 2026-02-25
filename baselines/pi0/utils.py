import cv2 
import torch
import numpy as np
import torch.nn.functional as F

def preprocess_video(video: np.ndarray, specs: dict) -> np.ndarray:
    # video of shape T H W C
    image = False
    try:
        assert video.ndim == 4 and video.shape[-1] == 3, "Video should be of shape (T, H, W, C)"
    except AssertionError:
        if video.ndim == 3:
            video = np.expand_dims(video, axis=0)
            image = True
        else:
            raise ValueError("Video should be of shape (T, H, W, C) or (H, W, C) for a single image")
        
    if 'cvtColor' in specs:
        tmp_video = []
        if specs['cvtColor'] == 'RGB2BGR':
            for frame in video: # change this
                tmp_video.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        elif specs['cvtColor'] == 'BGR2RGB':
            for frame in video:
                tmp_video.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        video = np.array(tmp_video)

    if 'resize' in specs:
        video = torch.tensor(video).permute(0, 3, 1, 2).float()
        video = F.interpolate(video, size=tuple(specs['resize']), mode='bilinear')
        video = video.permute(0, 2, 3, 1).numpy()

    if image:
        video = video.squeeze(0)
    
    return video

class ObservationBuffer:
    def __init__(self, 
                 buffer_size: int = 4,
                 init_obs: dict = None) -> None:
        self.buffer_size = buffer_size
        self.buffer = {k: [v]*buffer_size for k, v in init_obs.items()}

    def add(self, obs_dict:dict) -> None:
        for k, v in self.buffer.items():
            v.pop(0)
            v.append(obs_dict[k])

    def get_buffer(self) -> dict:
        return {k: np.array(v) for k, v in self.buffer.items()}

    def clear(self):
        self.buffer = {}