"""
uses langsam and yolov11 to identify the location of objects in the scene
"""
from .lang_segment_anything.lang_sam import LangSAM
from ultralytics import YOLO

class OpenVocabClassifier:
    def __init__(self, path_to_weights: str):
        self.langsam = LangSAM(sam_type="sam2.1_hiera_large")
        self.yolo = YOLO(path_to_weights)
        return
    