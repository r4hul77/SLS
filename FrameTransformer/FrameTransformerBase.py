import numpy as np
from utils_seed.datatypes import *
import typing
import copy

#Sensor Size = 11.3 mm x 7.1 mm
#Pixel Size = 5.86 µm x 5.86 µm

class FrameTransformerBase:

    def __init__(self, values):
        self.vals = values

    def convert_to_camspace(self, seed, width, height):
        seed_cam_frame = copy.copy(seed)
        seed_cam_frame.x_c = (seed.x_c /width) * 0.13335
        seed_cam_frame.y_c = (seed.y_c /  height) * 0.13335
        seed_cam_frame.w =  (seed.w/ width) * 0.13335
        seed_cam_frame.h = (seed.h/height) * 0.13335
        return seed_cam_frame

    def convert_to_img_space(self, seed, width, height):
        seed_cam_frame = copy.copy(seed)
        seed_cam_frame.x_c = (seed.x_c /0.13335) * width
        seed_cam_frame.y_c = (seed.y_c /  0.13335) * height
        seed_cam_frame.w =  (seed.w/ 0.13335) * width
        seed_cam_frame.h = (seed.h/0.13335) * height
        return seed_cam_frame