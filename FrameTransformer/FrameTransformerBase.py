import numpy as np
from utils_seed.datatypes import *
import typing
import copy

class FrameTransformerBase:

    def __init__(self, values):
        self.vals = values

    def convert(self, seed, width, height):
        seed_cam_frame = copy.copy(seed)
        seed_cam_frame.x_c = (seed.x_c /width) * 0.13335
        seed_cam_frame.y_c = (seed.y_c /  height) * 0.13335
        seed_cam_frame.w =  (seed.w/ width) * 0.13335
        seed_cam_frame.h = (seed.h/height) * 0.13335
        return seed_cam_frame