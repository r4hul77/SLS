import cv2
import copy
import numpy as np
from utils_seed.datatypes import *
import typing
import logging

def draw_bbox(img:np.array, seeds:SeedList):
    ret = copy.copy(img)
    h,w, _ = img.shape
    for seed in seeds:
        logging.debug("[Draw BBOX] seed recvd {}".format(seed))
        start_point = (int(seed.x_c - seed.w/2),  int(seed.y_c - seed.h/2))
        end_point = (int(seed.x_c + seed.w/2),  int(seed.y_c + seed.h/2))
        ret = cv2.rectangle(ret, start_point, end_point, (255,0, 0), 1)
        ret = cv2.putText(ret, "{0:.2f}".format(seed.p.item()), org=start_point, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 255), thickness=1)
        logging.debug("[Draw BBOX] seed start point {} end point {}".format(start_point, end_point))
    return ret