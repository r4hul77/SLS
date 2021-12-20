import copy
import dataclasses

import numpy as np
import utm
import logging

from typing import List
from utils.datatypes import *
from SeedDetectors.SeedDetectorBase import *

class SeedXDistanceEstimator:

    def __init__(self):
        self.res_dist = 0
        self.count    = 0
        self.first_seed = False

    def main(self, filtred_seeds:SeedList, lat, long, vel, accel, dt, heading=0, frame_idx = 0):

        self.res_dist += self.first_seed*(vel*dt + 0.5*accel*dt*dt)
        if(len(filtred_seeds)==0):
            return []
        self.first_seed = True
        ret = []
        distance    = self.res_dist + filtred_seeds[0].x_c
        for i, seed in enumerate(filtred_seeds):
            self.count += 1
            seed_utm = self.convertToUTM(seed, lat, long, heading)
            if(i>0):
                distance = seed.x_c - filtred_seeds[i-1].x_c
            logging.debug("[DistanceEstimator] distance {} Count {}".format(distance, self.count))
            ret.append(SeedDistStruct(seed_utm, self.count, distance, frame_idx))
        self.res_dist = -filtred_seeds[-1].x_c
        return ret

    def convertToUTM(self, seed: SeedCamSpace, lat, long, heading=0):
        e, n , z, zone_letter = utm.from_latlon(lat, long)
        heading_rad = np.pi*heading/180
        n += seed.x_c*np.cos(heading_rad) - seed.y_c*np.sin(heading_rad)
        e += seed.y_c*np.cos(heading_rad) + seed.x_c*np.sin(heading_rad)
        return SeedUTMSpace(n, e, z, zone_letter, seed.h, seed.w, seed.p)

if __name__ == "__main__":
    seed_pred1 = SeedCamSpace(10, 10, 5, 5, 0.8)
    seed_pred2 = SeedCamSpace(8, 8, 3, 3, 0.8)
    model = lambda x: x
    null = SeedDetector("Test", 0.5, model)
    seed_filter = SeedFilter(null)
    print(seed_filter.get_iou(seed_pred1, seed_pred2))