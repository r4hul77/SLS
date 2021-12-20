import copy
import dataclasses

import numpy as np
import utm
import logging

from typing import List


@dataclasses.dataclass
class SeedCamSpace:
    x_c: float #X center
    y_c: float #Y Center
    h  : float #Height
    w  : float #Width
    p  : float #Probality

SeedList = List[SeedCamSpace]

@dataclasses.dataclass
class SeedUTMSpace:
    n_c  : float
    e_c  : float
    zone : int
    zone_letter : str
    h    : float
    w    : float
    p    : float

class SeedDetector:
    '''This Class is responsible for taking in the img input and converting into seed predictions in Camera Frame'''
    def __init__(self, name, threshold, model):
        self.name = name
        self.threshold = threshold
        self.model = model


    def detect(self, input_img:np.array, seed_prob_dim=None) -> SeedList:
        preds = self.predict(input_img)
        filtered = self.filter(preds)
        ret = self.convert(filtered)

        if(seed_prob_dim):
            return ret, self.get_max_prob_seeds(preds, seed_prob_dim)

        return ret


    def get_max_prob_seeds(self, ret, n):
        return None

    def filter(self, predictions):
        print("Filtering the Threshold Not Implimented")
        return predictions

    def convert(self, predictions):
        # ToDo Impliment the abstract Method
        print("Convert Not Implimented")
        return []

    def predict(self, input_img:np.array):
        return self.model(input_img)


class SeedFilter:

    def __init__(self, seed_detector:SeedDetector, iou_threshold:float):
        self.seed_detector = seed_detector
        self.seed_queue : SeedList  = []
        self.iou_threshold = iou_threshold

    def main(self, frame, lat, long, velocity, acceleration, dt, seed_prob_dim=None):



        self.update(lat, long, velocity, acceleration, dt)

        logging.debug("[Filter Main] qeue {}".format(self.seed_queue))

        detectionsCam, max_prob = self.seed_detector.detect(frame, seed_prob_dim)

        detectionsCam.sort(key=lambda x:x.x_c)

        new_seeds = self.filter(detectionsCam, self.iou_threshold)

        for seed in new_seeds:
            self.seed_queue.append(seed)

        if(seed_prob_dim):
            return new_seeds, max_prob

        return new_seeds


    def update(self, lat, long, velocity, acceleration, dt):

        new_list = []

        for seed in self.seed_queue:
            seed.x_c -= velocity*dt + 0.5*acceleration*dt*dt



            if(seed.w//2+seed.x_c > -0.1):
                new_list.append(seed)

            else:
                logging.debug("[Filter Update] Rejected Seed {}".format(seed))

        logging.debug("New List Size {}".format(len(new_list)))
        self.seed_queue = new_list

    def get_iou(self, seed_0, seed_1):

        logging.debug("[get_iou] seed0:{}, seed1:{}".format(seed_0, seed_1))

        def get_x1x2(seed):
            return seed.x_c-seed.w/2, seed.x_c + seed.w*0.5

        def get_y1y2(seed):
            return seed.y_c-seed.h/2, seed.y_c + seed.h*0.5

        x00, x01 = get_x1x2(seed_0)
        x10, x11 = get_x1x2(seed_1)
        y00, y01 = get_y1y2(seed_0)
        y10, y11 = get_y1y2(seed_1)


        xi0, xi1 = max(x00, x10), min(x01, x11)
        yi0, yi1 = max(y00, y10), min(y01, y11)

        logging.debug("[get_iou] (x00, y00), (x01, y01): {}, {}".format((x00, y00), (x01, y01)))


        logging.debug("[get_iou] (x10, y10), (x11, y11): {}, {}".format((x10, y10), (x11, y11)))


        I = max(xi1 - xi0 + 0.13335, 0)*max(yi1 - yi0 + 0.13335, 0)

        logging.debug("[get_iou] I {}".format(I))

        xu0, xu1 = min(x00, x10), max(x01, x11)
        yu0, yu1 = min(y00, y10), max(y01, y11)

        U = (xu1 - xu0 + 0.13335)*(yu1 - yu0 + 0.13335)

        logging.debug("[get_iou] U {}".format(U))

        iou = I/U

        return iou

    def filter(self, detections, iou_tres):
        new_seeds = []
        if(len(self.seed_queue) > 0):
            seed_queue = self.seed_queue
            seeds_to_be_added = []
            for seed in detections:
                reject = False
                #if(seed.x_c < self.seed_queue[-1].x_c):
                #    continue
                for i, seed_q in enumerate(seed_queue):
                    iou = self.get_iou(seed_q, seed)
                    logging.debug("[IOU] iou : {}".format(iou))
                    if(iou>iou_tres):
                        reject = True
                        seed_queue.remove(seed_q)
                        seeds_to_be_added.append(seed)
                        break
                if(not reject):
                    new_seeds.append(seed)

            for seed in seeds_to_be_added:
                seed_queue.append(seed)

            self.seed_queue = seed_queue

        else:
            new_seeds = detections

        return new_seeds



@dataclasses.dataclass
class SeedDistStruct:
    seed     : SeedUTMSpace
    count    :    int
    distance : float
    frame_idx : int

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