import numpy as np
import utm
import logging

from typing import List
from utils_seed.datatypes import *
from SeedDetectors.SeedDetectorBase import *

class SeedFilter:

    def __init__(self, iou_threshold:float):
        self.seed_queue : SeedList  = []
        self.iou_threshold = iou_threshold

    def main(self, detections, lat, long, velocity, acceleration, dt):

        logging.debug("[Filter Main] Recived Detections {}".format(detections))

        self.update(lat, long, velocity, acceleration, dt)

        logging.debug("[Filter Main] qeue {}".format(self.seed_queue))

        detections.sort(key=lambda x:x.x_c)

        new_seeds = self.filter(detections, self.iou_threshold)
        for seed in new_seeds:
            self.seed_queue.append(seed)

        return new_seeds


    def update(self, lat, long, velocity, acceleration, dt):

        new_list = []

        for seed in self.seed_queue:
            seed.x_c -= velocity*dt + 0.5*acceleration*dt*dt



            if(seed.w / 2+seed.x_c > -0.1):
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

class SeedFilterCenterBased(SeedFilter):

    def __int__(self):
        super(SeedFilter, self).__init__(0)

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
                    flag = self.center_match(seed_q, seed)
                    logging.debug("[Flag] Flag : {}".format(flag))
                    if(flag):
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

    def center_match(self, seed, seed_q):
        return ((((seed.x_c - seed.w / 2) < seed_q.x_c) and ((seed.x_c+seed.w / 2) > seed_q.x_c))) #and
                #(((seed.y_c - seed.h / 2) < seed_q.y_c) and ((seed.y_c + seed.h / 2) > seed_q.y_c)))

