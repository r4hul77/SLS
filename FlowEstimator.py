import copy

import cv2 as cv
import numpy as np
import scipy
import logging

class FlowEstimator:

    def __init__(self, queue_size, debug):
        self.queue_size = queue_size
        self.debug = debug
        self.state = np.zeros((6,1))
        self.queue = []
        self.sift = cv.SIFT_create(nOctaveLayers=8, contrastThreshold=0.1, sigma=0.6)

        self.matcher = cv.BFMatcher()

    def main(self, frame):

        if len(self.queue) == self.queue_size:
            self.queue.pop(0)

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        kp, des = self.sift.detectAndCompute(gray, None)
        self.queue.append((frame, gray, kp, des))

        if(len(self.queue)  == self.queue_size):
            matches = self.find_matches()
        else:
            matches = [[], []]
        logging.debug("FlowEstimator Found {} Key Points".format(len(kp)))
        logging.debug("[FlowEstimator] Key Points {} ".format(kp))
        logging.debug("[Flow Estimator] Des {}".format(des))

        if self.debug:
            return self.state, self.viz(matches)
        else:
            return self.state

    def find_matches(self):
        return self.match(self.queue[1], self.queue[-1])

    def match(self, obj0, obj1):
        frame0, gray0, kp0, des0 = obj0
        frame1, gray1, kp1, des1 = obj1
        matches = []
        if(des0 is not None and des1 is not None):
            matches = self.matcher.knnMatch(des0, des1, k=1)
        good = []
        for m in matches:
           if kp0[m[0].queryIdx].pt[0] > kp1[m[0].trainIdx].pt[0] and (kp0[m[0].queryIdx].pt[1] - kp1[m[0].trainIdx].pt[1])**2 < 500:
                good.append(m[0])
        return good


    def draw_matches(self, matches, frame, train_kps, query_kps, color):
        img = copy.copy(frame)
        for match in matches:
            start_point = [int(c) for c in query_kps[match.queryIdx].pt]
            end_point = [int(c) for c in train_kps[match.trainIdx].pt]
            cv.arrowedLine(img, start_point, end_point, color, thickness=1)
        return img

    def viz(self, matches):
        frame, gray, kp, des = self.queue[-1]
        img = cv.drawKeypoints(frame, kp, frame)
        if(len(self.queue)==3):
            img = self.draw_matches(matches, img, self.queue[-1][2], self.queue[1][2], (128, 0, 128))

        return img
