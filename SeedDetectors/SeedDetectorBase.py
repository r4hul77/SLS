import logging

import numpy as np
from utils_seed.datatypes import *

class SeedDetector:
    '''This Class is responsible for taking in the img input and converting into seed predictions in Camera Frame'''
    def __init__(self, name, threshold, model):
        self.name = name
        self.threshold = threshold
        self.model = model


    def detect(self, input_img:np.array, seed_prob_dim=0) -> SeedList:
        '''Returns Predictions in Cam Space with a seed_prob_dim'''
        logging.debug("[Seed Detector Base] Img recived size : {}".format(input_img.shape))
        preds = self.predict(input_img)
        logging.debug("[Seed Detector Base] Predictions {}".format(preds))
        filtered = self.filter(preds)
        logging.debug("[Seed Detector Base] Filtered Predictions {}".format(filtered))
        filtred_detections = self.convert(filtered)
        logging.debug("[Seed Detector Base] Converted Detections Image Space {}".format(filtred_detections))

        ret = (filtred_detections, self.get_max_prob_seeds(preds, n=seed_prob_dim))

        return ret



    def get_max_prob_seeds(self, predictions, n):
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

