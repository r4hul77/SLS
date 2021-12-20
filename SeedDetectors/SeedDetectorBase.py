import numpy as np
from utils.datatypes import *

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
