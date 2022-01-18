from SeedDistanceEstimator import *
import torch
import logging
from SeedDetectors.SeedDetectorBase import *

class RetinaNetSeedDector(SeedDetector):

    def __init__(self, model_path, threshold, device = 0):
        super(RetinaNetSeedDector, self).__init__(name="Retina_Net", threshold=threshold, model=torch.load(model_path))
        self.device = device
    def predict(self, input_img:np.array):
        tensors = [ torch.Tensor(input_img[::-1] / 255).to(self.device).permute(2, 0, 1)]
        _, self.height, self.width = tensors[0].size()
        self.model.eval()

        with torch.no_grad():
            outs = self.model(tensors)
        ret = []
        logging.debug("[RetinaNetDetector] outs : {}".format(outs))
        return outs

    def get_max_prob_seeds(self, predictions, n=3):
        scores = [dict_obj["scores"].cpu() for dict_obj in predictions]
        ret = scores[0]
        size = ret.size(dim=0)
        if size<n:
            ret = torch.cat((ret, torch.zeros(n-size)))
        return ret[:n]

    def filter(self, predictions):
        ret = []
        for dict_obj in predictions:
            ret_dict = {}
            scores = torch.gt(dict_obj["scores"], self.threshold)
            ret_dict["scores"] = dict_obj["scores"][scores]
            ret_dict["labels"] = dict_obj["labels"][scores]
            ret_dict["boxes"]  = dict_obj["boxes"][scores]
            ret.append(ret_dict)
        logging.debug("[RetinaNetDetector] rets : {}".format(ret))
        return ret

    def convert(self, predictions):
        seeds_camera_space = []
        for i, box in enumerate(predictions[0]["boxes"]):
            x, y, h, w = self.convertToCamSpaceFromImgSpace(box)
            logging.debug("[convert] Box {} converted into {}".format(box, (x,y, h, w)))

            seeds_camera_space.append(SeedCamSpace(x, y, h, w, predictions[0]["scores"][i]))
        return seeds_camera_space

    def convertToCamSpaceFromImgSpace(self, box):
        box = box.cpu()
        x = (box[2] + box[0])/(2*self.width) * 0.13335
        y = (box[1] + box[3])/(2*self.height) * 0.13335
        w = (box[2] - box[0])/(self.width)*0.13335
        h = (box[3] - box[1])/(self.height)*0.13335
        return x, y, h, w
