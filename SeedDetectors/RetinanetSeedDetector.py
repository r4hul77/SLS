from SeedDistanceEstimator import *
import torch
import logging
from SeedDetectors.SeedDetectorBase import *

class RetinaNetSeedDector(SeedDetector):

    def __init__(self, model_path, threshold, device = 0):
        super(RetinaNetSeedDector, self).__init__(name="Retina_Net", threshold=threshold, model=torch.load(model_path))
        self.device = device
    def predict(self, input_img:np.array):
        tensors = [ torch.Tensor(input_img[:, :, ::-1] / 255).to(self.device).permute(2, 0, 1)]
        for tensor in tensors:
            logging.debug("[RetinaNet Detector] Predict tensor shapes {}".format(tensor.shape))
            logging.debug("[Shape of detector] input_img[::-1] shape {}, input_img[:, :, ::-1] shape {}, sum {}".format(input_img[::-1].shape, input_img[::-1].shape, input_img[::-1] - input_img[:, :, ::-1]))
        _, self.height, self.width = tensors[0].size()
        self.model.eval()

        with torch.no_grad():
            outs = self.model(tensors)
        ret = []
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
        return ret

    def convert(self, predictions):
        seeds_camera_space = []

        for i, box in enumerate(predictions[0]["boxes"]):
            x, y, h, w = self.convertToImgSpace(box)
            seeds_camera_space.append(SeedCamSpace(x, y, h, w, predictions[0]["scores"][i]))

        return seeds_camera_space

    def convertToImgSpace(self, pred):
        box = pred[:4].cpu()
        x = (box[2] + box[0])/2
        y = (box[1] + box[3])/2
        w = (box[2] - box[0])
        h = (box[3] - box[1])
        return x, y, h, w

model_path = "/home/harsha/Desktop/SLS-CNH/TrainedWeights/retina_fp_18_classified_allTrain2.pth"

detector = RetinaNetSeedDector(model_path=model_path, threshold=0.6)