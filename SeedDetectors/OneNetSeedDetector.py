import cv2

from utils_seed.filehandling import LoadFilesWithExtensions

from SeedDistanceEstimator import *
import sys
ONENET_PATH = "/home/harsha/Desktop/SLS-CNH/ObjectDetectionModels/OneNet/"
sys.path.insert(0, ONENET_PATH)
from detectron2.engine.defaults import DefaultPredictor
import argparse
from detectron2.config import get_cfg
from projects.OneNet.onenet import add_onenet_config


import torch

class OneNetSeedDector(SeedDetector):

    def __init__(self, cfg, threshold, device = 0, args='', checkpoint=''):
        super(OneNetSeedDector, self).__init__(name="OneNet", threshold=threshold, model=None)
        self.device = device
        self.predictor = DefaultPredictor(cfg)


    def predict(self, input_img:np.array):
        image = input_img[:, :, ::-1]
        logging.debug("[OneNet Detector] Predict input shapes {}".format(image.shape))

        preds = self.predictor(image)

        logging.debug("[OneNEt Detector] Output {}".format(preds['instances']))
        #
        # vals = outs['pred_logits'].softmax(-1)[0, :, :-1]
        ret = {}
        ret['scores'], ret['labels'] = preds['instances'].scores, preds['instances'].pred_classes
        ret['boxes']  = preds['instances'].pred_boxes.tensor
        logging.debug('Ret {}'.format(ret))
        return ret

    def box_cxcywh_to_xyxy(self, x):
        x_c, y_c, w, h = x.unbind(1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
             (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)

    def rescale_bboxes(self, out_bbox):
        img_w, img_h = self.height, self.width
        b = self.box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).to(self.device)
        return b


    def get_max_prob_seeds(self, predictions, n=3):
        ret = predictions["scores"]
        size = ret.size(dim=0)
        if size<n:
            ret = torch.cat((ret, torch.zeros(n-size)))
        return ret[:n]

    def filter(self, predictions):
        ret = []

        scores = torch.gt(predictions["scores"], self.threshold)
        ret_dict = {}
        ret_dict["scores"] = predictions["scores"][scores]
        ret_dict["labels"] = predictions["labels"][scores]
        ret_dict["boxes"]  = predictions["boxes"][scores]
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

cfg_path = "/home/harsha/onenet_outputs/output_onenet_r50fcos/config.yaml"
cfg = get_cfg()
add_onenet_config(cfg)
cfg.merge_from_file(cfg_path)
detector = OneNetSeedDector(cfg=cfg, threshold=0.2)