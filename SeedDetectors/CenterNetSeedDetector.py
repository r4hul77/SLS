import cv2

from utils_seed.filehandling import LoadFilesWithExtensions

from SeedDistanceEstimator import *
import sys
CENTERNET_PATH = "/home/harsha/Desktop/SLS-CNH/ObjectDetectionModels/CenterNet/src/lib"
sys.path.insert(0, CENTERNET_PATH)

from detectors.detector_factory import detector_factory
from models.model import create_model, load_model, save_model
from datasets.dataset_factory import get_dataset

import os
from opts import opts
import torch

class CenterNetSeedDector(SeedDetector):

    def __init__(self, arch, weights_path, img_size, threshold, device = 0):
        super(CenterNetSeedDector, self).__init__(name="CenterNet", threshold=threshold, model=None)
        self.device = device
        self.img_size = img_size
        TASK = 'ctdet'  # or 'multi_pose' for human pose estimation
        opt = opts().init('{} --arch {} --dataset seed_spacing --load_model {}'.format(TASK, arch, weights_path).split(' '))
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
        opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
        Dataset = get_dataset(opt.dataset, opt.task)
        opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
        TASK = 'ctdet'  # or 'multi_pose' for human pose estimation
        self.model = detector_factory[opt.task](opt)

    def predict(self, input_img:np.array):
        logging.debug("[CenterNet detector ] img.shape {}".format(input_img.shape))
        preds = self.model.run(input_img)['results']
        '''ISSUE WITH PADDING'''

        logging.debug("[CenterNetDetector] preds : {}".format(preds))

        return preds

    def adjust_size(self, img):
        h, w, _ = img.shape
        max_dim = max(w, h)
        self.r = max_dim/self.img_size
        if(self.r > 1):
            img = cv2.resize(img, dsize=(int(1/self.r*w), int(1/self.r*h)))
        else:
            self.r = 1
        logging.debug("[adjust size] resized image sizes {} when r is {}".format(img.shape, self.r))
        img = self.add_padding(img)
        return img


    def add_padding(self, img):
        h, w, _ = img.shape
        dw, dh = max(self.img_size - w, 0), max(self.img_size - h, 0) #To remove negetive padding
        self.pw, self.ph = (int(np.floor(dw/2)), int(np.ceil(dw/2))), (int(np.floor(dh/2)), int(np.ceil(dh/2)))
        logging.debug("[Add padding] pw, pwh = {}, {}".format(self.pw, self.ph))
        ret = cv2.copyMakeBorder(img, self.ph[0], self.ph[1], self.pw[0], self.pw[1], cv2.BORDER_CONSTANT, value=0)
        return ret

    def get_bboxs(self, preds):
        scaled_bboxs = {}
        for key in preds.keys():
            scaled_bboxs[key] = preds[key]*np.array([1/self.r, 1/self.r, 1/self.r, 1/self.r, 1]) + np.array([-self.pw[0], -self.ph[0], -self.pw[0], -self.ph[0], 0])
        return scaled_bboxs


    def get_max_prob_seeds(self, predictions, n=3):
        scores = predictions[1][:, -1]
        size = scores.size
        if size<n:
            scores = torch.cat((scores, torch.zeros(n-size)))
        return scores[:n]

    def filter(self, predictions):
        ret = predictions[1][predictions[1][:, -1] > self.threshold]
        return ret

    def convert(self, predictions):
        seeds_camera_space = []
        seeds, _ = predictions.shape
        for seed in range(seeds):
            x, y, h, w = self.convertToImgSpace(predictions[seed, :])
            logging.debug("[convert] Box {} converted into {}".format(predictions[seed, :], (x, y, h, w)))

            seeds_camera_space.append(SeedCamSpace(x, y, h, w, predictions[seed, -1]))
        return seeds_camera_space


    def convertToImgSpace(self, pred):
        box = pred[:4]
        x = (box[2] + box[0])/2
        y = (box[1] + box[3])/2
        w = (box[2] - box[0])
        h = (box[3] - box[1])
        return x, y, h, w

weights_path = "/home/harsha/Desktop/SLS-CNH/ObjectDetectionModels/CenterNet/exp/ctdet/seeds_center_net_r18/model_best.pth"
detector = CenterNetSeedDector(arch="res_18", weights_path=weights_path, threshold=0.24, img_size=512)

if __name__ == "__main__":
    weights_path = "/home/harsha/Desktop/SLS-CNH/ObjectDetectionModels/CenterNet/exp/ctdet/seeds_center_net_r18/model_best.pth"
    CenterNetDetector = CenterNetSeedDector(arch="res_18", weights_path=weights_path, threshold=0.5, img_size=512)
    dataset = "/home/harsha/Desktop/dataset/seed_detection_yolo5/test/images"
    imgs = LoadFilesWithExtensions(dataset, ["JPEG"])
    for img in imgs:
        input = cv2.imread(img)
        pred = CenterNetDetector.predict(input)
        print(pred)
    print(imgs)