import cv2

from utils_seed.filehandling import LoadFilesWithExtensions

from SeedDistanceEstimator import *
import sys
EFFICIENTDET_PATH = "/home/harsha/Desktop/SLS-CNH/ObjectDetectionModels/efficientdet-pytorch/"
sys.path.insert(0, EFFICIENTDET_PATH)

from effdet.factory import create_model_from_config
from effdet.config.model_config import efficientdet_model_param_dict
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.efficientdet import HeadNet
from effdet.config.model_config import efficientdet_model_param_dict
import os
import torch


def create_model(num_classes=1, image_size=512, architecture="tf_efficientdet_d0", weights_path=''):

    config = get_efficientdet_config(architecture)
    config.update({'num_classes': num_classes})
    config.update({'image_size': (image_size, image_size)})

    print(config)

    net = create_model_from_config(config, bench_task='predict', num_classes=1, checkpoint_path=weights_path)
    return net


class EfficientDetSeedDetector(SeedDetector):

    def __init__(self, weights_path, img_size=512, threshold=0.5, arch='tf_efficientdet_d0', device = 0):
        super(EfficientDetSeedDetector, self).__init__(name="EfficientDet", threshold=threshold, model=None)
        self.model = create_model(num_classes=1, image_size=512, architecture=arch, weights_path=weights_path)
        self.img_size = img_size
        self.device = device


    def predict(self, input_img:np.array):
        img = self.adjust_size(input_img)
        logging.debug("[Efficientdet ] img.shape {}, redux {}, padding w {}, padding h {}".format(img.shape, self.r, self.pw, self.ph))

        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = img/255.0

        tensors = torch.from_numpy(img)
        tensors =tensors.unsqueeze(0).type(torch.FloatTensor).to(self.device)
        logging.debug("[Tensors Shape] ts = {}".format(tensors.shape))
        _, self.height, self.width = tensors[0].size()

        self.model.eval()
        self.model.to(self.device)
        '''ISSUE WITH PADDING'''
        with torch.no_grad():
            outs = self.model(tensors.cuda())
            logging.debug("[EfficientDet] preds : {}".format(outs))

        return self.get_bboxs(preds=outs)
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
        scaled_bboxs = preds*torch.tensor([1/self.r, 1/self.r, 1/self.r, 1/self.r, 1, 1]).to(self.device) + torch.tensor([-self.pw[0], -self.ph[0], -self.pw[0], -self.ph[0], 0, 0]).to(self.device)
        return scaled_bboxs


    def get_max_prob_seeds(self, predictions, n=3):
        scores = predictions[0][:, -2]
        size = scores.size()[0]
        if size<n:
            scores = torch.cat((scores, torch.zeros(n-size)))
        return scores[:n].cpu()

    def filter(self, predictions):
        ret = predictions[0][predictions[0][:, -2] > self.threshold]
        return ret

    def convert(self, predictions):
        seeds_camera_space = []
        seeds, _ = predictions.shape
        for seed in range(seeds):
            x, y, h, w = self.convertToImgSpace(predictions[seed, :])
            logging.debug("[convert] Box {} converted into {}".format(predictions[seed, :], (x, y, h, w)))

            seeds_camera_space.append(SeedCamSpace(x.cpu(), y.cpu(), h.cpu(), w.cpu(), predictions[seed, -2].cpu()))
        return seeds_camera_space


    def convertToImgSpace(self, pred):
        box = pred[:4]
        x = (box[2] + box[0])/2
        y = (box[1] + box[3])/2
        w = (box[2] - box[0])
        h = (box[3] - box[1])
        return x, y, h, w

weights_path = "/home/harsha/Desktop/SLS-CNH/ObjectDetectionModels/efficientdet-pytorch/output/train/20220110-133350-tf_efficientdet_d0/model_best.pth.tar"
detector = EfficientDetSeedDetector( weights_path=weights_path, threshold=0.3, img_size=512)

if __name__ == "__main__":
    weights_path = "/home/harsha/Desktop/SLS-CNH/ObjectDetectionModels/efficientdet-pytorch/output/train/20220110-133350-tf_efficientdet_d0/model_best.pth.tar"
    EfficientDetDetector = EfficientDetSeedDetector( weights_path=weights_path, threshold=0.5, img_size=512)
    dataset = "/home/harsha/Desktop/dataset/seed_detection_yolo5/test/images"
    imgs = LoadFilesWithExtensions(dataset, ["JPEG"])
    for img in imgs:
        input = cv2.imread(img)
        pred = EfficientDetDetector.predict(input)
        print(pred)
    print(imgs)