import cv2
import sys
import torch

YOLOX_PATH = "/home/harsha/Desktop/SLS-CNH/ObjectDetectionModels/YOLOX"
sys.path.insert(0, YOLOX_PATH)

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis
from tools.demo import Predictor



from SeedDetectors.SeedDetectorBase import *

from utils_seed.filehandling import *

class YoloXSeedDector(SeedDetector):

    def __init__(self, exp_path, img_size, ckpt_file, threshold, device = 0):
        super(YoloXSeedDector, self).__init__(name="YoloX", threshold=threshold, model=None)
        self.device = device
        self.exp = get_exp(exp_path, self.name)
        self.exp.test_size =(img_size, img_size)
        ckpt  = torch.load(ckpt_file, map_location="cpu")
        self.model = self.exp.get_model()
        self.model.load_state_dict(ckpt['model'])
        self.img_size = img_size
        self.model.eval()
        self.predictor = Predictor(self.model, self.exp, ["seed", "background"], None, None, self.device, True, False)
        self.predictor.nmsthre = 0

    def predict(self, input_img:np.array):
        img = self.adjust_size(input_img)
        outputs, img_info = self.predictor.inference(img)
        logging.debug("[YoloX detector ] img.shape {}, redux {}, padding w {}, padding h {}".format(img.shape, self.r, self.pw, self.ph))
        logging.debug("[YOLOX DETECTOR] Outputs {}".format(outputs))

        return self.get_bboxs(preds=outputs[0])

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
        if(preds is None):
            return torch.tensor([[0, 0, 0, 0, 0, 0, 0]])
        return preds.to(self.device)*(torch.tensor([1/self.r, 1/self.r, 1/self.r, 1/self.r, 1, 1, 1]).to(self.device)) + torch.tensor([-self.pw[0], -self.ph[0], -self.pw[0], -self.ph[0], 0, 0, 0]).to(self.device)

    def get_max_prob_seeds(self, predictions, n=3):
        scores = torch.tensor([pred[4].cpu()*pred[5].cpu() for pred in predictions if pred.numel()>0])
        size = scores.size(dim=0)
        if size<n:
            scores = torch.cat((scores, torch.zeros(n-size)))
        return scores[:n]

    def filter(self, predictions):
        ret = predictions[torch.gt(predictions[:, 4]*predictions[:, 5], self.threshold), :]
        return ret

    def convert(self, predictions):
        seeds_camera_space = []
        seeds = predictions.size(dim=0)
        for seed in range(seeds):
            if(predictions[seed, 6]==1):
                x, y, h, w = self.convertToImgSpace(predictions[seed, :])
                logging.debug("[convert] Box {} converted into {}".format(predictions[seed, :], (x, y, h, w)))

                seeds_camera_space.append(SeedCamSpace(x, y, h, w, predictions[seed, 4]*predictions[seed, 5]))
            else:
                logging.debug("[convert YoloXdetector] Found a loggit with 0 may be background ?")
        return seeds_camera_space


    def convertToImgSpace(self, pred):
        box = pred[:4].cpu()
        x = (box[2] + box[0])/2
        y = (box[1] + box[3])/2
        w = (box[2] - box[0])
        h = (box[3] - box[1])
        return x, y, h, w

ckpt_file= "/home/harsha/Desktop/SLS-CNH/ObjectDetectionModels/YOLOX/YOLOX_outputs/yolox_s_seeds/latest_ckpt.pth"
exp_path = "/home/harsha/Desktop/SLS-CNH/ObjectDetectionModels/YOLOX/exps/example/custom/yolox_s_seeds.py"
detector = YoloXSeedDector(exp_path=exp_path, img_size=512, ckpt_file=ckpt_file, threshold=0.5)


if __name__ == "__main__":
    ckpt_file= "/home/harsha/Desktop/SLS-CNH/ObjectDetectionModels/YOLOX/YOLOX_outputs/yolox_s_seeds/latest_ckpt.pth"
    exp_path = "/home/harsha/Desktop/SLS-CNH/ObjectDetectionModels/YOLOX/exps/example/custom/yolox_s_seeds.py"
    YoloXDetector = YoloXSeedDector(exp_path=exp_path, img_size=512, ckpt_file=ckpt_file, threshold=0.5)
    dataset = "/home/harsha/Desktop/dataset/seed_detection_yolo5/test/images"
    imgs = LoadFilesWithExtensions(dataset, ["JPEG"])
    for img in imgs:
        input = cv2.imread(img)
        pred = YoloXDetector.detect(input)
        print(pred)
    print(imgs)