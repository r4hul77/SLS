import cv2
import sys
YOLOR_PATH = "/home/harsha/Desktop/SLS-CNH/ObjectDetectionModels/yolor"
sys.path.insert(0, YOLOR_PATH)
from SeedDetectors.SeedDetectorBase import *
from ObjectDetectionModels.yolor.models.models import *
from ObjectDetectionModels.yolor.utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer)
from utils_seed.filehandling import *
from ObjectDetectionModels.yolor.utils.datasets import letterbox
class YoloRSeedDector(SeedDetector):

    def __init__(self, cfg, img_size, weights, threshold, device = 0):
        super(YoloRSeedDector, self).__init__(name="YoloR", threshold=threshold, model=None)
        self.device = device
        self.model = Darknet(cfg, img_size).to(device)
        self.model.load_state_dict(torch.load(weights)['model'])
        self.img_size =img_size
        self.idx = 0
        self.r = 0 #Resize Factor
        self.pw = (0, 0) #Padding width left, right
        self.ph = (0, 0) #Padding height top, bottom

    def predict(self, input_img:np.array):
        img = self.adjust_size(input_img)
        logging.debug("[Yolor detector ] img.shape {}, redux {}, padding w {}, padding h {}".format(img.shape, self.r, self.pw, self.ph))

        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = img/255.0

        tensors = torch.from_numpy(img)
        tensors =tensors.unsqueeze(0).type(torch.FloatTensor).to(self.device)
        logging.debug("[Tensors Shape] ts = {}".format(tensors.shape))
        _, self.height, self.width = tensors[0].size()

        self.model.eval()
        '''ISSUE WITH PADDING'''
        with torch.no_grad():
            outs = self.model(tensors, augment=False)[0]
            ret = non_max_suppression(outs, self.threshold, 0.25, agnostic=True)
            logging.debug("[NMS yolorDetector] preds : {}".format(ret))

        return self.get_bboxs(preds=ret[0])

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
        return preds.to(self.device)*(torch.tensor([1/self.r, 1/self.r, 1/self.r, 1/self.r, 1, 1]).to(self.device)) + torch.tensor([-self.pw[0], -self.ph[0], -self.pw[0], -self.ph[0], 0, 0]).to(self.device)

    def get_max_prob_seeds(self, predictions, n=3):
        scores = torch.tensor([pred[-2].cpu() for pred in predictions if pred.numel()>0])
        size = scores.size(dim=0)
        if size<n:
            scores = torch.cat((scores, torch.zeros(n-size)))
        return scores[:n]

    def filter(self, predictions):
        ret = predictions[torch.gt(predictions[:, -2], self.threshold), :]
        return ret

    def convert(self, predictions):
        seeds_camera_space = []
        seeds = predictions.size(dim=0)
        for seed in range(seeds):
            x, y, h, w = self.convertToImgSpace(predictions[seed, :])
            logging.debug("[convert] Box {} converted into {}".format(predictions[seed, :], (x, y, h, w)))

            seeds_camera_space.append(SeedCamSpace(x, y, h, w, predictions[seed, -2]))
        return seeds_camera_space


    def convertToImgSpace(self, pred):
        box = pred[:4].cpu()
        x = (box[2] + box[0])/2
        y = (box[1] + box[3])/2
        w = (box[2] - box[0])
        h = (box[3] - box[1])
        return x, y, h, w

if __name__ == "__main__":
    weights_path = "/home/harsha/Desktop/SLS-CNH/ObjectDetectionModels/yolor/runs/train/yolor_csp_seeds/weights/best_overall.pt"
    cfg_path = "/home/harsha/Desktop/SLS-CNH/ObjectDetectionModels/yolor/cfg/yolor_csp_seeds.cfg"
    yolorDetector = YoloRSeedDector(cfg=cfg_path, img_size=480, weights=weights_path, threshold=0.5)
    dataset = "/home/harsha/Desktop/dataset/seed_detection_yolo5/test/images"
    imgs = LoadFilesWithExtensions(dataset, ["JPEG"])
    for img in imgs:
        input = cv2.imread(img)
        pred = yolorDetector.predict(input)
        print(pred)
    print(imgs)