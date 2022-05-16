import sys
import importlib
import argparse
import os
import time
import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import torchvision
import matplotlib.pyplot as plt
import numpy as np

def get_parser():

    parser = argparse.ArgumentParser(description="Analyze Seed Spacing")
    parser.add_argument('-d', '--detector_file', help="Seed Detector Location")
    parser.add_argument('-o', '--output_dir', help="output director for the results")
    return parser

def get_detector(detector_file):

    try:
        sys.path.append(os.path.dirname(detector_file))
    except Exception:
        raise ValueError("Can't find the file {}".format(detector_file))
    try:
        detector_current = importlib.import_module(os.path.basename(detector_file).split(".")[0])
    except Exception:
        raise  ImportError("Error Importing {}".format(detector_file))
    try:
        detector = detector_current.detector
    except Exception:
        raise ImportError("{} doesn't contains class named 'detector'".format(detector_file))
    return detector

class Test1DataSet(Dataset):

    def __init__(self, root_dir):
        self.images_dir = os.path.join(root_dir, 'images')
        self.labels_dir = os.path.join(root_dir, 'labels')
        self.images_list = [f for f in os.listdir(self.images_dir)
                            if os.path.isfile(os.path.join(self.images_dir, f))
                            ]

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        img_name = self.images_list[idx]

        #image = io.read_image(os.path.join(self.images_dir, img_name))
        image = cv2.imread(os.path.join(self.images_dir, img_name))
        cv2.imwrite('/home/harsha/Desktop/dataset/test_results/imgs/'+ str(idx) + '.jpeg', image)
        label = self.get_yolov5tensor(os.path.join(self.labels_dir, img_name))

        return {
            'img': image,
            'label': label
        }


    def get_yolov5tensor(self, txt_file):
        with open(txt_file[:-3]+'txt', 'r') as f:
            lines = f.readlines()
        ret = list(map(lambda line : list(map(lambda x: float(x), line.split(' '))), lines))
        return torch.tensor(ret)

def visualize(img, tensors):
    img = torch.squeeze(img).byte()
    _, h, w = img.shape
    tensors = torch.squeeze(tensors, dim=0)
    for tensor in tensors:
        pts = get_pt1pt2(tensor, w, h)
        print(pts)
        #img = draw_bounding_boxes(img, pts, colors=[(240, 10, 17)])
    img = torchvision.transforms.ToPILImage()(img)
    plt.imshow(img)
    plt.show()
    plt.clf()
    time.sleep(0.25)

def get_pt1pt2(tensor, w, h):
    x_c = w*tensor[1]
    y_c = h*tensor[2]
    w_c = w*tensor[3]
    h_c = w*tensor[4]
    pt1 = (x_c-w_c//2, y_c-h_c//2)
    pt2 = (x_c+w_c//2, y_c+h_c//2)
    return torch.tensor([[*pt1, *pt2]])


def save_ground_truth_file(dir, label, img_no, w=256, h=256):
    #np.savetxt(os.path.join(dir, str(img_no) + '.txt'), torch.squeeze(label, dim=0).numpy())
    torch_array = torch.squeeze(label, dim=0).numpy()
    torch_array[:, [1, 3]] *= w
    torch_array[:, [2, 4]] *= h
    mul = np.array([[1 ,0, -0.5, 0], [0, 1, 0, -0.5], [1, 0, 0.5, 0], [0, 1, 0, 0.5]]).transpose()
    torch_array[:, 1:] = torch_array[:, 1:] @ mul
    np.savetxt(os.path.join(dir, str(img_no) + '.txt'), torch_array.astype(int), fmt="%.2f")


def save_detections(dir, detections, img_no, h=256, w=256):
    np.savetxt(os.path.join(dir, str(img_no) + '.txt'), convert_to_numpy(detections, h, w), fmt="%.2f", delimiter=' ')

def convert_to_numpy(detections_list, h, w):
    ret = np.zeros((len(detections_list), 6))
    for i, detection in enumerate(detections_list):
        ret[i, 1:] = convert_to_yolo_format(detection, h, w)

    return ret

def convert_to_yolo_format(detection, h, w):

    left = detection.x_c - detection.w
    top = detection.y_c - detection.h
    right = detection.x_c + detection.w
    bot = detection.y_c + detection.h

    return np.array([detection.p, int(left), int(top), int(right), int(bot)])


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    detector = get_detector(args.detector_file)
    dataset = Test1DataSet(root_dir = "/home/harsha/Desktop/dataset/yolo5")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    dir = "/home/harsha/Desktop/dataset/ground_truth"
    out_dir = os.path.join("/home/harsha/Desktop/dataset/test_results", args.output_dir)
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    for i, sample in enumerate(tqdm.tqdm(dataloader)):
        save_ground_truth_file(dir, sample['label'], i)
        img = torch.squeeze(sample['img']).numpy()
        seeds, probs = detector.detect(img)
        save_detections(out_dir, seeds, i)
