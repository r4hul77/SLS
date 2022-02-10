from YoloRSeedDetector import *

weights_path = "/home/harsha/Desktop/SLS-CNH/TrainedWeights/yolor_p6_800_best.pt"
cfg_path = "/home/harsha/Desktop/SLS-CNH/ObjectDetectionModels/yolor/cfg/yolor_p6_seeds.cfg"
detector = YoloRSeedDector(cfg=cfg_path, img_size=512, weights=weights_path, threshold=0.5)