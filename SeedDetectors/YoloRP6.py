from YoloRSeedDetector import *

weights_path = "/home/harsha/yolor_256_3/best.pt"
cfg_path = "/home/harsha/Desktop/SLS-CNH/ObjectDetectionModels/yolor/cfg/yolor_p6_seeds.cfg"
detector = YoloRSeedDector(cfg=cfg_path, img_size=256, weights=weights_path, threshold=0.6)