from YoloRSeedDetector import *

weights_path = "/home/harsha/Desktop/SLS-CNH/TrainedWeights/YoloR_CSP_X_best.pt"
cfg_path = "/home/harsha/Desktop/SLS-CNH/ObjectDetectionModels/yolor/cfg/yolor_csp_seeds.cfg"
detector = YoloRSeedDector(cfg=cfg_path, img_size=512, weights=weights_path, threshold=0.63)