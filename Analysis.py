import os
import logging
import torch
from SeedDistanceEstimator import SeedXDistanceEstimator
import os.path
import time
import numpy
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import yaml
from DataSets import GoodFieldDataSet
import matplotlib.pyplot as plt
from tqdm import tqdm
from undistort import CameraCalibration
from utils_seed.validate import validate
from Filters.seed_filter import *
from FrameTransformer.FrameTransformerBase import FrameTransformerBase
from utils_seed.display import *
from utils_seed.validate import convert_to_dataframe
from FlowEstimator import *


def prob_plot(plt, start_frame, probs, label, ax=None):
    plt.plot(range(start_frame, len(probs)+start_frame), probs, label=label)


def save_multi_image(filename):
   pp = PdfPages(filename)
   fig_nums = plt.get_fignums()
   logging.debug("saving {} figures".format(len(fig_nums)))
   figs = [plt.figure(n) for n in fig_nums]
   for fig in figs:
      fig.savefig(pp, format='pdf')
   pp.close()
   logging.debug("saved figures")


def save_images(folder_name):
    if(not os.path.isdir(folder_name)):
        os.mkdir(folder_name)
    fig_nums = plt.get_fignums()
    logging.debug("saving {} figures in {}".format(len(fig_nums), folder_name))
    figs = [plt.figure(n) for n in fig_nums]
    for i, fig in enumerate(figs):
        fig.set_size_inches(32, 18)
        fig.savefig(folder_name + "/{}.png".format(i), format='png', dpi=100, bbox_inches='tight')
    logging.debug("saved figures")


def mkdir(path):
    if(not os.path.exists(path)):
        os.mkdir(path)

def analyse(exp, detector, results_path, camera_calib_path="/home/harsha/Desktop/SLS-CNH/Data/exp3/Camera_20_calib.json", frames_yaml='frame_info.txt', dataset_folder="/home/harsha/Desktop/SLS-CNH/Data/exp3/"):

    results_folder = os.path.join(results_path, exp)

    mkdir(results_folder)

    detections_folder = os.path.join(results_folder, "detections")

    mkdir(detections_folder)

    current_dataset_path = os.path.join(dataset_folder, exp)

    dataset = GoodFieldDataSet(folder=current_dataset_path)

    flow_estimator = FlowEstimator(queue_size=3, debug=True)

    with open(os.path.join(current_dataset_path, frames_yaml), "r") as stream:
        frame_info = yaml.safe_load(stream)

    start_frame = frame_info["start_frame"]
    end_frame   = frame_info["end_frame"]

    logging.debug("Start Frame, End Frame in frame_info.txt are {}".format((start_frame, end_frame)))

    dataset.set_end_frame(end_frame)
    dataset.seek_frame(start_frame)
    dataset_tqdm = tqdm(dataset, total=dataset.get_total_frames())

    frame_transformer = FrameTransformerBase(0)
    seed_filter = SeedFilter(iou_threshold=0.225)

    count = 0
    seed_distance = SeedXDistanceEstimator()
    seeds_utm = []
    start_time = time.time()
    probs = []
    camera_calib = CameraCalibration(camera_calib_path, h=300, w=480)

    for frame_idx, img_0, _, _, _, lat, long, vel, accel, heading in dataset_tqdm:
        height, width, _ = img_0.shape
        logging.debug("____________________________________Frame No {} ______________".format(frame_idx))

        undistorted = camera_calib.undistort(img_0)

        detections, seed_probs = detector.detect(input_img=undistorted, seed_prob_dim=3)

        state, viz = flow_estimator.main(undistorted)

        logging.debug("[Main] Seed Probs {}".format(seed_probs))
        viz = draw_bbox(viz, detections)
        probs.append(seed_probs)
        detectionsCamSpace = list(map(lambda x: frame_transformer.convert(x, width=width, height=height), detections))
        logging.debug("[Main] SeedS CamSpace {}".format(detectionsCamSpace))


        new_seeds = seed_filter.main(detections=detectionsCamSpace, lat=lat, long=long, velocity=vel, acceleration=accel, dt=1/40)
        count += len(new_seeds)
        logging.debug("ROOT COUNT {}".format(count))

        seeds_utm_now = seed_distance.main(new_seeds, lat, long, vel, accel, 1/40, heading=heading, frame_idx=frame_idx)
        logging.debug("[Main] Seeds UTM {}".format(seeds_utm_now))
        seeds_utm += seeds_utm_now


        cv2.imwrite(detections_folder+"/{}.JPEG".format(frame_idx), viz)

        dataset_tqdm.set_description("Count {} Fps {}".format(count, (frame_idx - start_frame + 1)/(time.time()-start_time)))

    logging.warning("Total Seeds Found {}".format(count))

    df_seeds = convert_to_dataframe(seeds_utm)

    df_seeds.to_csv(os.path.join(results_folder, 'results.csv'))

    validation_data = pd.read_csv(os.path.join(current_dataset_path, "ValidationData.csv")).to_numpy()
    data_feet = validation_data[:, 0]*0.3048 + validation_data[:, 1]*0.3048

    delta_valid = np.diff(data_feet)
    numpy.insert(delta_valid, 0, 0, axis=0)


    delta_dists = [seed.distance for seed in seeds_utm]


    frame_idxs  = [seed.frame_idx + start_frame for seed in seeds_utm]
    counts      = [seed.count for seed in seeds_utm]

    '''Plots For Vizualization'''

    plt.figure()
    plt.title("Histogram of Seed Distances")
    plt.ylabel("Number of Seeds(N)")
    plt.xlabel("Distance(m)")
    plt.hist(delta_dists, bins=75)


    plt.figure()
    plt.scatter(frame_idxs, counts)
    plt.title("Frame Indexs and Counts")
    plt.xlabel("Frame IDXS(N)")
    plt.ylabel("Counts(N)")

    plt.figure()
    plt.title("Counts Vs Distances")
    plt.scatter(counts, delta_dists, marker='+', norm=10)
    plt.xlabel("Counts (N)")
    plt.ylabel("Distances(m)")

    plt.figure()
    plt.scatter(frame_idxs, delta_dists)
    plt.title("Frame Indexes and Distances")
    plt.xlabel("Frame IDXS(N)")
    plt.ylabel("Distances (m)")


    plt.figure()
    plt.title("Seed Prob in Frame")
    prob_plot(plt, start_frame=start_frame, probs=[prob[0] for prob in probs], label="0 Prob")
    prob_plot(plt, start_frame=start_frame, probs=[prob[1] for prob in probs], label="1 Prob")
    prob_plot(plt, start_frame=start_frame, probs=[prob[2] for prob in probs], label="2 Prob")
    plt.xlabel("Frame No(N)")
    plt.ylabel("Prob(0<=p<=1)")
    plt.legend()

    ret_dict = validate(seeds_utm, os.path.join(current_dataset_path, "ValidationData.csv"), total_dist=50*0.3048)

    with open(os.path.join(results_folder, "validation_results.csv"), 'w') as file:
        for key in ret_dict.keys():
            file.write("%s, %f\n" % (key, ret_dict[key]))

    save_multi_image(os.path.join(results_folder, "results.pdf"))
    save_images(os.path.join(results_folder, "Graphs"))
    logging.warning("Total Mean, Stdev : {}".format((torch.mean(torch.tensor(delta_dists)), torch.std(torch.tensor(delta_dists)))))
    logging.warning("Total Time Taken : {}".format(time.time() - start_time))
    print("Done !")