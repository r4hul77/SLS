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
from utils.validate import validate
from SeedDetectors.RetinanetSeedDetector import *
from Filters.seed_filter import *


logging.basicConfig(filename="AnalysisLog.log", level=logging.DEBUG, filemode="w")

camera_calib = CameraCalibration("/home/harsha/Desktop/SLS-CNH/Data/exp3/Camera_20_calib.json", h=300, w=480)

def prob_plot(plt, start_frame, probs, label, ax):
    ax.plot(range(start_frame, len(probs)+start_frame), probs, label=label)


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
      fig.savefig(folder_name + "/{}.JPEG".format(i), format='JPEG', dpi=500)
    logging.debug("saved figures")

if __name__ == "__main__":

    folder = "/home/harsha/Desktop/SLS-CNH/Data/exp3/2-34"
    dataset = GoodFieldDataSet(folder=folder)

    file = "frame_info.txt"

    with open(folder + "/" + file, "r") as stream:
        frame_info = yaml.safe_load(stream)

    start_frame = frame_info["start_frame"]
    end_frame  = frame_info["end_frame"]

    logging.debug("Start Frame, End Frame in frame_info.txt are {}".format((start_frame, end_frame)))

    dataset.set_end_frame(end_frame)
    dataset.seek_frame(start_frame)
    dataset_tqdm = tqdm(dataset, total=dataset.get_total_frames())
    model_path = "/home/harsha/Desktop/SLS-CNH/TrainedWeights/retina_fp_18_classified_allTrain2.pth"

    retinaDetector = RetinaNetSeedDector(model_path=model_path, threshold=0.5)
    seed_filter = SeedFilter(seed_detector=retinaDetector, iou_threshold=0.225)

    count = 0
    seed_distance = SeedXDistanceEstimator()
    seeds_utm = []
    start_time = time.time()
    probs = []

    for frame_idx, img_0, _, _, _, lat, long, vel, accel, heading in dataset_tqdm:

        logging.debug("____________________________________Frame No {} ______________".format(frame_idx))
        undistorted = camera_calib.undistort(img_0)
        new_seeds, seed_probs = seed_filter.main(frame=undistorted, lat=lat, long=long, velocity=vel, acceleration=accel, dt=1/40, seed_prob_dim=3)
        probs.append(seed_probs)
        count += len(new_seeds)
        dataset_tqdm.set_description("Count {} Fps {}".format(count, (frame_idx - start_frame + 1)/(time.time()-start_time)))
        logging.debug("ROOT COUNT {}".format(count))
        seeds_utm += seed_distance.main(new_seeds, lat, long, vel, accel, 1/40, heading=heading, frame_idx=frame_idx)

    logging.warning("Total Seeds Found {}".format(count))

    validation_data = pd.read_csv(folder+"/ValidationData.csv").to_numpy()
    data_feet = validation_data[:, 0]*0.3048 + validation_data[:, 1]*0.3048

    delta_valid = np.diff(data_feet)
    numpy.insert(delta_valid, 0, 0, axis=0)


    delta_dists = [seed.distance for seed in seeds_utm]


    frame_idxs  = [seed.frame_idx + start_frame for seed in seeds_utm]
    counts      = [seed.count for seed in seeds_utm]



    '''Plots For Vizualization'''

    fig, ax = plt.subplots(5, 1)
    fig.tight_layout()
    fig.canvas.manager.full_screen_toggle()
    ax[0].set_title("Histogram of Seed Distances")
    ax[0].set_ylabel("Number of Seeds(N)")
    ax[0].set_xlabel("Distance(m)")
    ax[0].hist(delta_dists, bins=75)

    ax[1].scatter(frame_idxs, counts)
    ax[1].set_title("Frame Indexs and Counts")
    ax[1].set_xlabel("Frame IDXS(N)")
    ax[1].set_ylabel("Counts(N)")

    ax[2].set_title("Counts Vs Distances")
    ax[2].scatter(counts, delta_dists, marker='+', norm=10)
    ax[2].set_xlabel("Counts (N)")
    ax[2].set_ylabel("Distances(m)")

    ax[3].scatter(frame_idxs, delta_dists)
    ax[3].set_title("Frame Indexes and Distances")
    ax[3].set_xlabel("Frame IDXS(N)")
    ax[3].set_ylabel("Distances (m)")


    ax[4].set_title("Seed Prob in Frame")
    prob_plot(plt, start_frame=start_frame, probs=[prob[0] for prob in probs], label="0 Prob", ax=ax[4])
    prob_plot(plt, start_frame=start_frame, probs=[prob[1] for prob in probs], label="1 Prob", ax=ax[4])
    prob_plot(plt, start_frame=start_frame, probs=[prob[2] for prob in probs], label="2 Prob", ax=ax[4])
    ax[4].set_xlabel("Frame No(N)")
    ax[4].set_ylabel("Prob(0<=p<=1)")
    plt.legend()

    validate(seeds_utm, folder+"/ValidationData.csv", total_dist=50*0.3048)



    save_multi_image(folder+"/results.pdf")
    save_images(folder+"/Images")
    print("Total Mean, Stdev : {}".format((torch.mean(torch.tensor(delta_dists)), torch.std(torch.tensor(delta_dists)))))
    print("Total Time Taken : {}".format(time.time() - start_time))
    print("Done !")
    plt.show()

