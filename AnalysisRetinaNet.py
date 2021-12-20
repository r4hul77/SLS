import os.path
import time
import numpy
from matplotlib.widgets import Slider, Button
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import yaml
from statistics import mean, stdev
from DataSets import GoodFieldDataSet
from SeedDistanceEstimator import *
import torch
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
from undistort import CameraCalibration

logging.basicConfig(filename="AnalysisLog.log", level=logging.DEBUG, filemode="w")

camera_calib = CameraCalibration("/home/harsha/Desktop/SLS-CNH/Data/exp3/Camera_20_calib.json", h=300, w=480)

class RetinaNetSeedDector(SeedDetector):

    def __init__(self, model_path, threshold, device = 0):
        super(RetinaNetSeedDector, self).__init__(name="Retina_Net", threshold=threshold, model=torch.load(model_path))
        self.device = device
    def predict(self, input_img:np.array):
        tensors = [ torch.Tensor(input_img[::-1] / 255).to(self.device).permute(2, 0, 1)]
        _, self.height, self.width = tensors[0].size()
        self.model.eval()

        with torch.no_grad():
            outs = self.model(tensors)
        ret = []
        logging.debug("[RetinaNetDetector] outs : {}".format(outs))
        return outs

    def get_max_prob_seeds(self, predictions, n=3):
        scores = [dict_obj["scores"].cpu() for dict_obj in predictions]
        ret = scores[0]
        size = ret.size(dim=0)
        if size<n:
            ret = torch.cat((ret, torch.zeros(n-size)))
        return ret[:n]

    def filter(self, predictions):
        ret = []
        for dict_obj in predictions:
            ret_dict = {}
            scores = torch.gt(dict_obj["scores"], self.threshold)
            ret_dict["scores"] = dict_obj["scores"][scores]
            ret_dict["labels"] = dict_obj["labels"][scores]
            ret_dict["boxes"]  = dict_obj["boxes"][scores]
            ret.append(ret_dict)
        logging.debug("[RetinaNetDetector] rets : {}".format(ret))
        return ret

    def convert(self, predictions):
        seeds_camera_space = []
        for i, box in enumerate(predictions[0]["boxes"]):
            x, y, h, w = self.convertToCamSpaceFromImgSpace(box)
            logging.debug("[convert] Box {} converted into {}".format(box, (x,y, h, w)))

            seeds_camera_space.append(SeedCamSpace(x, y, h, w, predictions[0]["scores"][i]))
        return seeds_camera_space

    def convertToCamSpaceFromImgSpace(self, box):
        box = box.cpu()
        x = (box[2] + box[0])/(2*self.width) * 0.13335
        y = (box[1] + box[3])/(2*self.height) * 0.13335
        w = (box[2] - box[0])/(self.width)*0.13335
        h = (box[3] - box[1])/(self.height)*0.13335
        return x, y, h, w

def prob_plot(plt, start_frame, probs, label):
    plt.plot(range(start_frame, len(probs)+start_frame), probs, label=label)


def save_multi_image(filename):
   pp = PdfPages(filename)
   fig_nums = plt.get_fignums()
   print("saving {} figures".format(len(fig_nums)))
   figs = [plt.figure(n) for n in fig_nums]
   for fig in figs:
      fig.savefig(pp, format='pdf')
   pp.close()
   print("saved figures")


def save_images(folder_name):
    if(not os.path.isdir(folder_name)):
        os.mkdir(folder_name)
    fig_nums = plt.get_fignums()
    print("saving {} figures in {}".format(len(fig_nums), folder_name))
    figs = [plt.figure(n) for n in fig_nums]
    for i, fig in enumerate(figs):
      fig.savefig(folder_name + "/{}.JPEG".format(i), format='JPEG', dpi=500)
    print("saved figures")

def validate(seeds_preds, validation_Data_csv_path, total_dist=50*0.3048):

    validation_data = pd.read_csv(validation_Data_csv_path).to_numpy()
    validation_data = validation_data[:, 0]*0.3048 + validation_data[:, 1]*0.3048

    distances = np.diff(validation_data)

    np.insert(distances, 0, 0, 0)

    seed0 = seeds_preds[0]

    def dist(seed0, seed1):
        n = seed1.n_c - seed0.n_c
        e = seed1.e_c - seed0.e_c
        return ( n*n + e*e )**0.5

    seeds_filtered = []

    total_distances = [validation_data[0]]

    prev_distance = 0
    for seed in seeds_preds:
        distance = dist(seed0.seed, seed.seed)
        if(distance < total_dist):
            seeds_filtered.append(seed)
            if(prev_distance == distance):
                total_distances.append(total_distances[-1]+seed.distance)
            else:
                total_distances.append(distance)
        else:
            break
        prev_distance = distance

    print("Total Seeds Predicted in {} m : {}".format(total_dist, len(seeds_filtered)))
    print("Total Seeds Counted in {} m : {}".format(total_dist, validation_data.shape[0]))

    valid_size = distances.shape[0]
    preds_size = len(seeds_filtered)

    i = 0
    total = 0
    l2_error = 0
    size = min(valid_size, preds_size)
    max_size = max(valid_size, preds_size)

    for j in range(size):
        abs_val = abs(seeds_filtered[j].distance - distances[j])
        total += abs_val
        l2_error += abs_val*abs_val
        i = j

    if(valid_size < preds_size):

        for j in range(preds_size - i):
            total += seeds_filtered[i+j].distance
            l2_error += seeds_filtered[i+j].distance*seeds_filtered[i+j].distance

    else:
        for j in range(valid_size - i):
            total += distances[j+i]
            l2_error += distances[j+i]*distances[j+i]

    l2_error = np.sqrt(l2_error)

    pred_distances = [seed.distance for seed in seeds_filtered]


    plt.figure()
    plt.title("Counts Validation Vs Distances Validation")
    plt.scatter(range(len(distances)), distances, cmap='r', label="validation data")
    plt.scatter(range(len(pred_distances)), pred_distances, marker='*', label="predicted data")
    plt.xlabel("Counts (N)")
    plt.ylabel("Distances (m)")
    plt.legend()

    idxs = min(len(pred_distances), len(distances))

    plt.figure()
    counts, bins = np.histogram(distances, bins=30)
    plt.hist(bins[:-1], bins=bins, weights=counts, label="validation data")
    plt.title("Histogram of Distances in Validation and Prediction")
    counts, bins = np.histogram(pred_distances, bins=bins)
    plt.hist(bins[:-1], bins, weights=counts, alpha=0.3, label="predicted distances")
    plt.xlabel("Distances(m)")
    plt.ylabel("Number(N)")
    plt.legend()

    plt.figure()
    plt.title("Predicted Distance Vs Validation Distance")
    plt.scatter(pred_distances[:idxs], distances[:idxs])
    plt.xlabel("Predicted Distance(m)")
    plt.ylabel("Validation Distance(m)")

    plt.figure()
    plt.title("Predicted Distance Vs Validation Distance Abs Error")
    plt.scatter(range(idxs), abs(pred_distances[:idxs]-distances[:idxs]))
    plt.xlabel("Count(N)")
    plt.ylabel("Abs Error (m)")

    print("Total Residual Spacing Left {}".format(total))
    print("Avg L1 error {}".format(total/max_size))
    print("Avg L2 Error {}".format(l2_error/max_size))
    print(max_size)
    print("Mean of Predicted {} Vs Mean of Data {}".format(torch.mean(torch.tensor(pred_distances)), mean(distances)))
    print("Std of Predicted {} vs Std of Data {}".format(torch.std(torch.tensor(pred_distances)), stdev(distances)))

    plt.figure()
    plt.plot("Seeds at a distance from the start of the count")
    print("Idxs : {}".format(idxs))
    plt.scatter(total_distances[:idxs], [0]*idxs, label="Predicted")
    plt.scatter(validation_data[:idxs], [0]*idxs, alpha=0.3, label="Validation Data")
    plt.legend()
    plt.xlabel("Distance(m)")
    plt.subplots_adjust(left=0.25, bottom=0.25)
    ax_count = plt.axes([0.25, 0.1, 0.65, 0.03])
    freq_slider = Slider(
        ax=ax_count,
        label='Count [N]',
        valmin=0,
        valmax=100,
        valinit=10,
    )


    def update(val):
        print(val)
    freq_slider.on_changed(update)

if __name__ == "__main__":

    folder = "/home/harsha/Desktop/SLS-CNH/Data/exp3/2-34"
    dataset = GoodFieldDataSet(folder=folder)

    file = "frame_info.txt"

    with open(folder + "/" + file, "r") as stream:
        frame_info = yaml.safe_load(stream)

    start_frame = frame_info["start_frame"]
    end_frame  = frame_info["end_frame"]

    print("Start Frame, End Frame in frame_info.txt are {}".format((start_frame, end_frame)))

    dataset.set_end_frame(end_frame)
    dataset.seek_frame(start_frame)
    dataset_tqdm = tqdm(dataset)
    model_path = "/home/harsha/Desktop/SLS-CNH/CNNModels/retina_fp_18_classified_allTrain2.pth"

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
        dataset_tqdm.set_description("Count {} Fps {}".format(count, (frame_idx +1)/(time.time()-start_time)))
        logging.debug("ROOT COUNT {}".format(count))
        seeds_utm += seed_distance.main(new_seeds, lat, long, vel, accel, 1/40, heading=heading, frame_idx=frame_idx)
    print("Total Seeds Found {}".format(count))

    validation_data = pd.read_csv(folder+"/ValidationData.csv").to_numpy()
    data_feet = validation_data[:, 0]*0.3048 + validation_data[:, 1]*0.3048

    delta_valid = np.diff(data_feet)
    numpy.insert(delta_valid, 0, 0, axis=0)


    delta_dists = [seed.distance for seed in seeds_utm]


    frame_idxs  = [seed.frame_idx + start_frame for seed in seeds_utm]
    counts      = [seed.count for seed in seeds_utm]

    plt.figure(0)
    plt.title("Histogram of Seed Distances")
    plt.ylabel("Number of Seeds(N)")
    plt.xlabel("Distance(m)")
    plt.hist(delta_dists, bins=75)

#    plt.figure(1)
#    ax = plt.axes(projection='3d')
#    ax.set_title("3D plot of Distances, Counts and Frame Indexes For Debugging purposes Only")
#   ax.plot3D(frame_idxs, delta_dists, counts)
#    ax.set_xlabel("Frame_IDXS(N)")
#    ax.set_ylabel("Distances(m)")
#    ax.set_zlabel("Counts(N)")

    plt.figure(2)
    plt.scatter(frame_idxs, counts)
    plt.title("Frame Indexs and Counts")
    plt.xlabel("Frame IDXS(N)")
    plt.ylabel("Counts(N)")

    plt.figure(3)
    plt.title("Counts Vs Distances")
    plt.scatter(counts, delta_dists, marker='+', norm=10)
    plt.xlabel("Counts (N)")
    plt.ylabel("Distances(m)")

    plt.figure(4)
    plt.scatter(frame_idxs, delta_dists)
    plt.title("Frame Indexes and Distances")
    plt.xlabel("Frame IDXS(N)")
    plt.ylabel("Distances (m)")

    validate(seeds_utm, folder+"/ValidationData.csv", total_dist=50*0.3048)

    plt.figure(5)
    plt.title("Seed Prob in Frame")
    prob_plot(plt, start_frame=start_frame, probs=[prob[0] for prob in probs], label="0 Prob")
    prob_plot(plt, start_frame=start_frame, probs=[prob[1] for prob in probs], label="1 Prob")
    prob_plot(plt, start_frame=start_frame, probs=[prob[2] for prob in probs], label="2 Prob")
    plt.xlabel("Frame No(N)")
    plt.ylabel("Prob(0<=p<=1)")
    plt.legend()

    save_multi_image(folder+"/results.pdf")
    save_images(folder+"/Images")
    print("Total Mean, Stdev : {}".format((torch.mean(torch.tensor(delta_dists)), torch.std(torch.tensor(delta_dists)))))
    print("Total Time Taken : {}".format(time.time() - start_time))
    plt.show()
    print("Done !")
