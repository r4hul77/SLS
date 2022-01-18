import numpy as np
import pandas as pd
from statistics import mean, stdev
from SeedDistanceEstimator import *
import torch
import logging
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon

def print_jsd(x, y, bins=30, exp=""):
    c_x, b_x = np.histogram(x, bins=bins)
    p_x = c_x/np.sum(c_x)
    c_y, b_y = np.histogram(y, bins=b_x)
    p_y = c_y/np.sum(c_y)
    jsd = jensenshannon(p_x, p_y)
    logging.warning(exp+"Jensen Shannon Distance with nbins {} is {}".format(bins, jsd))


def validate(seeds_preds, validation_Data_csv_path, total_dist=50*0.3048):

    validation_data = pd.read_csv(validation_Data_csv_path).to_numpy()
    validation_data = validation_data[:, 0]*0.3048 + validation_data[:, 1]*0.3048
    distances_pred = [seed.distance for seed in seeds_preds]


    distances = np.diff(validation_data)

    print_jsd(distances_pred, distances, exp="Pred = x")
    print_jsd(distances, distances_pred, exp="Valid = x")

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

    logging.warning("Total Seeds Predicted in {} m : {}".format(total_dist, len(seeds_filtered)))
    logging.warning("Total Seeds Counted in {} m : {}".format(total_dist, validation_data.shape[0]))

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
    print_jsd(distances, pred_distances, exp="Filtered valid = x")
    print_jsd(pred_distances, distances, exp="Filterd pred = x")

    '''Plots'''

    fig, ax = plt.subplots(5, 1)
    fig.tight_layout()
    fig.canvas.manager.full_screen_toggle()
    ax[0].set_title("Counts Validation Vs Distances Validation")
    ax[0].scatter(range(len(distances)), distances, cmap='r', label="validation data")
    ax[0].scatter(range(len(pred_distances)), pred_distances, marker='*', label="predicted data")
    ax[0].set_xlabel("Counts (N)")
    ax[0].set_ylabel("Distances (m)")
    plt.legend()

    idxs = min(len(pred_distances), len(distances))

    counts, bins = np.histogram(distances, bins=30)
    ax[1].hist(bins[:-1], bins=bins, weights=counts, label="validation data")
    ax[1].set_title("Histogram of Distances in Validation and Prediction")
    counts_pred, bins = np.histogram(pred_distances, bins=bins)
    ax[1].hist(bins[:-1], bins, weights=counts_pred, alpha=0.3, label="predicted distances")


    ax[1].set_xlabel("Distances(m)")
    ax[1].set_ylabel("Number(N)")
    plt.legend()

    ax[2].set_title("Predicted Distance Vs Validation Distance")
    ax[2].scatter(pred_distances[:idxs], distances[:idxs])
    ax[2].set_xlabel("Predicted Distance(m)")
    ax[2].set_ylabel("Validation Distance(m)")

    ax[3].set_title("Predicted Distance Vs Validation Distance Abs Error")
    ax[3].scatter(range(idxs), abs(pred_distances[:idxs]-distances[:idxs]))
    ax[3].set_xlabel("Count(N)")
    ax[3].set_ylabel("Abs Error (m)")

    logging.warning("Total Residual Spacing Left {}".format(total))
    logging.warning("Avg L1 error {}".format(total/max_size))
    logging.warning("Avg L2 Error {}".format(l2_error/max_size))
    logging.warning("Mean of Predicted {} Vs Mean of Data {}".format(torch.mean(torch.tensor(pred_distances)), mean(distances)))
    logging.warning("Std of Predicted {} vs Std of Data {}".format(torch.std(torch.tensor(pred_distances)), stdev(distances)))

    ax[4].set_title("Seeds at a distance from the start of the count")
    logging.debug("Idxs : {}".format(idxs))
    ax[4].scatter(total_distances[:idxs], [0]*idxs, label="Predicted")
    ax[4].scatter(validation_data[:idxs], [0]*idxs, alpha=0.3, label="Validation Data")
    plt.legend()
    ax[4].set_xlabel("Distance(m)")
