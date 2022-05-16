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
    return jsd


def validate(seeds_graph, validation_Data_csv_path, total_dist=50*0.3048):

    validation_data = pd.read_csv(validation_Data_csv_path).to_numpy()
    validation_data = validation_data[:, 0]*0.3048 + validation_data[:, 1]*0.3048
    distances_pred = seeds_graph.get_distances()


    distances = np.diff(validation_data)

    ret = {}
    ret['jsd1_total'] = print_jsd(distances_pred, distances, exp="Pred = x")
    ret['jsd2_total'] = print_jsd(distances, distances_pred, exp="Valid = x")

    np.insert(distances, 0, 0, 0)

    seed0 = seeds_graph.head_node.seed_distance_info

    def dist(seed0, seed1):
        n = seed1.n_c - seed0.n_c
        e = seed1.e_c - seed0.e_c
        return ( n*n + e*e )**0.5

    seeds_filtered = []

    total_distances = [validation_data[0]]

    prev_distance = 0
    for seed in seeds_graph.get_seed_infos():
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

    ret["seed_count_pred_50ft"] = len(seeds_filtered)
    ret["seeed_count_valid_50ft"] = validation_data.shape[0]
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
    ret['jsd1_50_ft'] = print_jsd(distances, pred_distances, exp="Filtered valid = x")
    ret['jsd2_50_ft'] = print_jsd(pred_distances, distances, exp="Filterd pred = x")

    '''Plots'''

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
    counts_pred, bins = np.histogram(pred_distances, bins=bins)
    plt.hist(bins[:-1], bins, weights=counts_pred, alpha=0.3, label="predicted distances")
    plt.xlabel("Distances(m)")
    plt.ylabel("Number(N)")
    plt.legend()

    plt.figure()
    plt.title("Predicted Distance Vs Validation Distance")
    plt.scatter(pred_distances[:idxs], distances[:idxs])
    plt.xlabel("Predicted Distance(m)")
    plt.ylabel("Validation Distance(m)")

    plt.title("Predicted Distance Vs Validation Distance Abs Error")
    plt.scatter(range(idxs), abs(pred_distances[:idxs]-distances[:idxs]))
    plt.xlabel("Count(N)")
    plt.ylabel("Abs Error (m)")

    logging.warning("Total Residual Spacing Left {}".format(total))
    logging.warning("Avg L1 error {}".format(total/max_size))
    logging.warning("Avg L2 Error {}".format(l2_error/max_size))
    ret["RMS"] = l2_error/max_size
    logging.warning("Mean of Predicted {} Vs Mean of Data {}".format(torch.mean(torch.tensor(pred_distances)), mean(distances)))
    ret["Mean_Predicted"] = torch.mean(torch.tensor(pred_distances))
    ret["Mean_Valid"] = mean(distances)
    logging.warning("Std of Predicted {} vs Std of Data {}".format(torch.std(torch.tensor(pred_distances)), stdev(distances)))
    ret["stdev_predicted"] = torch.std(torch.tensor(pred_distances))
    ret["stdev_valid"] = stdev(distances)

    plt.figure()
    plt.title("Seeds at a distance from the start of the count")
    logging.debug("Idxs : {}".format(idxs))
    plt.scatter(total_distances[:idxs], [0]*idxs, label="Predicted")
    plt.scatter(validation_data[:idxs], [0]*idxs, alpha=0.3, label="Validation Data")
    plt.legend()
    plt.xlabel("Distance(m)")

    compare_adj_matrices(convert_to_adj_matrix(distances), seeds_graph.make_adj_matrix())

    return ret

def convert_to_dataframe(seed_dist_list):
    seeds = [seed_dist.seed for seed_dist in seed_dist_list]
    df = pd.DataFrame(seeds)
    df1 = pd.DataFrame(seed_dist_list)
    df1 = df1.drop('seed', axis=1)
    return df.join(df1)

def convert_to_adj_matrix(distances):
    ret_mat = np.zeros((len(distances)+1, len(distances)+1))
    for i in range(len(distances)):
        ret_mat[i, i+1] = distances[i]
        ret_mat[i+1, i] = distances[i]
    return ret_mat

def compare_adj_matrices(ground_truth, predictions):
    #TODO Write Infinity Norm
    #print(np.linalg.norm(ground_truth, axis=1) - np.linalg.norm(predictions, axis=1))
    pass