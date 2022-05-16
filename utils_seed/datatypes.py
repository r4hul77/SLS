import dataclasses

from typing import List

import numpy as np


@dataclasses.dataclass
class SeedCamSpace:
    x_c: float #X center
    y_c: float #Y Center
    h  : float #Height
    w  : float #Width
    p  : float #Probality

SeedList = List[SeedCamSpace]

@dataclasses.dataclass
class SeedUTMSpace:
    n_c  : float
    e_c  : float
    zone : int
    zone_letter : str
    h    : float
    w    : float
    p    : float


@dataclasses.dataclass
class SeedDistStruct:
    seed     : SeedUTMSpace
    count    :    int
    distance : float
    frame_idx : int

class Node:

    def __init__(self, seed_distance_info):
        self.seed_distance_info = seed_distance_info
        self.next_distance =  0.
        self.next_seed  = None


class Graph:

    def __init__(self):
        self.head_node = None
        self.terminal_node = None
        self.size = 0

    def insert_node(self, seed_distance_info):
        new_node = Node(seed_distance_info)
        self.size += 1
        if(not self.head_node):
            self.head_node = new_node
            self.terminal_node = self.head_node

        else:
            self.terminal_node.next_distance = seed_distance_info.distance
            self.terminal_node.next_seed = new_node
            self.terminal_node = self.terminal_node.next_seed

    def make_adj_matrix(self):
        matrix = np.zeros((self.size, self.size))
        current_node = self.head_node
        for i in range(self.size-1):
            matrix[i, i+1] = current_node.next_distance
            matrix[i+1, i] = current_node.next_distance
            current_node = current_node.next_seed
        return matrix

    def get_distances(self):
        ret_distances = []
        current_node = self.head_node
        while(current_node.next_seed):
            ret_distances.append(current_node.next_distance)
            current_node = current_node.next_seed
        return ret_distances

    def get_frame_idxs(self, start_frame = 0):
        frame_idxs = []
        current_node = self.head_node
        while(current_node):
            frame_idxs.append(current_node.seed_distance_info.frame_idx + start_frame)
            current_node = current_node.next_seed
        return frame_idxs

    def get_counts(self):
        return list(range(self.size))

    def get_seed_infos(self):
        current_node = self.head_node
        ret_seeds = []
        while(current_node):
            ret_seeds.append(current_node.seed_distance_info)
            current_node = current_node.next_seed
        return ret_seeds
