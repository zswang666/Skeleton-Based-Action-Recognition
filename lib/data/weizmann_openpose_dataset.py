# general
import os
import sys
import numpy as np
import random
import logging
import json
import networkx as nx
import matplotlib.pyplot as plt

# local packages
from .base_dataset import BaseDataset
from .utils import *
from .weizmann_openpose_graph import *

# debug
import ipdb

# define logger
logger = logging.getLogger(__name__)

class Dataset(BaseDataset):
    '''
    TODO: ignore joints, load joint mapping from file
    '''
    def initialize(self, config):
        logger.info("Initializing weizmann-openpose dataset...")

        # data configuration
        self._data_path = config.data_path
        self._ignore_empty_sample = config.ignore_empty_sample
        self._use_adjacency_matrix = config.use_adjacency_matrix
        self._ignore_uncertain_node = config.ignore_uncertain_node
        if self._ignore_uncertain_node:
            raise NotImplementedError
        self._node_certainty_threshold = config.node_certainty_threshold
        self._primitive_range = config.primitive_range
        self._n_edge_types = config.n_edge_types

        # specify label mapping
        cnames = sorted(os.listdir(self._data_path))
        self._label_mapping = dict()
        for idx, cname in enumerate(cnames):
            self._label_mapping[cname] = idx

        # load data
        self._data_list = []
        self._label_list = []
        self._n_frames_all = []
        for cname in cnames: # iterate class
            cpath = os.path.join(self._data_path,cname)
            for pname in os.listdir(cpath): # iterate person
                # get directory path containing a sequence of .json
                ppath = os.path.join(cpath,pname)
                self._data_list.append(ppath)
                # get label for the sequence
                self._label_list.append(self._label_mapping[cname])
                # get maximum sequence length
                n_frames = len(sorted(os.listdir(ppath)))
                self._n_frames_all.append(n_frames)
        self._n_frames_all = np.array(self._n_frames_all)
        self._max_frames = int(self._n_frames_all.max())

        # specify data container shape
        self._n_joints = 18
        self._n_channels = 3 # node-x,node-y,confidence
        self._n_nodes = self._n_joints * self._primitive_range
        self._data_shape = (self._max_frames, \
                            self._n_joints, \
                            self._n_channels)

        # specify full-skeleton adjacency matrix
        if self._use_adjacency_matrix:
            self._full_joints_mapping = joint_label_mapping(self._primitive_range)
            self._graph = create_graph(self._primitive_range)
            self._adjacency_matrix = create_adjacency_matrix(self._graph, self._n_nodes, 
                                                             self._n_edge_types, self._full_joints_mapping)

        # image size, for pose normalization
        self._img_X = 180
        self._img_Y = 144

    def __getitem__(self, index):
        # get sequence directory and label
        dpath = self._data_list[index]
        label = self._label_list[index]

        # load data (.json for each frame)
        data = np.zeros(self._data_shape)
        json_files = sorted(os.listdir(dpath)) # IMPORTANT!! sorted()
        frame_idx = 0
        for json_file in json_files:
            # load one frame
            json_path = os.path.join(dpath,json_file)
            with open(json_path, 'r') as f:
                frame_info = json.load(f)
            people_info = frame_info["people"]
            n_people = len(people_info)
            if n_people==0 and self._ignore_empty_sample: # no skeleton
                continue
            else:
                # There is only one person in WEIZMANN dataset.
                # we use total confidence to filter out false positive
                x_list = []
                y_list = []
                c_list = []
                for person_info in people_info: # iterate through frames
                    x = np.array(person_info["pose_keypoints"][0::3], dtype=np.double)
                    y = np.array(person_info["pose_keypoints"][1::3], dtype=np.double)
                    p = person_info["pose_keypoints"][2::3]
                    x_list.append(x/self._img_X)
                    y_list.append(y/self._img_Y)
                    c_list.append(p)
                c_total = np.sum(c_list, axis=1)
                which_person = np.argmax(c_total)

                data[frame_idx,:,0] = x_list[which_person]
                data[frame_idx,:,1] = y_list[which_person]
                data[frame_idx,:,2] = c_list[which_person]
                # update frame index
                frame_idx += 1

        # obtain adjacency matrix
        am = self._adjacency_matrix.copy()
        
        # pack output
        out = [data, am, label, frame_idx]

        return out

    def frame_num_statistics(self):
        stats = dict()
        stats["mean"] = self._n_frames_all.mean()
        stats["std"] = self._n_frames_all.std()
        stats["min"] = self._n_frames_all.min()
        stats["max"] = self._max_frames
        
        return stats

    def visualize_graph(self, mode=0):
        if mode==0:
            nx.draw(self._graph, with_labels=True, font_weight="bold")
            plt.show()
        elif mode==1:
            draw_graph3d(self._graph)
        else:
            raise ValueError("No such mode in visualize_graph")

    def _normalize_pose(self):
        raise NotImplementedError

    @property
    def data_list(self):
        return self._data_list

    @property
    def label_mapping(self):
        return self._label_mapping

    @property
    def full_joints_mapping(self):
        return self._full_joints_mapping

    @property
    def data_shape(self):
        return self._data_shape
    
    @property
    def n_nodes(self):
        return self._n_nodes

    @property
    def n_joints(self):
        return self._n_joints

    @property
    def n_channels(self):
        return self._n_channels

    @property
    def max_frames(self):
        return self._max_frames

    def __len__(self):
        return len(self._label_list)

    def name(self):
        return "WeizmannOpenposeDataset"
