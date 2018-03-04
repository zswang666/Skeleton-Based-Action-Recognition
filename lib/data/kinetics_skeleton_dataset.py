# sys
import os
import sys
import numpy as np
import random
import logging
import json

# pytorch
import torch

# local packages
from .base_dataset import BaseDataset
from .utils import *

# debug
import ipdb

# define logger
logger = logging.getLogger(__name__)

class Dataset(BaseDataset):
    def initialize(self, config):
        logger.info("Initializing kinetic-skeleton dataset...")
        
        # dataset configuration
        self._data_path = config.data_path
        self._label_path = config.label_path
        self._ignore_empty_sample = config.ignore_empty_sample
        self._max_people = config.max_people

        # get data filenames and load label
        self._data_names = np.array(os.listdir(self._data_path))
        with open(self._label_path,'r') as f:
            label_info = json.load(f)
        lnames = [l for l in label_info.keys()]
        self._labels = np.array([label_info[lname]["label_index"] for lname in lnames])
        
        # filter out data without skeleton label
        has_skeleton = np.array([label_info[lname]["has_skeleton"] for lname in lnames])
        if self._ignore_empty_sample:
            self._data_names = self._data_names[has_skeleton]
            self._labels = self._labels[has_skeleton]

        # data shape
        self._n_channels = 3
        self._n_frames = 300
        self._n_joints = 18

    def __getitem__(self, index):
        # load one piece of data
        dname = self._data_names[index]
        dpath = os.path.join(self._data_path,dname)
        with open(dpath, 'r') as f:
            video_info = json.load(f)

        # get label
        label = video_info["label_index"]

        # fill in data container
        for frame_info in video_info["data"]:
            frame_idx = frame_info["frame_index"]
            for pidx, skeleton_info in enumerate(frame_info["skeleton"]):
                if pidx>=self._max_people:
                    logger.debug("A sample contains more than {} people".format(self._max_people))
                    break
                pose = skeleton_info["pose"]
                score = skeleton_info["score"]
                logger.error("still working on...")
                ipdb.set_trace()

        return 1 #DEBUG

    def __len__(self):
        return len(self._labels)

    def name(self):
        return "KineticsSkeletonDataset"
