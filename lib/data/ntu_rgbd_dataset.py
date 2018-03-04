# sys
import os
import sys
import numpy as np
import random
import logging

# pytorch
import torch

# local packages
from .base_dataset import BaseDataset
from .utils import *

# define logger
logger = logging.getLogger(__name__)

class Dataset(BaseDataset):
    def initialize(self, config):
        logger.info("Initializing ntu-rgbd dataset, currently doing nothing...")
        # dataset configuration
        '''
        self._data_path = config.data_path
        self._label_path = config.label_path
        self._random_choose = config.random_choose
        self._random_shift = config.random_shift
        self._window_size = config.window_size
        self._mean_subtraction = config.mean_subtraction
        self._temporal_downsample_step = config.temporal_downsample_step
        self._normalization = config.normalization
        '''

    def __getitem__(self, index):
        raise NotImplementedError
    
    def __len__(self):
        raise NotImplementedError
    
    def name(self):
        return "NTU_RGBD_Dataset"
