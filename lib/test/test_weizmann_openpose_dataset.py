import os
import attrdict
import logging
import ipdb

from data.weizmann_openpose_dataset import Dataset

logging.basicConfig(level=logging.INFO)

def main():
    print(os.path.abspath(".."))
    test_dataset()

def test_dataset():
    print("Testing WEIZMANN-openpose dataset")
    cfg = attrdict.AttrDict()
    cfg.data_path = "../datasets/WEIZMANN/openpose/keypoints"
    cfg.ignore_empty_sample = True
    cfg.use_adjacency_matrix = True
    cfg.ignore_uncertain_node = False # not yet
    cfg.node_certainty_threshold = 0.3
    cfg.primitive_range = 5
    cfg.n_edge_types = 2

    dataset = Dataset()
    dataset.initialize(cfg)
    for data in dataset:
        ipdb.set_trace()

def test_labelmapping():
    raise NotImplementedError

def test_dataloader():
    raise NotImplementedError

if __name__=="__main__":
    main()
