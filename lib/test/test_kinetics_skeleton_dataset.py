import os
import attrdict
import logging
import ipdb

from data.kinetics_skeleton_dataset import KineticsSkeletonDataset

logging.basicConfig(level=logging.INFO)

def main():
    print(os.path.abspath(".."))
    test_dataset()

def test_dataset():
    print("Testing kinetics-skeleton dataset")
    cfg = attrdict.AttrDict()
    cfg.data_path = "../datasets/kinetics-skeleton/kinetics_train"
    cfg.label_path = "../datasets/kinetics-skeleton/kinetics_train_label.json"
    cfg.ignore_empty_sample = True
    cfg.max_people = 2

    dataset = KineticsSkeletonDataset()
    dataset.initialize(cfg)
    for data in dataset:
        ipdb.set_trace()

def test_labelmapping():
    raise NotImplementedError

def test_dataloader():
    raise NotImplementedError

if __name__=="__main__":
    main()
