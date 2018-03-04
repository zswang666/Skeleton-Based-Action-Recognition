import os
import time
import random
import numpy as np
import argparse
import logging
import functools
import pprint
import matplotlib.pyplot as plt
import pickle

# torch
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

# local packages
from tool.utils import *

def main():
    # get configuration
    parser = get_parser()
    p = parser.parse_args()
    if p.config is not None:
        cfg = parse_yaml(p.config)
    output_dir = validate_dir(p.output_dir)

    # dataset and dataloader
    dataset = dynamic_import(cfg.dataset_train).Dataset()
    dataset.initialize(cfg.dataset_train_args)
    dataloader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=False, # Must be False here
                            drop_last=True,
                            num_workers=cfg.workers)

    # network
    model_cfg = dynamic_import(cfg.model).create_model_config(cfg, dataset)
    net = dynamic_import(cfg.model).Model(model_cfg)
    net.double()
    if cfg.cuda:
        net = net.cuda()
    #print(net)

    # load model
    ckpt = torch.load(p.model_path)
    net.load_state_dict(ckpt["net_state"])

    net.eval()
    inv_label_mapping = inverse_dict(dataset.label_mapping)
    results = dict()
    hist_data = dict()
    primitives_seq_list = []
    for cname in dataset.label_mapping.keys():
        results[cname] = []
        hist_data[cname] = []
    for i, data in enumerate(dataloader, 0):
        # unpack data
        annotation = data[0]
        adj_matrix = data[1]
        target = data[2]
        seq_length = data[3]

        # prepare data
        padding = torch.zeros(len(annotation), dataset.max_frames, \
                              dataset.n_joints, cfg.model_args.state_dim-cfg.dataset_train_args.annotation_dim).double()
        init_input = torch.cat((annotation, padding), 3)
        if cfg.cuda:
            init_input = init_input.cuda()
            adj_matrix = adj_matrix.cuda()
            annotation = annotation.cuda()
            target = target.cuda()
            seq_length = seq_length.cuda()

        init_input = Variable(init_input)
        adj_matrix = Variable(adj_matrix)
        annotation = Variable(annotation)
        target = Variable(target)
        seq_length = Variable(seq_length)

        # inference and dump to numpy
        output, primitives_prob, seg_length = net(init_input, annotation, adj_matrix, seq_length, True)

        # trim primitive prob
        primitives_prob = primitives_prob.data.cpu().numpy()[0]
        seg_length = seg_length.data.cpu().numpy()[0]
        primitives_seq = primitives_prob[:seg_length-1].argmax(axis=1)

        target_np = target.data.cpu().numpy()[0]
        
        primitives_seq_list.append(primitives_seq)
        results[inv_label_mapping[target_np]].append(primitives_seq)
        hist_data[inv_label_mapping[target_np]].extend(primitives_seq)
    pprint.pprint(results)

    # dump to file
    dump_data = {"results": results,
                 "primitives_seq_list": primitives_seq_list,
                 "hist_data": hist_data,
                 "data_list": dataset.data_list}
    pickle.dump(dump_data, open(os.path.join(output_dir,"vis_data.pkl"),"wb"))

    # histogram
    plot_and_save_hist(hist_data, output_dir, cfg.model_args.n_primitives)

    # state transition
    visualizer = EasyTransitionVis(cfg.model_args.n_primitives, output_dir)
    for k, v in results.items():
        visualizer.draw(v,k)

class EasyTransitionVis(object):
    def __init__(self, num_labels, output_dir):
        self._num_labels = num_labels
        self._output_dir = output_dir

        self._cmap = plt.get_cmap('PiYG', self._num_labels+1)
        cmap_list = [self._cmap(i)[:3] for i in range(self._cmap.N)]
        cmap_list[0] = (0.,0.,0.)
        self._cmap = self._cmap.from_list('Custom cmap', cmap_list, self._cmap.N)

    def draw(self, data, name):
        n_samples = len(data)

        # get max length
        max_len = 0
        for datum in data:
            datum_len = len(datum)
            if datum_len > max_len:
                max_len = datum_len

        # visualize data
        mat = np.zeros((n_samples,max_len))
        for idx, datum in enumerate(data):
            for t,d in enumerate(datum):
                mat[idx,t] = d + 1
        self._discrete_matshow(mat, name)

    def _discrete_matshow(self, mat, name):
        plt.clf()
        plt.imshow(mat, cmap=self._cmap, vmin=0, vmax=self._num_labels+1)
        colorbar_index(ncolors=self._num_labels+1, cmap=self._cmap)
        plt.title(name)
        plt.xlabel("mega-timesteps")
        plt.ylabel("samples")
        plt.savefig(os.path.join(self._output_dir,"transition_{}.jpg".format(name)))

def plot_and_save_hist(data, output_dir, n_primitives):
    for k, v in data.items():
        plt.clf()
        hist, _ = np.histogram(v, np.arange(n_primitives+1))
        plt.bar(np.arange(n_primitives), hist)
        plt.title(k)
        plt.xlabel("different primitives")
        plt.ylabel("counts")
        plt.savefig(os.path.join(output_dir,"hist_{}.jpg".format(k)))

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="configuration file (.yaml)")
    parser.add_argument("--model_path", type=str, help="trained model path (checkpoint file)")
    parser.add_argument("--output_dir", type=str, help="output directory")

    return parser

if __name__=="__main__":
    main()
