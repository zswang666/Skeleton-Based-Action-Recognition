from collections import namedtuple

import torch
import torch.nn
from torch.autograd import Variable

from .networks import *
from .ggnn import Model as GGNN

class AttrProxy(object):
    """Translates index lookups into attribute lookups."""
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))

class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()

        # model configuration
        self._n_primitives = cfg.n_primitives
        self._primitive_range = cfg.primitive_range
        self._n_segments = int(cfg.max_frames/self._primitive_range)
        self._state_dim = cfg.state_dim
        self._n_node = cfg.n_node
        self._annotation_dim = cfg.annotation_dim
        self._cuda = cfg.cuda

        # primitive bank
        self._ggnns = ListModule(self, "ggnn_")
        for i in range(self._n_primitives):
            self._ggnns.append(nn.Sequential(GGNN(cfg),
                                             nn.Linear(self._n_node,1)))

        # output model
        self._rnn = nn.LSTM(
            input_size=self._n_primitives,
            hidden_size=self._rnn_hidden_size,
            num_layers=self._rnn_layers,
            batch_first=True,
            dropout=0.5
        )
        self._out = nn.Linear(self._rnn_hidden_size, self._output_size)

    def forward(self, init_input, data, adj_matrix, seq_length, output_primitives=False):
        # run ggnns for each segment (with length=primitive_range)
        primitive_along_time = []
        for i in range(self._n_segments):
            # obtain data segment according to primitive_range
            seg_range = range(i*self._primitive_range, (i+1)*self._primitive_range)
            init_seg = init_input[:,seg_range]
            data_seg = data[:,seg_range]
            # ravel time axis, TODO: reshape here consistent to adjacency matrix?!
            init_seg = init_seg.view(-1,self._n_node,self._state_dim)
            data_seg = data_seg.view(-1,self._n_node,self._annotation_dim)
            # inference through primitive bank (ggnns)
            ggnn_out = []
            for ggnn in self._ggnns:
                ggnn_out.append(ggnn([init_seg,data_seg,adj_matrix]))
            primitive_along_time.append(torch.cat(ggnn_out,dim=1))
        primitive_along_time = torch.stack(primitive_along_time, dim=1)

        # run RNN
        r_out, (h_n, h_c) = self._rnn(primitive_along_time, None)
        r_out = r_out
        # trim the sequences
        seg_length = seq_length / self._primitive_range - 1
        flattened_r_out = r_out.contiguous().view(r_out.shape[0]*r_out.shape[1], self._rnn_hidden_size)
        offset = Variable(torch.arange(seg_length.shape[0]).long()*self._n_segments)
        if self._cuda:
            offset = offset.cuda()
        masked_r_out = flattened_r_out[offset+seg_length]

        # output layer for classification
        out = self._out(masked_r_out)

        if output_primitives:
            return out, primitive_along_time, seg_length
        else:
            return out

    def load_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            own_state[name].copy_(param)
       
def create_model_config(cfg, dataset):
    model_cfg = dict()
    model_cfg["n_node"] = dataset.n_nodes
    model_cfg["n_edge_types"] = cfg.dataset_train_args.n_edge_types
    model_cfg["n_steps"] = cfg.model_args.n_steps
    model_cfg["state_dim"] = cfg.model_args.state_dim
    model_cfg["annotation_dim"] = cfg.dataset_train_args.annotation_dim
    model_cfg["primitive_range"] = cfg.dataset_train_args.primitive_range
    model_cfg["n_primitives"] = cfg.model_args.n_primitives
    model_cfg["max_frames"] = dataset.max_frames
    model_cfg["rnn_layers"] = cfg.model_args.rnn_layers
    model_cfg["rnn_hidden_size"] = cfg.model_args.rnn_hidden_size
    model_cfg["output_size"] = cfg.model_args.output_size
    model_cfg["cuda"] = cfg.cuda
    model_cfg = namedtuple("GenericDict", model_cfg.keys())(**model_cfg)

    return model_cfg
