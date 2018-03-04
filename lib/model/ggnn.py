import torch
import torch.nn

from .networks import *

class AttrProxy(object):
    """Translates index lookups into attribute lookups."""
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))

class Model(nn.Module):
    """
    Gated Graph Sequence Neural Networks (GGNN)
    Mode: SelectNode
    Implementation based on https://arxiv.org/abs/1511.05493
    see https://github.com/JamesChuanggg/ggnn.pytorch/blob/master/model.py
    """
    def __init__(self, opt):
        super(Model, self).__init__()

        assert (opt.state_dim >= opt.annotation_dim,  \
                'state_dim must be no less than annotation_dim')

        self.state_dim = opt.state_dim
        self.annotation_dim = opt.annotation_dim
        self.n_edge_types = opt.n_edge_types
        self.n_node = opt.n_node
        self.n_steps = opt.n_steps

        '''
        self.in_fcs = ListModule(self, "in_")
        self.out_fcs = ListModule(self, "out_")
        for i in range(self.n_edge_types):
            # incoming and outgoing edge embedding
            self.in_fcs.append(nn.Linear(self.state_dim, self.state_dim))
            self.out_fcs.append(nn.Linear(self.state_dim, self.state_dim))
        '''
        for i in range(self.n_edge_types):
            # incoming and outgoing edge embedding
            in_fc = nn.Linear(self.state_dim, self.state_dim)
            out_fc = nn.Linear(self.state_dim, self.state_dim)
            self.add_module("in_{}".format(i), in_fc)
            self.add_module("out_{}".format(i), out_fc)

        self.in_fcs = AttrProxy(self, "in_")
        self.out_fcs = AttrProxy(self, "out_")


        # Propagation Model
        self.propagator = LSTMPropagator(self.state_dim, self.n_node, self.n_edge_types)

        # Output Model
        self.out = nn.Sequential(
            nn.Linear(self.state_dim + self.annotation_dim, self.state_dim),
            nn.Tanh(),
            nn.Linear(self.state_dim, 1)
        )

        self._initialization()

    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, data):
        prop_state = data[0]
        annotation = data[1]
        A = data[2]
        for i_step in range(self.n_steps):
            in_states = []
            out_states = []
            for i in range(self.n_edge_types):
                in_states.append(self.in_fcs[i](prop_state))
                out_states.append(self.out_fcs[i](prop_state))
            in_states = torch.stack(in_states).transpose(0, 1).contiguous()
            in_states = in_states.view(-1, self.n_node*self.n_edge_types, self.state_dim)
            out_states = torch.stack(out_states).transpose(0, 1).contiguous()
            out_states = out_states.view(-1, self.n_node*self.n_edge_types, self.state_dim)

            prop_state = self.propagator(in_states, out_states, prop_state, A)

        join_state = torch.cat((prop_state, annotation), 2)

        output = self.out(join_state)
        output = output.sum(2)

        return output
