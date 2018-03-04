import torch
import torch.nn as nn

class ListModule(object):
    '''
    Create a list of that contains pytorch nn module.
    Usage:
        # self is a nn.Module instance
        >> nnlist = ListModule(self, "test_")
        >> for i in range(3): 
        >>     nnlist.append(nn.Linear(2,3))
    see https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/4
    '''
    def __init__(self, module, prefix, *args):
        self.module = module
        self.prefix = prefix
        self.num_module = 0
        for new_module in args:
            self.append(new_module)

    def append(self, new_module):
        if not isinstance(new_module, nn.Module):
            raise ValueError("Not a Module")
        else:
            self.module.add_module(self.prefix + str(self.num_module), new_module)
            self.num_module += 1

    def __getitem__(self, i):
        if i<0 or i>=self.num_module:
            raise IndexError("Out of bound")
        return getattr(self.module, self.prefix + str(i))

    def __len__(self):
        return self.num_module

class LSTMPropagator(nn.Module):
    '''
    Gated Propagator for GGNN
    Using LSTM gating mechanism
    see https://github.com/JamesChuanggg/ggnn.pytorch/blob/master/model.py
    '''
    def __init__(self, state_dim, n_node, n_edge_types):
        super(LSTMPropagator, self).__init__()

        self.n_node = n_node
        self.n_edge_types = n_edge_types

        self.reset_gate = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Sigmoid()
        )
        self.update_gate = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Sigmoid()
        )
        self.tansform = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Tanh()
        )

    def forward(self, state_in, state_out, state_cur, A):
        A_in = A[:, :, :self.n_node*self.n_edge_types]
        A_out = A[:, :, self.n_node*self.n_edge_types:]

        a_in = torch.bmm(A_in, state_in)
        a_out = torch.bmm(A_out, state_out)
        a = torch.cat((a_in, a_out, state_cur), 2)

        r = self.reset_gate(a)
        z = self.update_gate(a)
        joined_input = torch.cat((a_in, a_out, r * state_cur), 2)
        h_hat = self.tansform(joined_input)

        output = (1 - z) * state_cur + z * h_hat

        return output
