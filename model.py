import torch
import torch.nn.functional as F
from module import *

class Net(nn.Module):
    def __init__(self, max_depth, max_number_child, device, embed_dict, cgf_input_dim, cgf_bias, rst_input_dim, n_layers,
                 rst_bias, rst_drop_prob=0.2, rst_bidirect=True, cgf_drop_prob=0.2, cgf_bidirect=True):
        super(Net, self).__init__()
        self.device = device
        self.n_layers = n_layers

        self.cgf_input_dim = cgf_input_dim
        self.cgf_bidirectional = cgf_bidirect
        #modify here for h, c in cfg and rst
        if self.cgf_bidirectional:
            self.cgf_rnn = nn.GRU(self.cgf_input_dim,  self.cgf_input_dim//2, self.n_layers, cgf_bias, dropout=cgf_drop_prob,
                                   bidirectional=cgf_bidirect)
        else:
            self.cgf_rnn = nn.GRU(self.cgf_input_dim,  self.cgf_input_dim, self.n_layers, cgf_bias, dropout=cgf_drop_prob,
                                   bidirectional=cgf_bidirect)

        self.cgf = CGF(self.cgf_rnn, embed_dict, self.n_layers, self.cgf_bidirectional, self.cgf_input_dim, max_depth, max_number_child)

        self.RST_depth = max_depth
        self.rst_input_dim = rst_input_dim
        self.rst_bidirectional = rst_bidirect
        if self.rst_bidirectional:
            self.rst_rnn = nn.GRU(self.rst_input_dim, self.rst_input_dim//2, self.n_layers, rst_bias, dropout=rst_drop_prob,
                                   bidirectional=rst_bidirect)
        else:
            self.rst_rnn = nn.GRU(self.rst_input_dim, self.rst_input_dim, self.n_layers, rst_bias, dropout=rst_drop_prob,
                                   bidirectional=rst_bidirect)


        self.RST = RST(self.rst_rnn, self.RST_depth, self.cgf, self.n_layers, self.rst_bidirectional,  self.rst_input_dim)
        self.final_fc = nn.Linear(100, 2)

        self.m = nn.Softmax(dim=2)

    def forward(self, root, bodytext):
        out = self.RST.forward(root, bodytext, 0, self.device)
        out = self.m(self.final_fc(out))

        return out
