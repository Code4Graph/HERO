import torch
import torch.nn as nn
import string
import sys
import random
import numpy as np

class RST(nn.Module):
    def __init__(self, gru, RST_depth, cgf, n_layers, rst_bidirect, rst_input_dim):
        super(RST, self).__init__()
        self.gru = gru
        self.depth = RST_depth
        self.cgf = cgf
        self.n_layers = n_layers
        self.rst_bidirectional = rst_bidirect
        self.rst_input_dim = rst_input_dim

    # here for handling RST
    def post_order_traversal_RST(self, root, bodytext, layer, device):
        if root.label() == 'EDU':
            if int(root[0]) in bodytext:
                out = self.cgf.forward(bodytext[int(root[0])], layer, device)
            else:
                out = torch.zeros_like(torch.empty(1, 1, self.rst_input_dim )).to(device)

            if self.rst_bidirectional:
                h = torch.zeros(2*self.n_layers, 1, self.rst_input_dim//2).to(device)
            else:
                h = torch.zeros(self.n_layers, 1, self.rst_input_dim).to(device)
            return out, h

        len_root = len(root)
        if len_root == 2:
            left_embed, h_left = self.post_order_traversal_RST(root[0], bodytext, layer + 1, device)
            right_embed, h_right = self.post_order_traversal_RST(root[1], bodytext,  layer + 1, device)
        else:
            left_embed, h_left = self.post_order_traversal_RST(root[0], bodytext, layer + 1, device)
            right_embed, h_right = None, None


        if layer <= self.depth:
            if (left_embed is not None) and (right_embed is  not None):
                h = torch.mean(torch.stack([h_left, h_right], 0), dim=0)
                inpt = torch.cat((left_embed, right_embed), 0)
                out, h = self.gru(inpt, h)
                out = torch.mean(out, 0, True)
            else:
                out = left_embed
                h = h_left
        else:
            if (left_embed is not None) and (right_embed is not None):
                out = torch.mean(torch.cat((left_embed, right_embed), 0), 0, True)
            else:
                out = left_embed
            if self.rst_bidirectional:
                h = torch.zeros(2 * self.n_layers, 1, self.rst_input_dim // 2).to(device)
            else:
                h = torch.zeros(self.n_layers, 1, self.rst_input_dim).to(device)
        return out, h

    def forward(self, root, bodytext, layer, device):
        res, _ = self.post_order_traversal_RST(root, bodytext, layer, device)
        return res

class CGF(nn.Module):
    def __init__(self, gru, embed_dictionary, n_layers, cgf_bidirectional, cgf_input_dim, depth, max_child_number):
        super(CGF, self).__init__()
        self.gru = gru
        self.embed_dict = embed_dictionary
        self.n_layers= n_layers
        self.cgf_bidirectional = cgf_bidirectional
        self.cgf_input_dim = cgf_input_dim
        self.depth = depth
        self.max_child_number = max_child_number

    def post_order_traverse1_CGF(self, root, layer,  device):
        # if root is not None:
        if isinstance(root, str):
            if self.cgf_bidirectional:
                h = torch.zeros(2 * self.n_layers, 1, self.cgf_input_dim // 2).to(device)
            else:
                h = torch.zeros(self.n_layers, 1, self.cgf_input_dim).to(device)
            root = root.lower()
            if root in self.embed_dict:
                return torch.Tensor(self.embed_dict[root].reshape(1, 1, -1)).to(device), h
            else:
                return torch.zeros_like(torch.empty(1, 1, self.cgf_input_dim)).to(device), h

        if len(root) == 0:
            if self.cgf_bidirectional:
                h = torch.zeros(2 * self.n_layers, 1, self.cgf_input_dim // 2).to(device)
            else:
                h = torch.zeros(self.n_layers, 1, self.cgf_input_dim).to(device)
            return torch.zeros_like(torch.empty(1, 1, self.cgf_input_dim)).to(device), h

        if layer <= self.depth:
            stack_x = None
            stack_h = []

            # child end cut
            # for i in range(0, min(len(root),self.max_child_number)):
            stop = -1 if len(root) < self.max_child_number else len(root) - 1 - self.max_child_number
            for i in range(len(root) - 1, stop, -1):
                x, h = self.post_order_traverse1_CGF(root[i], layer + 1, device)
                if i == len(root) - 1:
                    stack_x = x
                else:
                    stack_x = torch.cat((stack_x, x), 0)
                stack_h.append(h)

            stack_h = torch.mean(torch.stack(stack_h, 0), dim=0)
            out, h = self.gru(stack_x, stack_h)
            out = torch.mean(out, dim=0, keepdim=True)

            return out, h

        else:
            stack_x = None
            if self.cgf_bidirectional:
                h = torch.zeros(2 * self.n_layers, 1, self.cgf_input_dim // 2).to(device)
            else:
                h = torch.zeros(self.n_layers, 1, self.cgf_input_dim).to(device)

            # child end cut
            # for i in range(0, min(len(root), self.max_child_number)):
            stop = -1 if len(root) < self.max_child_number else len(root) - 1 - self.max_child_number
            for i in range(len(root) - 1, stop, -1):
                x, _ = self.post_order_traverse1_CGF(root[i], layer + 1, device)
                if i == len(root) - 1:
                    stack_x = x
                else:
                    stack_x = torch.cat((stack_x, x), 0)

            return stack_x, h

    def forward(self, root, layer, device):
        res, _  = self.post_order_traverse1_CGF(root, layer, device)
        return res




