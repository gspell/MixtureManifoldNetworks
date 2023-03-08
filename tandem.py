"""
Definition of Tandem model in PyTorch

Also handles the construction of a Tandem/MMN model
"""
import torch
import torch.nn as nn

import pdb
cuda = True if torch.cuda.is_available() else False

class Forward(nn.Module):
    def __init__(self, layer_sizes, conv_layers=None):
        super(Forward, self).__init__()
        self.linears, self.bn = nn.ModuleList([]), nn.ModuleList([])
        self.convs_f = None
        for i in range(len(layer_sizes) - 1):
            in_size = layer_sizes[i]
            out_size = layer_sizes[i+1]
            self.linears.append(nn.Linear(in_size, out_size))
            self.bn.append(nn.BatchNorm1d(out_size))
        if conv_layers is not None:
            self.convs_f = nn.ModuleList([])
            in_channel = 1
            for idx, (out_channel, kernel_size, stride) in conv_layers.items():
                if stride == 2:
                    pad = int(kernel_size / 2 - 1)
                elif stride == 1:
                    pad = int((kernel_size - 1) / 2)
                else:
                    Exception("Now only support stride = 1 or 2, contact Ben")

                self.convs_f.append(nn.ConvTranspose1d(in_channel, out_channel,
                                                       kernel_size,
                                                       stride=stride,
                                                       padding=pad))
                in_channel = out_channel
        if self.convs_f:
            self.convs_f.append(nn.Conv1d(in_channel, out_channels=1,
                                          kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        """
        x -> y (forward)
        :param x: input
        :return: y (output)
        """
        out = x
        for i in range(len(self.linears)-1):
            out = nn.functional.relu(self.bn[i](self.linears[i](out)))
        out = self.linears[-1](out)
        if self.convs_f:
            out = out.unsqueeze(1)
            for idx, conv in enumerate(self.convs_f):
                out = conv(out)
            out = out.squeeze()
        return out

class Backward(nn.Module):
    def __init__(self, layer_sizes, conv_layers=None):
        super(Backward, self).__init__()
        self.linears, self.bn = nn.ModuleList([]), nn.ModuleList([])
        self.convs_b = None
        for i in range(len(layer_sizes) - 1):
            in_size = layer_sizes[i]
            out_size = layer_sizes[i+1]
            self.linears.append(nn.Linear(in_size, out_size))
            self.bn.append(nn.BatchNorm1d(out_size))
        if conv_layers is not None:
            self.convs_b = nn.ModuleList([])
            in_channel = 1
            for idx, (out_channel, kernel_size, stride) in conv_layers.items():
                if stride == 2:
                    pad = int(kernel_size / 2 -1)
                elif stride == 1:
                    pad = int((kernel_size - 1) / 2)
                else:
                    Exception("Now only support stride = 1 or 2, contact Ben")
                self.convs_b.append(nn.Conv1d(in_channel, out_channel,
                                              kernel_size,
                                              stride=stride,
                                              padding=pad))
                in_channel = out_channel
        if self.convs_b:
            self.convs_b.append(nn.Conv1d(in_channel, out_channels=1,
                                          kernel_size=1, stride=1, padding=0))
    def forward(self, y):
        """
        x <- y (forward)
        :param y: output
        :return: x (input)
        """
        out = y
        if self.convs_b:
            out = out.unsqueeze(1)
            for idx, conv in enumerate(self.convs_b):
                out = conv(out)
            out = out.squeeze(1)
        
        for i in range(len(self.linears)-1):
            out = nn.functional.relu(self.bn[i](self.linears[i](out)))
        out = self.linears[-1](out)
        return out

################################################################################
# MODEL CONSTRUCTION/LOADING
################################################################################
def load_model_checkpoint(model, model_path):
    print("Loading model from: {}".format(model_path))
    load_dict = torch.load(model_path, map_location=torch.device('cuda'))
    model.load_state_dict(load_dict)
    return model

def make_forward_model(layer_sizes_f, conv_layers=None, model_path=None):
    model_f = Forward(layer_sizes_f, conv_layers)
    if cuda:
        model_f.cuda()
    if model_path is not None:
        model_f = load_model_checkpoint(model_f, model_path)
    return model_f

def make_backward_model(layer_sizes_b, conv_layers=None, model_path = None):
    model_b = Backward(layer_sizes_b, conv_layers)
    if cuda:
        model_b.cuda()
    if model_path is not None:
        model_b = load_model_checkpoint(model_b, model_path)
    return model_b

def make_multiple_bwd_models(num_models, layer_sizes_b, conv_layers=None, model_paths = None):
    models_b = []
    for model_idx in range(num_models):
        if model_paths is None:
            model_b = make_backward_model(layer_sizes_b, conv_layers, None)
        else:
            model_b = make_backward_model(layer_sizes_b, conv_layers, model_paths[model_idx])
        model_b.eval()
        models_b.append(model_b)
    return models_b
    
