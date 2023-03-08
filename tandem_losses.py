"""
Definitions for loss functions associated with training Tandem models (and MMTN)
"""

import numpy as np
import os
import pdb
import pandas as pd
import time

import torch
import torch.nn as nn

cuda = True if torch.cuda.is_available() else False

################################################################################
# DEFINE LOSS FUNCTIONS
################################################################################

def boundary_loss(x, task = "sine"):
    """
    Used to calculate boundary loss predicted x
    :param x: predicted input tensor
    :return: boundary loss
    """
    if task == "sine":
        mean = np.array([0, 0])
        X_range = np.array([2, 2])
    elif task == "1d_sine":
        mean = np.array([0])
        X_range = np.array([2])
    elif task == "robotic_arm":
        # These were the values originally found in Ben's Tandem code
        # I have changed them to account for my own understanding
        # (e.g., what I *think* they actually should be)
        """
        X_lower = np.array([-0.87, -1.87, -1.92, -1.73])
        X_upper = np.array([1.018, 1.834, 1.897, 2.053])
        X_range = np.array([1.88, 3.7, 3.82, 3.78])
        mean = (X_lower + X_upper) / 2
        """
        """
        # This is what I find in Ben's NA code -- so probably what he used?
        X_lower = np.array([-0.6, -1.2, -1.2, -1.2])
        X_upper = np.array([0.6, 1.2, 1.2, 1.2])
        X_range = np.array([1.2, 2.4, 2.4, 2.4])
        mean = (X_lower + X_upper) / 2
        """

        # Below, this is THREE STD away from MEAN in each dim
        X_lower = np.array([-0.75, -1.5, -1.5, -1.5])
        X_upper = np.array([0.75, 1.5, 1.5, 1.5])
        X_range = np.array([1.5, 3.0, 3.0, 3.0])
        mean = (X_lower + X_upper) / 2

        """
        # Below, this is TWO STD away from MEAN in each dim
        X_lower = np.array([-0.5, -1.0, -1.0, -1.0])
        X_upper = np.array([0.5, 1.0, 1.0, 1.0])
        X_range = np.array([1.0, 2.0, 2.0, 2.0])
        mean = (X_lower + X_upper) / 2
        """
    elif task == "ballistics":
        # NOTE THAT BEN'S TANDEM CODE HAS DIFFERENT BOUNDS THAN THE NA CODE!!!
        dim_1 = [-1, 1]
        dim_2 = [0.5, 2.5]
        dim_3 = [np.pi/2 * 0.1, np.pi/2 * 0.8 + np.pi/2 * 0.1]
        #dim_4 = [0.4836, 1.5164]
        #dim_4 = [0., 2.]
        #dim_4 = [0.33333, 1.66667]
        #dim_4 = [0.166667, 1.833333]
        #dim_4 = [0.5, 1.5]
        #dim_4 = [0.08333333, 1.91666667]
        #dim_4 = [0.46667, 1.53333]
        dim_4 = [0.0666667, 1.933333]
        """
        X_lower = np.array([dim_1[0], dim_2[0], dim_3[0], dim_4[0]])
        X_upper = np.array([dim_1[1], dim_2[1], dim_3[1], dim_4[1]])
        X_range = (X_upper - X_lower)
        mean = (X_lower + X_upper) / 2
        """
        # These were the values originally found in Ben's Tandem code
        # I have changed them to account for my own understanding AND
        # because I changed the distributions used at data generation
        """
        X_lower = np.array([-1, 0.5, 0.157, 0.46])
        X_upper = np.array([1, 2.5, 1.256, 1.46])
        X_range = np.array([2, 2, 1.099, 1])
        mean = (X_lower + X_upper) / 2
        """
        # This is for replacing the Poisson distribution with a Gaussian
        """
        X_lower = np.array([-1.0, 0.5, 0.157, 0.4836])
        X_upper = np.array([1., 2.5, 1.413, 1.5164])
        X_range = np.array([2.0, 2.0, 1.256, 1.0328])
        mean = (X_lower + X_upper) / 2
        """
        """
        # this must be what I used the first time after changing distn
        # this was for all uniform distns
        X_lower = np.array([-1, 0.5, 0.157, 0])
        X_upper = np.array([1, 2.5, 1.256, 2])
        X_range = np.array([2, 2, 1.099, 1])
        mean = np.array([0, 1.5, 0.7854, 1])
        #mean = (X_lower + X_upper) / 2
        """

        # We do actually know the true lower bound for final dim -- 0 (poisson)
        X_lower = np.array([-1.0, 0.5, np.pi/2 * 0.1, 0.467])
        X_upper = np.array([1.0, 2.5, np.pi/2 * 0.8 + np.pi/2 * 0.1, 1.567])
        X_range = np.array([2.0, 2.0, np.pi/2 * 0.8, 1.1])
        mean = np.array([0, 1.5, np.pi / 4, 1.0]) # np.pi / 4 = 0.785

        
    elif task == "meta_material":
        X_lower = np.array([-1,-1,-1,-1,-1,-1,-1,-1])
        X_upper = np.array([1.272,1.272,1.272,1.272,1,1,1,1])
        X_range = np.array([2.272,2.272,2.272,2.272,2,2,2,2])
        mean = (X_lower + X_upper) / 2
    elif task == "peurifoy":
        dim = 8
        X_lower = np.array([-1 for i in range(dim)])
        X_upper = np.array([1 for i in range(dim)])
        X_range = np.array([2 for i in range(dim)])
        mean = (X_lower + X_upper) / 2

    if torch.cuda.is_available():
        input_diff = torch.abs(x - torch.tensor(mean, dtype=torch.float, device='cuda'))
        mean_diff = input_diff - 0.5*torch.tensor(X_range, dtype = torch.float, device='cuda')
    else:
        input_diff = torch.abs(x - torch.tensor(mean, dtype=torch.float))
        mean_diff = input_diff - 0.5 * torch.tensor(X_range, dtype=torch.float)
    relu = nn.ReLU()
    bdy_loss = relu(mean_diff)
    if torch.cuda.is_available():
        bdy_loss = bdy_loss.cuda()
    return torch.mean(bdy_loss)

def compute_repulsion_loss(prev, x, sigma = 1.0):
    # prev is passed in as: model_b(s) -- inverse model operating on s
    repulsion_distance = torch.subtract(prev, x)
    repulsion_distance = torch.square(repulsion_distance) / (sigma ** 2)
    repulsion_loss = 1 - torch.exp(-1 * repulsion_distance)
    repulsion_loss = torch.mean(repulsion_loss)
    # Note, I haven't weighted it yet -- needs l * repulsion_loss
    return repulsion_loss

def compute_repulsion_loss_multiple_models(bwd_model_preds, x, sigma = 1.0):
    repulsion_loss = 0.0
    for r in range(len(bwd_model_preds)):
        repulsion_loss += compute_repulsion_loss(bwd_model_preds[r], x, sigma)
    return repulsion_loss
                   
def loss(ys, labels, x = None, prev = None, sigma = 1, l=0, epoch = 0, num_epochs = 500):
    global sine
    bdy_loss = 0.0
    repulsion_loss = 0.0

    # Note: Ben does use nn.functional.mse_loss for his loss during training
    mse_loss = nn.functional.mse_loss(ys, labels) 

    if x is not None:
        bdy_loss = boundary_loss(x)
    
    if prev is not None and len(prev) > 0:
        for r in range(len(prev)):
            repulsion_loss += compute_repulsion_loss(prev[r], x, sigma)
            anneal_factor = (num_epochs - epoch) / num_epochs
            #repulsion_loss += anneal_factor * repulsion_loss
            #prev_loss += torch.mean((500-epoch)/500*l*(1-(torch.exp(-1 * torch.square(torch.subtract(prev[r], x)) / (sigma ** 2)))))
    
    #return 100*bdy + mse - prev_loss    
    return bdy_loss + mse_loss - l * repulsion_loss
