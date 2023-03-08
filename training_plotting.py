"""
training file
"""

import numpy as np
import os
import pdb
import pandas as pd
import time

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns

################################################################################
# PLOTTING OF TRAINING CURVES
################################################################################

def plot_fwd_model_losses(fwd_train_losses, fwd_eval_losses, out_dir):
    num_epochs = len(fwd_train_losses)
    num_evals  = len(fwd_eval_losses)
    eval_step  = int(num_epochs / num_evals)
    
    plt.figure()
    plt.title("Forward Model Losses (M.S.E.)", fontweight = "bold")
    plt.plot(range(num_epochs), fwd_train_losses, label = "Train")
    plt.plot(range(0, num_epochs, eval_step), fwd_eval_losses, label = "Eval")
    plt.xlabel("Epochs", fontweight = "bold")
    plt.ylabel("Mean-Square-Error", fontweight = "bold")
    plt.legend()
    plt.yscale("log")
    plt.grid()
    filename = os.path.join(out_dir, "fwd_model_losses_during_training.png")
    plt.savefig(filename, bbox_inches = "tight")
    plt.close()

def plot_bwd_model_mse_losses(bwd_train_losses, bwd_eval_losses,
                              model_idx, out_dir):
    num_epochs = len(bwd_train_losses)
    num_evals  = len(bwd_eval_losses)
    eval_step  = int(num_epochs / num_evals)

    plt.figure()
    plt.title("Backward Model Losses", fontweight = "bold")
    plt.plot(range(num_epochs), bwd_train_losses, label = 'Train')
    
    eval_num_epochs = range(0, num_epochs, eval_step)    
    plt.plot(eval_num_epochs, bwd_eval_losses, label = "Eval")

    plt.grid()
    plt.legend()
    plt.xlabel('Epochs', fontweight = "bold")
    plt.ylabel('M.S.E Loss', fontweight = "bold")
    plt.yscale("log")
    filename = "bwd_model_{}_mse_losses_during_training.png".format(model_idx)
    filename = os.path.join(out_dir, filename)
    plt.savefig(filename, bbox_inches = "tight")
    plt.close()

def plot_bwd_model_rep_losses(bwd_train_losses, bwd_eval_losses,
                              model_idx, out_dir):
    num_epochs = len(bwd_train_losses)
    num_evals  = len(bwd_eval_losses)
    eval_step  = int(num_epochs / num_evals)

    plt.figure()
    plt.title("Backward Model Losses", fontweight = "bold")
    plt.plot(range(num_epochs), bwd_train_losses, label = 'Train')
    
    eval_num_epochs = range(0, num_epochs, eval_step)    
    plt.plot(eval_num_epochs, bwd_eval_losses, label = "Eval")

    plt.grid()
    plt.legend()
    plt.xlabel('Epochs', fontweight = "bold")
    plt.ylabel('Repulsion Loss', fontweight = "bold")
    plt.yscale("log")
    filename = "bwd_model_{}_rep_losses_during_training.png".format(model_idx)
    filename = os.path.join(out_dir, filename)
    plt.savefig(filename, bbox_inches = "tight")
    plt.close()
