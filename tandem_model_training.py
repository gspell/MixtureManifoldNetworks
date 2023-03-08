"""
training file
"""

import numpy as np
import os
import pdb
import pandas as pd

import torch
import torch.nn as nn
from torch.optim import lr_scheduler

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm

import training_plotting as plotting

import tandem_losses as losses
cuda = True if torch.cuda.is_available() else False

################################################################################
# GENERIC TRAINING UTILITIES
################################################################################
def make_optimizer_f(model, train_params):
    if train_params['optim'] == 'Adam':
        opt = torch.optim.Adam(model.parameters(),
                               lr=train_params['learning_rate'],
                               weight_decay=train_params['reg_scale'])
    else:
        msg = 'Optimizer has always been Adam so far -- check optimizer'
        raise Exception(msg)
    return opt

def make_optimizer_b(model, train_params):
    # Note that in Ben's code, he doesn't use weight decay in bwd model opt
    if train_params['optim'] == 'Adam':
        opt = torch.optim.Adam(model.parameters(),
                               lr=train_params['learning_rate'])
    else:
        msg = 'Optimizer has always been Adam so far -- check optimizer'
        raise Exception(msg)
    return opt

def make_lr_scheduler(optm, train_params):
    # Note that Ben seems to also use a LR scheduler like this
    # Decrease LR when the validation error stops improving
    # THE FACTOR CAN CHANGE BY TASK -- flags.lr_decay_rate in Ben's code
    lr_sched = lr_scheduler.ReduceLROnPlateau(
        optimizer=optm, mode='min', factor=train_params['lr_decay_rate'],
        patience=10, verbose=True, threshold=1e-4)
    return lr_sched

################################################################################
# FORWARD MODEL TRAINING -- COMMON TO TANDEM AND MMN
################################################################################
def fwd_model_training(model_f, train_data, test_data, train_params,
                       out_dir, results_file, train_plot_dir):
    opt_f = make_optimizer_f(model_f, train_params)
    lr_sched = make_lr_scheduler(opt_f, train_params)
    fwd_model_losses = train_fwd_model(
        train_data, test_data, opt_f, lr_sched, model_f,
        train_params, out_dir)
    msg = "Forward Model Final Eval Error: {}"
    print(msg.format(fwd_model_losses[1][-1]))
    results_file.write((msg + "\n").format(fwd_model_losses[1][-1]))
    plotting.plot_fwd_model_losses(fwd_model_losses[0], fwd_model_losses[1],
                                   train_plot_dir)
    
def train_fwd_model(train_data, test_data, opt_fwd, lr_sched, model_fwd,
                    train_params, out_dir):
    print("\nTraining Forward Model")
    forward_train_losses, forward_eval_losses = [], []
    best_err_f = 20 # arbitrarily high, I guess?
    # TRAIN LOOP FOR FORWARD MODEL
    for epoch in range(train_params["num_epochs"]):
        # CALL TRAINING FOR AN EPOCH
        epoch_losses = train_fwd_model_epoch(train_data, opt_fwd, model_fwd)
        epoch_loss = np.mean(epoch_losses)
        forward_train_losses.append(epoch_loss)
        lr_sched.step(epoch_loss)

        if epoch % train_params["eval_step"] == 0:
            print("Eval epoch " + str(epoch))
            eval_epoch_losses = eval_fwd_model(test_data, model_fwd)
            eval_epoch_loss = np.mean(eval_epoch_losses)
            if epoch_loss < best_err_f:
                best_err_f = epoch_loss
                torch.save(model_fwd.state_dict(), "{}/modelf.pt".format(out_dir))
            forward_eval_losses.append(eval_epoch_loss)
            msg = "Forward train loss on epoch {}: {}"
            print(msg.format(epoch, forward_train_losses[-1]))
            msg = "Forward eval loss on epoch {}: {}"
            print(msg.format(epoch, eval_epoch_loss))
    # I believe the model/optimizer should *not* need to be returned, but check
    return forward_train_losses, forward_eval_losses

def train_fwd_model_epoch(train_data, opt_fwd, model_fwd):
    """
    Forward model maps from X-space to Y-space.
    Also consider it D-space to R-space in Tandem paper 
        - (From "design" to "response")
    Note Karthik has called the x's to be g (geometry) and the y's to be s (spectra)
    """
    model_fwd.train()
    epoch_losses = []
    for i, (x, y) in enumerate(train_data):
        
        if cuda:
            x = x.cuda()
            y = y.cuda()
        opt_fwd.zero_grad()
        y_hat = model_fwd(x) # out is in 'y-space' -- s is the truth in that space
        # MSE between out and s (no boundary loss)       
        #-- note the loss is for difference in Y-space        
        l = losses.loss(y_hat, y, x = None) 
        l.backward()
        opt_fwd.step()
        epoch_losses.append(l.cpu().detach().numpy())
    # I believe the model/optimizer should *not* need to be returned, but check
    return epoch_losses

def eval_fwd_model(test_data, model_fwd):
    model_fwd.eval()
    eval_epoch_losses = []
    #test_loss = 0
    for i, (g, s) in enumerate(test_data):

        if cuda:
            g = g.cuda()
            s = s.cuda()

        out = model_fwd(g)
        l = losses.loss(out, s, x = None)
        #test_loss += l
        eval_epoch_losses.append(l.cpu().detach().numpy())
    return eval_epoch_losses
    #return test_loss.cpu().data.numpy() / (i + 1)


################################################################################
# BACKWARD MODEL TRAINING -- SEPARATE FOR TANDEM AND MMN
################################################################################
def train_MMTN_bwd_model(train_data, test_data, opt, lr_sched, model_fwd,
                         models_bwd, model_idx, train_params, out_dir):
    """ This is to train a single backward model (among several), possibly 
        with repulsion loss, so the others are needed too """
    models_bwd[model_idx].train()
    model_fwd.eval()
    for n in range(0, model_idx):
        models_bwd[n].eval()
    
    num_models = len(models_bwd)
    print("\nTraining Inverse Model {} of {}".format(model_idx + 1, num_models))
    
    best_err = float("inf")

    bwd_train_mse_losses, bwd_eval_mse_losses = [], []
    bwd_train_rep_losses, bwd_eval_rep_losses = [], []
    for epoch in range(train_params["num_epochs"]): 
        epoch_mse_loss, epoch_rep_loss, epoch_loss = train_MMN_bwd_model_epoch(
            train_data, opt, model_fwd, models_bwd, model_idx, epoch, train_params)
        bwd_train_mse_losses.append(epoch_mse_loss)
        bwd_train_rep_losses.append(epoch_rep_loss)

        lr_sched.step(epoch_loss)

        if epoch % train_params["eval_step"] == 0:
            models_bwd[model_idx].eval()
            print("Eval epoch " + str(epoch))
            eval_epoch_mse_losses = []
            eval_epoch_rep_losses = []
            eval_epoch_losses = []
            for i, (g, s) in enumerate(test_data):
                if cuda:
                    g = g.cuda()
                    s = s.cuda()
                g_out = models_bwd[model_idx](s)
                s_out = model_fwd(g_out)
                prev = []
                """
                for p_idx in range(0, model_idx):
                    prev.append(models_bwd[p_idx](s))
                """
                l = losses.loss(s_out, s, x=g_out, prev=prev, sigma=train_params["sigma"],
                                l=train_params["lam"], epoch=epoch, num_epochs = train_params["num_epochs"])
                mse_loss = nn.functional.mse_loss(s, s_out)
                rep_loss = torch.tensor([0.0], device = "cuda")
                if prev is not None and len(prev) > 0:
                    for r in range(len(prev)):
                        rep_loss += losses.compute_repulsion_loss(
                            prev[r], g_out, train_params["sigma"])
                eval_epoch_mse_losses.append(mse_loss.cpu().detach().numpy())
                eval_epoch_rep_losses.append(rep_loss.cpu().detach().numpy())
                eval_epoch_losses.append(l.cpu().detach().numpy())
            eval_epoch_mse_loss = np.mean(eval_epoch_mse_losses)
            eval_epoch_rep_loss = np.mean(eval_epoch_rep_losses)
            eval_epoch_loss     = np.mean(eval_epoch_losses)
            bwd_eval_mse_losses.append(eval_epoch_mse_loss)
            bwd_eval_rep_losses.append(eval_epoch_rep_loss)

            if (epoch > 0) and (model_idx > 0):
                rep_loss_diff = bwd_eval_rep_losses[-2] - bwd_eval_rep_losses[-1]
                rep_loss_per_diff = abs(rep_loss_diff) / (bwd_eval_rep_losses[-2] + 1e-10)
                if rep_loss_per_diff < 0.001:
                    print("Turning repulsion loss weight off")
                    anneal_on = False
            msg = "Backwards {} train loss on epoch {}: {}"
            print(msg.format(model_idx + 1, epoch, bwd_train_mse_losses[-1]))
            msg = "Backwards {} eval loss on epoch {}: {}"
            print(msg.format(model_idx + 1, epoch, eval_epoch_mse_loss))

            if eval_epoch_loss < best_err:
            #if eval_epoch_mse_loss < best_err:
                best_err = eval_epoch_mse_loss
                torch.save(models_bwd[model_idx].state_dict(),
                           "{}/model_b{}.pt".format(out_dir, model_idx))

    print("done training backwards model {}".format(model_idx+1))
    return (bwd_train_mse_losses, bwd_eval_mse_losses), (bwd_train_rep_losses, bwd_eval_rep_losses)

def train_MMN_bwd_model_epoch(train_data, opt, model_fwd, models_bwd,
                              model_idx, epoch, train_params):
    models_bwd[model_idx].train()
    epoch_mse_losses, epoch_rep_losses, epoch_total_loss = [], [], []
    for i, (g, s) in enumerate(train_data):
        if cuda:
            g = g.cuda()
            s = s.cuda()
        opt.zero_grad()
        g_out = models_bwd[model_idx](s)
        s_out = model_fwd(g_out)
        prev = []
        """
        for p_idx in range(0, model_idx):
            prev.append(models_bwd[p_idx](s))
        """
        mse_loss = nn.functional.mse_loss(s, s_out)
        bdy_loss = losses.boundary_loss(g_out)
        rep_loss = torch.tensor([0.0], device = "cuda")
        #rep_loss = 0.0
        if len(prev) > 0:
            rep_loss = losses.compute_repulsion_loss_multiple_models(
                prev, g_out, train_params["sigma"])
        rep_loss_weight = get_repulsion_loss_weight(
            train_params["lam"], epoch, train_params["num_epochs"],
            anneal = "none")
        l = mse_loss + bdy_loss * train_params["bdy_loss_weight"] - rep_loss_weight * rep_loss
        l.backward()
        opt.step()
        epoch_mse_losses.append(mse_loss.cpu().detach().numpy())
        epoch_rep_losses.append(rep_loss.cpu().detach().numpy())
        epoch_total_loss.append(l.cpu().detach().numpy())
    return np.mean(epoch_mse_losses), np.mean(epoch_rep_losses), np.mean(epoch_total_loss)

def get_repulsion_loss_weight(lam, epoch, num_epochs, anneal = "none"):
    global anneal_on
    if anneal == "linear":
        anneal_factor = (num_epochs - epoch) / num_epochs
        weight = lam * anneal_factor
    elif anneal == "on_off":
        if anneal_on is True:
            weight = lam * 1.0
        elif anneal_on is False:
            weight = 0.0
    else:
        weight = lam
    return weight

