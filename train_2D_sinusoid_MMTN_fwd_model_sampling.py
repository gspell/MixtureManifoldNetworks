"""
training file (originally copied from train_sinusoid_MMTN_bens_data.py)
"""

import numpy as np
import os
import pdb
import pandas as pd
import time

import torch
import torch.nn as nn
from torch.optim import lr_scheduler

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns

from tandem import make_forward_model, make_multiple_bwd_models
import tandem_model_training as training_fcns
import tandem_losses as losses
from tandem_MMTN_inference import MMTN_inference as inference
from simulator_library import true_2d_sinusoid_fn

import Data.data_reader as data_reader

import training_plotting as plotting
import visualize_sinusoid_manifolds as vis

cuda = True if torch.cuda.is_available() else False

################################################################################
# POINT SAMPLING/GENERATION
################################################################################
def sample_pts_for_fwd_model_generation(num_points, x_low=-1, x_high=1):
    x_dimension = 2
    num_sample_dimension = []
    num_sample_dimension.append(int(np.floor(np.sqrt(num_points))))
    num_sample_dimension.append(int(np.ceil(num_points / num_sample_dimension[0])))
    xx = []
    for i in range(x_dimension):
        xx.append(np.random.uniform(x_low, x_high, size=num_sample_dimension[i]))
    x = np.array(np.meshgrid(*xx)) # shape: [x_dim, #point, #point]
    # Reshape sampled data into one long list
    x_sampled = np.concatenate([np.reshape(np.ravel(x[i, :]), [-1, 1] ) for
                                i in range(x_dimension)], axis=1)
    #x_sampled = np.random.uniform(x_low, x_high, size = num_points)
    #x_sampled = np.expand_dims(x_sampled, axis=1)
    # put into a tensor and on cuda device
    x_sampled = torch.Tensor(x_sampled)
    if cuda:
        x_sampled = x_sampled.cuda()
    return x_sampled

def generate_pts_from_fwd_model(model_f, num_points, x_low=-1, x_high=1):
    x_sampled = sample_pts_for_fwd_model_generation(num_points, x_low, x_high)
    with torch.no_grad():
        y_gen = model_f(x_sampled)
    return (x_sampled, y_gen)

################################################################################
# 
################################################################################
def get_bwd_model_resim_error_on_test(test_data, model_b):
    inf_err = []
    for i, (g, s) in enumerate(test_data):
        if torch.cuda.is_available():
            g = g.cuda()
            s = s.cuda()
        g_out = model_b(s)
        g_out_np = g_out.cpu().detach().numpy()
        s_out = true_2d_sinusoid_fn(g_out_np)
        resim_err = np.abs(s_out - np.squeeze(s.cpu().detach().numpy())) ** 2
        inf_err.append(np.mean(resim_err))
    return np.mean(inf_err)

################################################################################
# BIG OL' TRAIN CHAIN
################################################################################
def train(data, model_f, models_b, train_params, num_models, out_dir,
          num_bwd_models_trained=0, train_fwd_model=True):
    global anneal_on
    train_plot_dir = os.path.join(out_dir, "training_plots")
    os.makedirs(train_plot_dir, exist_ok = True)
    train_data, test_data = data # unpack the data tuple
    
    #set up results file
    results = open("{}/results_during_training.txt".format(out_dir), "w")
    results.write("Number of epochs: {}\n".format(train_params["num_epochs"]))
    results.write("Sigma: {}\n".format(train_params["sigma"]))
    results.write("Sigma squared: {}\n".format(train_params["sigma"] ** 2))
    results.write("Lambda: {}\n".format(train_params["lam"]))

    # FORWARD MODEL
    if train_fwd_model is True:
        training_fcns.fwd_model_training(
            model_f, train_data, test_data, train_params,
            out_dir, results, train_plot_dir)
    model_f.eval()
    torch.save(model_f.state_dict(), "{}/modelf.pt".format(out_dir))
    
    # SAMPLE/GENERATE DATA TO USE TO TRAIN BACKWARD MODEL
    x_sampled, y_gen = generate_pts_from_fwd_model(model_f,
                                                   num_sample_points)
    gen_train_data, gen_eval_data = data_reader.get_data_into_loaders(
        x_sampled.cpu().detach().numpy(), y_gen.cpu().detach().numpy(),
        batch_size=1024,
        test_ratio=0.2,
        DataSetClass = data_reader.SimulatedDataSet_class)
        #DataSetClass = data_reader.SimulatedDataSet_regress)
    
    # BACKWARD MODELS
    inference_errs = []
    model_idx = -1
    for model_idx in range(num_bwd_models_trained):
        inf_err = get_bwd_model_resim_error_on_test(test_data,
                                                    models_b[model_idx])
        inference_errs.append(inf_err)
        bwd_filepath = "{}/model_b{}.pt".format(out_dir, model_idx)
        torch.save(models_b[model_idx].state_dict(), bwd_filepath)
        msg = "Inference error for backwards model {}: {}"
        print(msg.format(model_idx+1, inf_err))
        results.write((msg + "\n").format(model_idx+1, inf_err))
        
    #train backwards models
    for jj in range(num_models-num_bwd_models_trained):
        anneal_on = True
        model_idx += 1
        opt = training_fcns.make_optimizer_b(models_b[model_idx], train_params)
        lr_sched = training_fcns.make_lr_scheduler(opt, train_params)
        """
        bwd_mse_losses, bwd_rep_losses = training_fcns.train_MMTN_bwd_model(
            train_data, test_data, opt, model_f, models_b, model_idx,
            train_params, out_dir)
        """
        bwd_mse_losses, bwd_rep_losses = training_fcns.train_MMTN_bwd_model(
            gen_train_data, gen_eval_data, opt, lr_sched, model_f, models_b,
            model_idx, train_params, out_dir)
        """
        inf_err = get_bwd_model_resim_error_on_test(test_data,
                                                    models_b[model_idx])
        """
        inf_err = get_bwd_model_resim_error_on_test(gen_eval_data,
                                                    models_b[model_idx])        
        inference_errs.append(inf_err)
        
        msg = "Inference error for backwards model {}: {}"
        print(msg.format(model_idx+1, inf_err))
        results.write((msg + "\n").format(model_idx+1, inf_err))

        msg = "Final Eval MSE Loss for backwards model {}: {}"
        print(msg.format(model_idx+1, bwd_mse_losses[1][-1]))
        results.write((msg + "\n").format(model_idx+1, bwd_mse_losses[1][-1]))

        msg = "Final Eval Rep Loss for backwards model {}: {}"
        print(msg.format(model_idx+1, bwd_rep_losses[1][-1]))
        results.write((msg + "\n").format(model_idx+1, bwd_rep_losses[1][-1]))
        # PLOT TRAINING CURVES        
        plotting.plot_bwd_model_mse_losses(bwd_mse_losses[0], bwd_mse_losses[1],
                                           model_idx+1, train_plot_dir)
        plotting.plot_bwd_model_rep_losses(bwd_rep_losses[0], bwd_rep_losses[1],
                                           model_idx+1, train_plot_dir)
    results.close()

    ############################################################################
    # INFERENCE
    ############################################################################
    
    results = inference(test_data, model_f, models_b, out_dir,
                        "results_at_end_of_training.txt", true_2d_sinusoid_fn)
    vis.plot_sinusoid_geometry_visualization(models_b, out_dir,
                                             num_points = 500)    
    
############################################################################
# MAIN
############################################################################    
def train_over_multiple_manifolds():
    for num_manifolds in nums_manifolds:
        print("\n")
        num_bwd_models_trained = num_manifolds - 1
        model_dir = os.path.join(out_dir,
                                 "num_manifolds_{}/".format(num_manifolds))
        model_dir = os.path.join(model_dir, "lam_{}".format(lam))
        model_dir = os.path.join(model_dir, "run{}".format(i))
        os.makedirs(model_dir, exist_ok=True)

        fwd_model_path = os.path.join(fwd_model_dir, f"run{i}", "modelf.pt")
        if num_manifolds > 1:
            temp_dir = "num_manifolds_{}/".format(num_bwd_models_trained)
            run_load_dir = os.path.join(model_load_dir, temp_dir)
            temp_dir = "lam_{}/run{}".format(lam, i)
            run_load_dir = os.path.join(run_load_dir, temp_dir)

            #fwd_model_path = os.path.join(run_load_dir, "modelf.pt")
            bwd_model_paths = [None for manifold in range(num_manifolds)]
            for ii in range(num_manifolds - 1):
                model_file = "model_b{}.pt".format(ii)
                bwd_model_path = os.path.join(run_load_dir, model_file)
                bwd_model_paths[ii] = bwd_model_path
            train_fwd_model = False
        else:
            #fwd_model_path = None
            bwd_model_paths = None
            #train_fwd_model = True
            train_fwd_model = False
        model_f = make_forward_model(task, model_path=fwd_model_path)
        models_b = make_multiple_bwd_models(num_manifolds, task,
                                            bwd_model_paths)
        train(data, model_f, models_b, train_params, num_manifolds,
              model_dir, num_bwd_models_trained=num_bwd_models_trained,
              train_fwd_model=train_fwd_model)
        
if __name__ == '__main__':
    task = "sine"
    num_real_points = 8000
    root_dir = "2D_Sinusoid_Models/sample_fwd_model/"
    """
    root_dir = os.path.join(
        root_dir, "single_fwd_model_trained_{}_pts".format(num_real_points))
    """
    root_dir = os.path.join(
        root_dir, "multiple_fwd_models_trained_{}_pts".format(num_real_points))
    os.makedirs(root_dir, exist_ok = True)

    fwd_model_dir = "2D_Sinusoid_Models/bdy_loss_weight_100/"
    fwd_model_dir = os.path.join(fwd_model_dir, "num_manifolds_1/lam_0.0/")
    #fwd_model_dir = os.path.join(fwd_model_dir, "run0")
    #fwd_model_path = os.path.join(fwd_model_dir, "modelf.pt")

    # Choosing some squares for a while below here
    #nums_sample_points = [36, 100, 256, 1024, 2500, 3600, 6400, 8100, 10000]
    #nums_sample_points = [14400, 22500, 40000, 62500, 90000]

    # NEW NUMS POINTS TO SAMPLE -- ALSO CHANGED TEST RATIO (AND BDY LOSS)
    #nums_sample_points = [10000, 14400]
    #nums_sample_points = [6400, 8100, 25600, 40000, 62500, 90000, 122500]
    #nums_sample_points = [2500, 3600, 4900]
    nums_sample_points = [900, 1600]
    lam = 0.0
    sigma = 1.0
    # HAVE CHECKED THESE SETTINGS IN BEN'S CODE AND LOADED FLAGS
    bdy_loss_weight = 100    
    batch_size = 1024
    eval_batch_size = 1024
    lr_decay_rate = 0.5
    learning_rate = 0.001
    optim = 'Adam'
    reg_scale = 0.0005
    num_epochs = 500
    eval_step = 20
    # ROLL THE SETTINGS INTO RELEVANT DICTS TO MAKE EASY TO PASS AROUND
    train_params = {
        "learning_rate": learning_rate,
        "optim": optim,
        "lr_decay_rate": lr_decay_rate,
        "reg_scale": reg_scale,
        "num_epochs": num_epochs,
        "eval_step": eval_step,
        "lam": lam,
        "sigma": sigma,
        "bdy_loss_weight": bdy_loss_weight,
    }

    start_time = time.time()
    #data = data_reader.get_bens_sine_wave_data(subsample=num_real_points)
    data = data_reader.get_bens_sine_wave_data()

    nums_manifolds = [1, 2, 3, 4, 5, 6]
    #for i in range(0, 12, 3):
    #for i in range(1, 12, 3):
    for i in range(2, 12, 3):
        for num_sample_points in nums_sample_points:
            out_dir = os.path.join(root_dir,
                                   "num_points_{}".format(num_sample_points))
            os.makedirs(out_dir, exist_ok = True)
            model_load_dir = out_dir        
            train_over_multiple_manifolds()
    end_time = time.time()
    print("\nTook {} seconds to train MMTNs".format(end_time - start_time))

