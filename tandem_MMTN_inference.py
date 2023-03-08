import numpy as np
import os
import pdb
import time

import torch
from torch.nn.functional import mse_loss

import tandem_losses as losses

cuda = True if torch.cuda.is_available() else False

def true_sinusoid_fn(x):
    if type(x) == torch.Tensor:
        x = x.cpu().detach().numpy()
    y = (np.sin(3 * np.pi * x[:, 0]) + np.cos(3 * np.pi * x[:, 1]))
    # make shape [batch, 1]
    return np.expand_dims(y, axis=1)

################################################################################
# END OF TRAINING INFERENCE AND RESULTS WRITING
################################################################################
def get_fwd_model_err(x, y, model_f):
    # fwd-model maps x to f_hat_x (e.g., y_hat).
    # Assess fwd-model by how far off f_hat_x is from true y
    f_hat_x = model_f(x)
    fwd_model_err = mse_loss(f_hat_x, y, reduction="none").mean(axis=1)
    return fwd_model_err

def single_manifold_inference(data, model_f, model_b, true_fn):
    x_pts, y_pts, x_preds, y_hats = [], [], [], []
    model_f.eval()
    model_b.eval()
    start_time = time.time()
    for i, (x, y) in enumerate(data):
        if torch.cuda.is_available:
            x = x.cuda()
            y = y.cuda()
        x_pts.append(x.cpu().detach().numpy())
        y_pts.append(y.cpu().detach().numpy())
        x_hat = model_b(y)
        f_hat_x_hat = model_f(x_hat)
        x_preds.append(x_hat.cpu().detach().numpy())
        y_hats.append(f_hat_x_hat.cpu().detach().numpy())
    end_time = time.time()
    inference_time = end_time - start_time
    x_preds = np.concatenate(x_preds)
    y_hats = np.concatenate(y_hats)
    x_pts = np.concatenate(x_pts)
    y_pts = np.concatenate(y_pts)
    if true_fn is not None:
        y_preds = true_fn(x_preds)
    else:
        y_preds = np.zeros(y_pts.shape)
    print("Took {} seconds to do inference over {} points".format(inference_time, len(x_pts)))
    return x_pts, y_pts, x_preds, y_preds, y_hats

def get_model_inference_errs(batch_data, model_f, model_b, true_fn):
    """ 
    For a batch of data and for a fwd-bwd model pair: 
        -get the forward model errs, and the resim errs
    """
    # x shape: [batch, input_dim]:
    #     -input_dim is e.g., 1 for 1D Sinusoid, 2 for 2D Sinusoid,
    #                         4 (?) for Robotic Arm
    # y shape: [batch, output_dim]:
    #     -output_dim is e.g., 1 for 1D Sinusoid, 1 for 2D Sinusoid,
    #                          ? for Robotic Arm
    
    x, y = batch_data

    y_hat = model_f(x) # AKA f_hat_x
    fwd_model_err = mse_loss(y_hat, y, reduction = "none").mean(axis=1)

    x_hat_1 = model_b(y_hat)
    y_hat_hat = model_f(x_hat_1)
    surrogate_err = mse_loss(y_hat_hat, y_hat, reduction = "none").mean(axis=1)
    
    x_hat = model_b(y) # shape: [batch, input_dim]
    f_hat_x_hat = model_f(x_hat) # shape: [batch, output_dim]
    est_resim_err = mse_loss(f_hat_x_hat, y, reduction = "none").mean(axis=1)
    #x_hat_np = x_hat.cpu().detach().numpy()
    y_resim = true_fn(x_hat) # now check for type in simulator -- convert numpy

    # This is for a weird case with ballistics:
    #     - apparently it sometimes predicts "unreal" points
    # This is how it is handled in Ben's "evaluation_helper.py" code
    if y.shape[1] == 1:
        valid_index = (y_resim != -999).squeeze()
        if np.sum(valid_index) != len(valid_index):
            est_resim_err[~valid_index] = np.nan
            y_resim[~valid_index] = np.nan
            #fwd_model_err = fwd_model_err[valid_index]
            #y_resim = y_resim[valid_index]
            #y = y[valid_index]
    #resim_err = np.mean(np.square(y_resim - y.cpu().detach().numpy()), axis=1)
    resim_err = np.square(y_resim - y.cpu().detach().numpy())
    resim_err = np.nanmean(resim_err, axis=1)
    #losses.boundary_loss(x_hat)
    errors = {
        "resim_est": est_resim_err.cpu().detach().numpy(),
        "resim_err": resim_err,
        "fwd_model_err": fwd_model_err.cpu().detach().numpy(),
        "surrogate_err": surrogate_err.cpu().detach().numpy()
    }
    return errors

        
def MMTN_inference(data, model_f, models_b, out_dir,
                   filename = "results_at_inference.txt",
                   true_fn = true_sinusoid_fn):
    
    num_bwd_models = len(models_b)
    resim_est_individual, resim_errs_individual = [], []
    fwd_errs_individual, surrogate_errs_individual = [], []
    overall_inf_err = []
    y_pts = []
    start_time = time.time()
    for i, (x, y) in enumerate(data):
        if torch.cuda.is_available:
            x = x.cuda()
            y = y.cuda()
        y_pts.append(y.cpu().detach().numpy())
        est_resims, resim_errs, fwd_errs, surrogate_errs = [], [], [], []
        for n in range(num_bwd_models):
            errors = get_model_inference_errs(
                (x, y), model_f, models_b[n], true_fn)
                
            est_resims.append(errors["resim_est"].tolist())
            resim_errs.append(errors["resim_err"].tolist())
            fwd_errs.append(errors["fwd_model_err"].tolist())
            surrogate_errs.append(errors["surrogate_err"].tolist())
        est_resims = np.stack(est_resims).squeeze().transpose()
        resim_errs = np.stack(resim_errs).squeeze().transpose()
        fwd_errs = np.stack(fwd_errs).squeeze().transpose()
        surrogate_errs = np.stack(surrogate_errs).squeeze().transpose()
        if len(est_resims.shape) == 1:
            est_resims = np.expand_dims(est_resims, axis=1)
            resim_errs = np.expand_dims(resim_errs, axis=1)
            fwd_errs = np.expand_dims(fwd_errs, axis=1)
            surrogate_errs = np.expand_dims(surrogate_errs, axis=1)
        resim_est_individual.append(est_resims)
        resim_errs_individual.append(resim_errs)
        fwd_errs_individual.append(fwd_errs)
        surrogate_errs_individual.append(surrogate_errs)
    end_time = time.time()
    inf_time = end_time - start_time
    y_pts = np.concatenate(y_pts).squeeze()
    resim_est_individual = np.concatenate(resim_est_individual, axis=0)
    resim_errs_individual = np.concatenate(resim_errs_individual, axis=0)
    fwd_errs_individual = np.concatenate(fwd_errs_individual, axis=0)
    surrogate_errs_individual = np.concatenate(surrogate_errs_individual, axis=0)
    print("Took {} seconds to do inference over {} points".format(inf_time, len(y_pts)))    
    """
    if np.isnan(resim_errs_individual).sum() != 0:
        pdb.set_trace()
    """
    """
    x_hat_idxs_1 = np.argmin(resim_est_individual[:, 0:1], axis=1)
    x_hat_idxs_2 = np.argmin(resim_est_individual[:, 0:2], axis=1)
    x_hat_idxs_3 = np.argmin(resim_est_individual[:, 0:3], axis=1)
    x_hat_idxs_oracle = np.argmin(resim_errs_individual, axis=1)
    """
    # LOL, what the hell, numpy will take nan as min/argmin is there (ballistics)
    
    #x_hat_idxs = np.argmin(resim_est_individual, axis=1)
    # It's possible that there are NaN's here, so we use nanargmin
    # BUT if there is only one manifold and there is a NaN, will have a problem
    try:
        x_hat_idxs = np.nanargmin(resim_est_individual, axis=1)
    except:
        pdb.set_trace()

    surr_err_idxs = np.nanargmin(surrogate_errs_individual, axis=1)
    resim_err_overall, surrogate_err_overall = [], []
    for i, x_hat_idx in enumerate(x_hat_idxs):
        resim_err_overall.append(resim_errs_individual[i][x_hat_idx])
        surrogate_err_overall.append(surrogate_errs_individual[i][surr_err_idxs[i]])
    assert np.isnan(resim_err_overall).sum() == 0
    final_err = np.nanmean(resim_err_overall)
    final_surrogate = np.nanmean(surrogate_err_overall)

    write_results_to_file(final_err, final_surrogate, resim_est_individual,
                          resim_errs_individual, fwd_errs_individual,
                          surrogate_errs_individual, out_dir, filename)
    
    return final_err, resim_est_individual, resim_errs_individual, fwd_errs_individual, surrogate_errs_individual, y_pts

def write_results_to_file(final_err, final_surrogate, resim_est_individual,
                          resim_errs_individual, fwd_errs_individual,
                          surrogate_errs_individual, out_dir, filename):
    #set up results file
    filename = os.path.join(out_dir, filename)
    results = open(filename, "w")
    #results.write("Sigma: {}\n".format(sigma))
    #results.write("Sigma squared: {}\n".format(sigma ** 2))
    #results.write("Lambda: {}\n".format(lam))
    
    msg = "Overall error: {}"
    print(msg.format(final_err))
    results.write((msg + "\n").format(final_err))
    
    #fwd_errs_individual = np.stack(fwd_errs_individual).mean(axis=0)    
    msg = "Resimulation Estimate Errors: {}"
    print(msg.format(resim_est_individual.mean(axis=0)))
    results.write((msg + "\n").format(resim_est_individual.mean(axis=0)))
    
    #resim_errs_individual = np.stack(resim_errs_individual).mean(axis=0)
    msg = "Resimulation Individual Errors: {}"
    print(msg.format(resim_errs_individual.mean(axis=0)))
    results.write((msg + "\n").format(resim_errs_individual.mean(axis=0)))
    
    msg = "Forward Model Individual Errors: {}"
    print(msg.format(fwd_errs_individual.mean(axis=0)))
    results.write((msg + "\n").format(fwd_errs_individual.mean(axis=0)))

    msg = "Surrogate Individual Errors: {}"
    print(msg.format(surrogate_errs_individual.mean(axis=0)))
    results.write((msg + "\n").format(surrogate_errs_individual.mean(axis=0)))

    msg = "Surrogate error: {}"
    print(msg.format(final_surrogate))
    results.write((msg + "\n").format(final_surrogate))
    
    results.close()    
