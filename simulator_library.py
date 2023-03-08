"""
A file to hold functions to call as "true_fcn" for inverse-problems
(e.g., when doing inference)
"""
import numpy as np
import torch
import pdb

from Data.robotic_arm_data_gen import determine_final_position
from Data.Inverse_ballistics_original import InverseBallisticsModel
import predict_meta_material_with_NA as meta_materials_sim
################################################################################
# TRUE FUNCTION (SIMULATOR) DEFINITIONS
################################################################################
def true_1d_sinusoid_fn(x):
    if type(x) == torch.Tensor:
        x = x.cpu().detach().numpy()    
    y = (np.sin(3 * np.pi * x) + 2 * x)
    #return np.expand_dims(y, axis=1)
    return y

def true_robotic_arm_fn(x):
    if type(x) == torch.Tensor:
        x = x.cpu().detach().numpy()    
    y, _ = determine_final_position(x[:, 0], x[:, 1:], evaluate_mode=True)
    return y

def true_ballistics_fn(x):
    if type(x) == torch.Tensor:
        x = x.cpu().detach().numpy()    
    x[:, 3] *= 15
    IB  = InverseBallisticsModel()
    y = IB.forward_process(x, output_full = True)
    # make shape [batch, 1]
    y = np.expand_dims(y, axis=1)
    return y

BDIMNNA_DIR = "/home/gps10/Documents/BDIMNNA/"
SIM_MODELS_DIR = (BDIMNNA_DIR + "Simulated_DataSets/" +
                  "Meta_material_Neural_Simulator/")
def get_prediction_across_ensemble(X):
    sim_models_dir = SIM_MODELS_DIR + "state_dicts/"
    sim_models_list = meta_materials_sim.get_neural_simulators_list(
        sim_models_dir)
    pred_list = []
    model_dir = SIM_MODELS_DIR + 'meta_material'
    if type(X) == np.ndarray:
        X = torch.from_numpy(X).to(torch.float)
    for sim_model in sim_models_list:
        y_pred = meta_materials_sim.predict_from_model(model_dir, X, sim_model)
        pred_list.append(np.copy(np.expand_dims(y_pred, axis=2)))
    # Take the mean of the predictions
    pred_all = np.concatenate(pred_list, axis=2)
    pred_mean = np.mean(pred_all, axis=2)
    return pred_mean
