"""
Just do inference over saved models -- Two inverse models
"""

import numpy as np
import os
import pdb
import pandas as pd

import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits import mplot3d
import seaborn as sns
import pdb

import Data.data_reader as data_reader

cuda = True if torch.cuda.is_available() else False

################################################################################
# 
################################################################################
def plot_true_3D_sinusoid_contour(out_dir = "./"):
    ax = make_true_3d_sinusoid_contour_axes(num_points = 100)
    save_path = os.path.join(out_dir, "sinusoid_3D_contour.png")
    plt.savefig(save_path, bbox_inches = "tight")
    plt.close()
    
def make_true_3d_sinusoid_contour_axes(num_points = 100):
    fig = plt.figure(figsize = (12, 12))
    ax = plt.axes(projection = '3d')
    x0, x1 = np.linspace(-1, 1, num_points), np.linspace(-1, 1, num_points)
    X0, X1 = np.meshgrid(x0, x1)
    Y = true_sinusoid_fn_mesh(X0, X1)
    ax.contour3D(X0, X1, Y, 60, cmap = "binary")
    return ax

################################################################################
# 
################################################################################
def true_sinusoid_fn_mesh(x0, x1):
    y = (np.sin(3 * np.pi * x0) + np.cos(3 * np.pi * x1))
    return y

def true_sinusoid_fn(x):
    y = (np.sin(3 * np.pi * x[:, 0]) + np.cos(3 * np.pi * x[:, 1]))
    return y

################################################################################
# 
################################################################################
def plot_3d_sinusoid_predictions_with_true_contour(bwd_models, out_dir = "./", num_points = 300):
    ax = make_true_3d_sinusoid_contour_axes(200)
    test_y = np.linspace(-1, 1, num_points)
    test_y = torch.tensor(test_y, dtype = torch.float).unsqueeze(1)
    if torch.cuda.is_available():
        test_y = test_y.cuda()
    for i, bwd_model in enumerate(bwd_models):
        x_hat = bwd_model(test_y)
        x_hat = x_hat.cpu().detach().numpy()
        label = "Inverse Model {}"
        #ax.plot3D(x_hat[:, 0], x_hat[:, 1], test_y.cpu().detach().numpy(), label = label.format(i))
        ax.scatter3D(x_hat[:, 0], x_hat[:, 1], test_y.cpu().detach().numpy(), label = label.format(i))
    plt.title("Sinusoid Inverse Model Geometry Visualization", fontweight = "bold")        
    plt.legend()
    
    el_angles = [15, 30, 45, 60, 75, 90]
    az_angles = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]
    for el_angle in el_angles:
        for az_angle in az_angles:
            ax.view_init(el_angle, az_angle)
            save_path = os.path.join(out_dir, "geometry_visualization_3d_with_contour_{}_{}.png".format(el_angle, az_angle))
            plt.savefig(save_path, bbox_inches = "tight")
    plt.close()
    
def plot_3D_sinusoid_visualization(bwd_models, out_dir, num_points = 500):
    fig = plt.figure()
    ax = plt.axes(projection = '3d')
    plt.title("Sinusoid Inverse Model Geometry Visualization", fontweight = "bold")
    test_y = np.linspace(-1, 1, num_points)
    test_y = torch.tensor(test_y, dtype = torch.float).unsqueeze(1)
    if torch.cuda.is_available():
        test_y = test_y.cuda()
    for i, bwd_model in enumerate(bwd_models):
        x_hat = bwd_model(test_y)
        x_hat = x_hat.cpu().detach().numpy()
        label = "Inverse Model {}"
        #ax.plot3D(x_hat[:, 0], x_hat[:, 1], test_y.cpu().detach().numpy(), label = label.format(i))
        ax.scatter3D(x_hat[:, 0], x_hat[:, 1], test_y.cpu().detach().numpy(), label = label.format(i))
    plt.legend()
    save_path = os.path.join(out_dir, "geometry_visualization_3d.png")
    plt.savefig(save_path, bbox_inches = "tight")
    plt.close()
    
################################################################################
# 
################################################################################
def plot_train_and_eval_data(out_dir = "./"):
    train_data, eval_data = data_reader.get_sine_wave_train_eval_data_arrays()
    # Train data first
    x_train, y_train = train_data
    plt.scatter(x_train[:, 0], x_train[:, 1])
    save_path = os.path.join(out_dir, "train_data_scatter_x.png")
    plt.savefig(save_path, bbox_inches = "tight")
    
################################################################################
# 
################################################################################
def plot_1D_sinusoid_fwd_model_visualization(
        fwd_model, out_dir, num_points = 500):
    plt.figure()
    title = "1D-Sinusoid Forward Model Visualization"
    plt.title(title, fontweight="bold")
    
    # ALSO PUT THE TRUE FCN ON
    x_pts = np.linspace(-1, 1, 512)
    y_pts = np.sin(3 * np.pi * x_pts) + 2 * x_pts
    plt.plot(x_pts, y_pts, 'r', label = "True Fcn")

    # MAP x_pts TO y_pts_est USING FORWARD MODEL
    test_x = torch.tensor(x_pts, dtype=torch.float).unsqueeze(1) # make [batch, 1] shape rather than [batch]
    if torch.cuda.is_available():
        test_x = test_x.cuda()
    y_hat = fwd_model(test_x)
    y_hat = y_hat.cpu().detach().numpy()
    plt.scatter(x_pts, y_hat, s = 10, label = "Fwd. Model Est.")
    plt.legend()
    save_path = os.path.join(out_dir, "fwd_model_visualization.png")
    plt.savefig(save_path, bbox_inches = "tight")
    plt.close()
    
def plot_1D_sinusoid_geometry_visualization(
        bwd_models, out_dir, num_points = 500):
    plt.figure()
    title = "Sinusoid Inverse Model Geometry Visualization"
    plt.title(title, fontweight="bold")
    
    # ALSO PUT THE TRUE FCN ON
    x_pts = np.linspace(-1, 1, 512)
    y_pts = np.sin(3 * np.pi * x_pts) + 2 * x_pts
    plt.plot(x_pts, y_pts, 'r', label = "True Fcn")
    
    y_min, y_max = -2.69, 2.69
    test_y = np.linspace(y_min, y_max, num_points)
    test_y = torch.tensor(test_y, dtype = torch.float).unsqueeze(1)
    if torch.cuda.is_available():
        test_y = test_y.cuda()
    for i, bwd_model in enumerate(bwd_models):
        x_hat = bwd_model(test_y)
        x_hat = x_hat.cpu().detach().numpy()
        label = "Inverse Model {}"
        plt.scatter(x_hat, test_y.cpu().detach().numpy(),
                    s = 10, label = label.format(i))
    plt.legend()
    save_path = os.path.join(out_dir, "geometry_visualization.png")
    plt.savefig(save_path, bbox_inches = "tight")
    plt.close()

def plot_sinusoid_geometry_visualization(bwd_models, out_dir, num_points = 500):
    plt.figure()
    plt.title("Sinusoid Inverse Model Geometry Visualization", fontweight = "bold")
    test_y = np.linspace(-1, 1, num_points)
    test_y = torch.tensor(test_y, dtype = torch.float).unsqueeze(1)
    if torch.cuda.is_available():
        test_y = test_y.cuda()
    for i, bwd_model in enumerate(bwd_models):
        x_hat = bwd_model(test_y)
        x_hat = x_hat.cpu().detach().numpy()
        label = "Inverse Model {}"
        plt.scatter(x_hat[:, 0], x_hat[:, 1], s = 10, label = label.format(i))
    plt.legend()
    save_path = os.path.join(out_dir, "geometry_visualization.png")
    plt.savefig(save_path, bbox_inches = "tight")
    plt.close()
    
def inference_with_visualization(model_f, model_b, out_dir):
    train_data, test_data = data_reader.read_data_sine_wave()
    model_f.eval()
    for model in model_b:
        model.eval()
    print("Starting inference")

    plt.figure(1)
    plt.clf()
    plt.title("Geometry visualization with k=4 and repulsion term")
    #plt.title("Geometry visualization with k=1")
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.figure(2)
    plt.clf()
    #plt.xlim([-1, 1])
    #plt.ylim([-1, 1])
    plt.title("Visualization of all inverse models with spectra predictions")

    plt.figure(3)
    plt.clf()
    plt.title("Visualization of all inverse model errors with simulator error")
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    mesh = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
    mesh_ys = np.sin(3 * np.pi * mesh[0]) + np.cos(3 * np.pi * mesh[1])
    plt.pcolormesh(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100), mesh_ys, shading='auto',cmap=plt.get_cmap('gray'))
    markers = ['o', 'x', 'd', '*']
    test_s = np.linspace(-1, 1, 100)
    s_in = torch.tensor(test_s, dtype=torch.float).unsqueeze(1)


    if torch.cuda.is_available():
        s_in = s_in.cuda()
    for n in range(len(model_b)):
        g = model_b[n](s_in)
        test_g = g.cpu().detach().numpy()
        true_s = np.sin(3 * np.pi * test_g[:, 0]) + np.cos(3 * np.pi * test_g[:, 1])
        err = (true_s - test_s) ** 2
        plt.figure(1)
        plt.scatter(test_g[:, 0], test_g[:, 1], s=10, label='Inverse Model {}'.format(n + 1))

        plt.figure(2)
        plt.scatter(test_g[:, 0], test_g[:, 1], s=20, c=test_s, marker = markers[n], label='Inverse Model {}'.format(n+1))

        plt.figure(3)
        plt.scatter(test_g[:, 0], test_g[:, 1], s=20, c=np.sqrt(err), marker=markers[n],
                    label='Inverse Model {}'.format(n + 1))

        plt.figure(100 + n)
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])
        plt.title("Visualization of model {} error with simulator error".format(n+1))
        plt.scatter(test_g[:, 0], test_g[:, 1], s=20, c=np.sqrt(err), label='Inverse Model {}'.format(n + 1))
        plt.colorbar(label='sqrt error')
        plt.savefig("{}/err_viz_model{}.png".format(out_dir, n+1))

        plt.figure(200 + n)
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])
        plt.title("Visualization of model {} with simulator error and predictions".format(n + 1))
        plt.pcolormesh(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100), mesh_ys, shading='auto',
                       cmap=plt.get_cmap('gray'))
        plt.scatter(test_g[:, 0], test_g[:, 1], s=20, c=np.sqrt(err))
        plt.colorbar(label='sqrt error')
        plt.clim(0, 0.9)
        plt.savefig("{}/err_viz_preds_model{}.png".format(out_dir, n + 1))

    plt.figure(1)
    plt.legend()
    plt.savefig("{}/geometry_visualization.png".format(out_dir))

    plt.figure(2)
    plt.legend()
    plt.colorbar(label='corresponding spectra value')
    plt.savefig("{}/geometry_visualization_predictions.png".format(out_dir))

    plt.figure(3)
    plt.legend()
    plt.colorbar(label='sqrt error')
    plt.clim(0, 0.9)
    plt.savefig("{}/geometry_visualization_errors.png".format(out_dir))

################################################################################
# VISUALIZE MANIFOLDS
################################################################################
def plot_inverse_model_outputs(test_g, test_g2, out_dir):
    num_points = len(test_g)
    plt.figure()
    title = "Visualization of Inverse Model Output Geometries ({} Points)"
    plt.title(title.format(num_points), fontweight = "bold")
    plt.scatter(test_g[:, 0], test_g[:, 1], s = 10, label = "Inverse Model 1")
    plt.scatter(test_g2[:, 0], test_g2[:, 1], s = 10, label = "Inverse Model 2")
    plt.legend()
    filename = "bwd_model_outputs_geometry_vis_{}_pts.png"
    plt.savefig(os.path.join(out_dir, filename.format(num_points)))
    plt.close()

def plot_inverse_model_outputs_with_spectral_colorbar(test_g, test_g2, test_s,
                                                      out_dir):
    num_points = len(test_g)
    plt.figure()
    title = "Visualization of Inverse Model Output Geometries with Predictions"
    plt.scatter(test_g[:, 0], test_g[:, 1], s=20, c=test_s, marker = 'o',
                label='Inverse Model 1')
    plt.scatter(test_g2[:, 0], test_g2[:, 1], s=20, c=test_s, marker = 'x',
                label='Inverse Model 2')
    plt.colorbar(label='Spectra Predictions', orientation='horizontal')
    #plt.clim(-20, 0)    
    plt.legend()
    plt.savefig("{}/geometry_visualization_ys_{}.png".format(out_dir, num_points))

def plot_inverse_model_outputs_with_simulator_error(
        mesh_out, test_g, test_g2, test_err, model_num, out_dir):
    num_points = len(test_g)
    plt.figure()
    ax=plt.gca()
    title = "Visualization of Inverse Model {} Output Geometries"
    title = title + " with Simulator Error".format(model_num)
    plt.title(title, fontweight = "bold")
    plt.imshow(mesh_out, cmap = plt.cm.get_cmap('gray'), aspect = 'auto',
               extent = (min(min(test_g[:, 0]), min(test_g2[:, 0])),
                         max(max(test_g[:, 0]), max(test_g2[:, 0])),
                         min(min(test_g[:, 1]), min(test_g2[:, 1])),
                         max(max(test_g[:, 1]), max(test_g2[:, 1]))))
    #colorbar = np.sqrt(test_err) # could also do np.log(test_err)
    colorbar = np.log(test_err)
    plt.scatter(test_g[:, 0], test_g[:, 1], s = 20, c = colorbar,
                cmap = 'plasma')
    plt.plot(test_g[:, 0], test_g[:, 1])
    #plt.colorbar(label = 'Sqrt Error by Simulator', orientation = "horizonal")
    plt.colorbar(label = 'Log MSE by Simulator', orientation = "horizontal")
    #plt.clim(-20, 0)
    #plt.clim(0, 1.4)
    plt.savefig("{}/geometry_visualalization_model{}_mse_{}".format(
        out_dir, model_num, num_points))
    plt.close()

def plot_inverse_model_errors():
    plt.figure()
    title = "Visualization of both inverse model errors"
    pass

################################################################################
# TAKEN FROM "eval_MMTN_no/with_repulsion_loss.py" -- Not processed
################################################################################
def geometry_visualization():
    #geometry visualization
    for num in [100, 500]:
        test_s = np.linspace(-1, 1, num)
        s_in = torch.tensor(test_s, dtype=torch.float).unsqueeze(1)
        if torch.cuda.is_available():
            s_in = s_in.cuda()
        g = model_b(s_in)
        test_g = g.cpu().detach().numpy()

        g2 = model_b2(s_in)
        test_g2 = g2.cpu().detach().numpy()

        true_s = np.sin(3 * np.pi * test_g[:, 0]) + np.cos(3 * np.pi * test_g[:, 1])
        true_s2 = np.sin(3 * np.pi * test_g2[:, 0]) + np.cos(3 * np.pi * test_g2[:, 1])

        fwd_s = model_f(g).squeeze(1).cpu().detach().numpy()
        fwd_s2 = model_f(g2).squeeze(1).cpu().detach().numpy()

        test_err = (true_s - test_s)**2
        test_err2 = (true_s2 - test_s) ** 2
        test_err_fwd = (fwd_s - test_s)**2
        test_err_fwd2 = (fwd_s2 - test_s)**2

        print("x1_1: min: {}, max: {}".format(min(test_g[:,0]), max(test_g[:,0])))
        x1_1 = np.linspace(min(test_g[:, 0]), max(test_g[:, 0]), 100)
        x2_1 = np.linspace(min(test_g[:, 1]), max(test_g[:, 1]), 100)

        x1_2 = np.linspace(min(test_g2[:, 0]), max(test_g2[:, 0]), 100)
        x2_2 = np.linspace(min(test_g2[:, 1]), max(test_g2[:, 1]), 100)

        mesh1 = np.meshgrid(x1_1, x2_1)
        mesh2 = np.meshgrid(x1_2, x2_2)
        print(mesh1)
        mesh_out1 = np.sin(3 * np.pi * mesh1[0]) + np.cos(3 * np.pi * mesh1[1])
        mesh_out2 = np.sin(3 * np.pi * mesh2[0]) + np.cos(3 * np.pi * mesh2[1])

        plot_inverse_model_outputs(test_g, test_g2, out_dir)
        plot_inverse_model_outputs_with_spectral_colorbar(
            test_g, test_g2, test_s, out_dir)
        plot_inverse_model_outputs_with_simulator_error(
            mesh_out1, test_g, test_g2, test_err, 1, out_dir)
        plot_inverse_model_outputs_with_simulator_error(
            mesh_out2, test_g2, test_g, test_err2, 2, out_dir)
        
        plt.figure(12)
        plt.clf()
        plt.title("Visualization of both inverse model errors")
        plt.scatter(test_g[:, 0], test_g[:, 1], s=20, c=np.log(test_err),
                    marker = 'o', cmap='plasma')
        plt.scatter(test_g2[:, 0], test_g2[:, 1], s=20, c=np.log(test_err2),
                    marker='x',cmap='plasma')
        plt.plot(test_g[:, 0], test_g[:, 1], label='Inverse Model 1')
        plt.plot(test_g2[:, 0], test_g2[:, 1], label='Inverse Model 2')
        plt.colorbar(label='Error', orientation='horizontal')
        plt.legend()
        plt.savefig("{}/geometry_visualization_mse_{}".format(out_dir, num))
        print("Saved to {}/geometry_visualization_mse_{}".format(out_dir, num))

        plt.figure(13)
        plt.clf()
        plt.title("Visualization of inverse model 1 output geometries")
        plt.imshow(mesh_out1, cmap=plt.cm.get_cmap('gray'),
                   extent=(min(min(test_g[:, 0]), min(test_g2[:,0])),
                           max(max(test_g[:, 0]), max(test_g2[:,0])),
                           min(min(test_g[:, 1]), min(test_g2[:,1])),
                           max(max(test_g[:, 1]), max(test_g2[:,1]))))
        plt.scatter(test_g[:, 0], test_g[:, 1], s=20, c=np.log(test_err_fwd), cmap='plasma')
        plt.plot(test_g[:, 0], test_g[:, 1])
        plt.colorbar(label='log error by Forward model', orientation='horizontal')
        plt.savefig("{}/geometry_visualization_model1_forward_mse_{}".format(out_dir, num))

        plt.figure(14)
        plt.clf()
        plt.title("Visualization of inverse model 2 output geometries")
        plt.imshow(mesh_out2, cmap=plt.cm.get_cmap('gray'),
                   extent=(min(min(test_g[:, 0]), min(test_g2[:,0])), max(max(test_g[:, 0]), max(test_g2[:,0])), min(min(test_g[:, 1]), min(test_g2[:,1])), max(max(test_g[:, 1]), max(test_g2[:,1]))))
        plt.scatter(test_g2[:, 0], test_g2[:, 1], s=20, c=np.log(test_err_fwd2), cmap='plasma')
        plt.plot(test_g2[:, 0], test_g2[:, 1], label='Inverse Model 2')
        plt.colorbar(label='log error by Forward model', orientation='horizontal')
        plt.savefig("{}/geometry_visualization_model2_forward_mse_{}".format(out_dir, num))

        plt.figure(15)
        plt.clf()
        plt.title("Visualization of both inverse model errors")
        plt.scatter(test_g[:, 0], test_g[:, 1], s=20, c=np.log(test_err_fwd), marker = 'o', cmap='plasma')
        plt.scatter(test_g2[:, 0], test_g2[:, 1], s=20, c=np.log(test_err_fwd2), marker='x', cmap='plasma')
        plt.plot(test_g[:, 0], test_g[:, 1], label='Inverse Model 1')
        plt.plot(test_g2[:, 0], test_g2[:, 1], label='Inverse Model 2')
        plt.colorbar(label='Error by Forward model', orientation='horizontal')
        plt.legend()
        plt.savefig("{}/geometry_visualization_forward_mse_{}".format(out_dir, num))
    
if __name__ == "__main__":
    plot_train_and_eval_data()
