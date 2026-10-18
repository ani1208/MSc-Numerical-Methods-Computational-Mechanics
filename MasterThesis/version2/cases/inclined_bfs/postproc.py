######################################################################
import os
import json
import glob
from pathlib import Path
import deepxde as dde
from deepxde.backend import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import copy
from datetime import datetime
from numpy import genfromtxt
import matplotlib.ticker as ticker
from scipy.interpolate import griddata
from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator
from absl import app, flags
from ml_collections import config_flags
import importlib.util
# dde.config.set_default_float("float64")
# torch.set_default_dtype(torch.float64)
import warnings
import wandb
warnings.filterwarnings("ignore", category=DeprecationWarning)
######################################################################

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["cmr10"],      
    "mathtext.fontset": "cm",     
    "axes.formatter.use_mathtext": True,
    "font.size": 11,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.2,
})

def inclined_backward_step(angle_deg):
    
    dist = 2.5/np.tan(angle_deg*np.pi/180)
    xy2 = np.array([    [-2., 1],
                        [-1., 1],
                        [-0.6923076923076925, 1],
                        [0., 1.],
                        [dist, -1.5], #[0.1,0.18,0.27,0.5,1,1.8] = angle from bottomwall [85, 80, 75, 63, 45, 30]
                        [5.2153846153846146, -1.5],
                        # [7., -1.5],
                        # [25, -1.5]])
                        [20, -1.5],
                        [42, -1.5]])
    x = xy2[:,0]/6 - (xy2[:,0]/6)[0]  # scale to within 2 and with start as 0
    y2 = xy2[:,1]/6
    y1 = np.ones(x.shape)*y2[0]*3.5
    Ld = x[-1]
    return x,y1,y2,Ld

def input_transform(x):
    x_scaled = (x[:,0:1]-x_min)/(x_max-x_min)
    y_scaled = (x[:,1:2]-y_min)/(y_max-y_min)
    if fourier_enabled:
        xy = torch.cat([x_scaled, y_scaled], dim=1)
        proj = xy @ B.T
        # proj returns N,128 -> N, 2 and 2, 128
        # returns N, 256 due to N,128 and N, 128 stacked side by side , hence input is 256 for each point
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=1) 
    else:
        return torch.cat([x_scaled, y_scaled], dim=1)

def sdf_query(x_tensor, y_tensor):
    # device = x_tensor.device
    # grid = sdf_grid_tensor.to(device)

    # normalize coords to [-1, 1] for grid_sample
    x_norm = 2 * (x_tensor - x_[0]) / (Ld - x_[0]) - 1
    y_norm = 2 * (y_tensor - y2_[-1]) / (y1_[0] - y2_[-1]) - 1

    sample_pts = torch.cat([x_norm, y_norm], dim=1).unsqueeze(0).unsqueeze(0)  # (1,1,N,2)
    vals = torch.nn.functional.grid_sample(
        sdf_grid_tensor, sample_pts, mode='bilinear', align_corners=True, padding_mode='border'
    )
    return vals.view(-1, 1)

NU_TILDE_SCALE = 0.01

def output_transform(x, y):
    u = y[:, 0:1]
    v = y[:, 1:2]
    p = y[:, 2:3]
    nu_tilde_raw = y[:, 3:4]
    xcoord = x[:, 0:1]
    ycoord = x[:, 1:2]
    if bc_enforcement:
        d = sdf_query(xcoord, ycoord)
        # d = torch.tanh(d / 0.02)
        # d = torch.where(d > 0.02, torch.ones_like(d), torch.zeros_like(d))
        u = d * u
        v = d * v
        nu_tilde = torch.abs(nu_tilde_raw)
        return torch.cat([u, v, p, nu_tilde], dim=1)
    else:
        # nu_tilde = NU_TILDE_SCALE * torch.sigmoid(nu_tilde_raw)
        # nu_tilde = nu_tilde_raw**2
        nu_tilde = torch.abs(nu_tilde_raw)
        # softplus has no upper bound - so try this
        # nu_tilde = torch.nn.functional.softplus(nu_tilde_raw)
        return torch.cat([u, v, p, nu_tilde], dim=1)

######################################################################
#################RESTORE###############################################################################################################################################
pt_path = "20260825_184640/net_finish.pt"
base_dir = Path(pt_path).parent
folder_name = f"{base_dir}/post"
os.makedirs(folder_name, exist_ok=True)

file = open(f"runs/{base_dir}/config_used.json")
data = json.load(file)
# W&B
wandb_project = data["wandb"]["project"]
wandb_name = data["wandb"]["name"]
wandb_tag = data["wandb"]["tag"]
case_folder = data["wandb"]["folder"]

# Arch
num_neurons = data["arch"]["hidden_dim"]
num_layers = data["arch"]["num_layers"]
num_out_dim = data["arch"]["out_dim"]
num_in_dim = data["arch"]["input_dim"]
activation = data["arch"]["activation"]
fourier_enabled = data["arch"]["fourier_enabled"]
fourier_amp = data["arch"]["fourier_emb"]["embed_scale"]
fourier_dim = data["arch"]["fourier_emb"]["embed_dim"]

# Optim
optimizer_1 = data["optim"]["optimizer_1"]
optimizer_2 = data["optim"]["optimizer_2"]
use_lbfgs = data["optim"]["use_lbfgs"]
lr_schedule = data["optim"]["lr_schedule"]
learning_rate = data["optim"]["learning_rate"]
decay_rate = data["optim"]["decay_rate"]
decay_steps = data["optim"]["decay_steps"]

# Training
SEED = data["training"]["seed"]
itr_1 = data["training"]["max_steps_optimizer_1"]
itr_2 = data["training"]["max_steps_optimizer_2"]
sampling = data["training"]["sampling"]
resample_every = data["training"]["sampling_schedule"]["every"]
anchor_incl = data["training"]["anchors"]
domain_pts = data["training"]["sampling_points_domain"]
bc_pts = data["training"]["sampling_points_boundary"]
bc_enforcement = data["training"]["exact_bc_enforcement"]
rar_enabled = data["training"]["rar_enabled"]
rar_points = data["training"]["rar_points"]
rar_itr_ = data["training"]["rar_itr_"]

# Loss weighting
loss_strategy_enabled = data["loss_weighting"]["gradnorm"]
loss_weights_dict = data["loss_weighting"]["loss_weights"]
loss_update_start = data["loss_weighting"]["update_schedule"]["start"]
loss_update_every = data["loss_weighting"]["update_schedule"]["every"]
loss_momentum = data["loss_weighting"]["momentum"]
loss_clip = data["loss_weighting"]["clip"]
loss_explicit_enabled = data["loss_weighting"]["explicit_enabled"]
loss_explicit_weight_scale = data["loss_weighting"]["explicit_scale"]
loss_indices = data["loss_weighting"]["explicit_indices"]

# Pseudo-time
pseudo_time_enabled = data["pseudo_time"]["enabled"]
pseudo_time_strategy = data["pseudo_time"]["strategy"]
pts_tau = data["pseudo_time"]["pts_tau"]
pts_momentum = data["pseudo_time"]["momentum"]
prev_net_update_every = data["pseudo_time"]["prev_net_update_every"]

# Re ramp
re_ramp_enabled = data["re_ramp"]["enabled"]
re_ramp_schedule = data["re_ramp"]["schedule"]

# Logging
log_every_steps = data["logging"]["log_every_steps"]
log_errors = data["logging"]["log_errors"]
log_lr = data["logging"]["log_lr"]
log_losses = data["logging"]["log_losses"]
log_raw_losses = data["logging"]["log_raw_losses"]
log_loss_weights = data["logging"]["log_loss_weights"]
log_pts_weights = data["logging"]["log_pts_weights"]

# Saving
ckpt_path = data["saving"]["ckpt_path"]
save_every_steps = data["saving"]["save_every_steps"]
num_keep_ckpts = data["saving"]["num_keep_ckpts"]

file.close()

# set seed 
np.random.seed(SEED)
torch.manual_seed(SEED)
dde.config.set_random_seed(SEED)

labels_keys = list(loss_weights_dict.keys())
labels = [lbl.replace("_", " ") for lbl in labels_keys]  

# Case variables 
Re = 36000
theta = 45

if fourier_enabled:
    # change the amplitude to 0.5 or 1 if required in fourier lift
    B = torch.load(f"runs/{base_dir}/fourier_B.pt")

x_, y1_, y2_, Ld = inclined_backward_step(theta)
geom = dde.geometry.Polygon([
    [x_[0], y1_[0]], [x_[-1], y1_[-1]], [x_[-1], y2_[-1]], [x_[6], y2_[6]],
    [x_[5], y2_[5]], [x_[4], y2_[4]], [x_[3], y2_[3]],
    [x_[2], y2_[2]], [x_[1], y2_[1]], [x_[0], y2_[0]]
])

x_min, x_max = x_[0], x_[7]
y_min, y_max = y2_[6], y1_[0]

net = torch.load(f"runs/{base_dir}/net_finish.pt", weights_only=False) 
net.eval()

with open(os.path.join(f"runs/{base_dir}", "wandb_run_info.json")) as f:      
    run_info = json.load(f)
API = wandb.Api()  
run = API.run(f"{run_info['wandb_project']}/{run_info['wandb_run_id']}")
#################RESTORE###############################################################################################################################################
cv1 = 7.1
thetalist = [45]
for theta_plot in thetalist:
    f = open(f"{folder_name}/details_theta{theta_plot}.txt", 'w')
    x_dense = np.linspace(x_[0], x_[-1], 500)
    y_dense = np.linspace(y2_[-1], y1_[0], 200)
    X, Y = np.meshgrid(x_dense, y_dense)
    points = np.hstack((X.ravel()[:, None], Y.ravel()[:, None]))
    inside_mask = geom.inside(points)
    points_t = torch.tensor(points, dtype=torch.float32)
    with torch.no_grad():
        prediction = net(points_t).cpu().numpy()
    if isinstance(prediction, tuple) or isinstance(prediction, list):
        prediction = prediction[0]
    u_component = prediction[:, 0] 
    v_component = prediction[:, 1]
    p           = prediction[:,2]
    nu_tilde    = prediction[:,3]
    u_mag_flat = np.sqrt(u_component**2 + v_component**2)
    u_mag_flat[~inside_mask] = np.nan
    U_mag = u_mag_flat.reshape(X.shape)

    plt.figure(figsize=(10, 5), dpi=300)
    contour = plt.contourf(X, Y, U_mag, levels=200, cmap='Spectral_r', antialiased=True)
    cbar = plt.colorbar(contour, orientation='horizontal', pad=0.15, shrink=0.3)
    cbar.ax.tick_params(labelsize=5)
    cbar.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    cbar.set_label('U Magnitude', fontsize=7)
    plt.xlim(x_[0] - 0.1, x_[-1] + 0.1)
    plt.ylim(y2_[-1] - 0.1, y1_[0] + 0.1)
    plt.gca().set_aspect('equal')
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.xlabel("X coord.", fontsize=7)
    plt.ylabel("Y coord.", fontsize=7)
    plt.title(f"Velocity Field (theta={theta_plot} deg)", fontsize=7)
    plt.savefig(f"{folder_name}/velContour_theta{theta_plot}.png", bbox_inches='tight', dpi=300)
    plt.close()

    u_component[~inside_mask] = np.nan
    v_component[~inside_mask] = np.nan
    u_component = u_component.reshape(X.shape)
    v_component = v_component.reshape(X.shape)
    plt.figure(figsize=(10, 5), dpi=300)
    strm = plt.streamplot(
        X, Y, u_component, v_component,
        color=np.sqrt(u_component**2 + v_component**2), 
        cmap='Spectral_r',
        density=3.0,      
        linewidth=0.3,     
        arrowsize=0.5,      
        minlength=0.05,
    )
    cbar = plt.colorbar(strm.lines, orientation='horizontal', pad=0.15, shrink=0.3)
    cbar.ax.tick_params(labelsize=5)
    cbar.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    cbar.set_label('U Magnitude', fontsize=7)
    plt.xlim(x_[0] - 0.1, x_[-1] + 0.1)
    plt.ylim(y2_[-1] - 0.1, y1_[0] + 0.1)
    plt.gca().set_aspect('equal')
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.xlabel("X coord.", fontsize=7)
    plt.ylabel("Y coord.", fontsize=7)
    umax = np.nanmax(U_mag)
    plt.title(f"Velocity Field {umax}", fontsize=7)
    # plt.box(False)
    plt.savefig(f"{folder_name}/velStream_theta{theta_plot}.png", bbox_inches='tight', dpi=300)


    plt.figure(figsize=(10, 5), dpi=300)
    p[~inside_mask] = np.nan
    P_mag = p.reshape(X.shape)
    contour = plt.contourf(X, Y, P_mag, levels=200, cmap='Spectral_r', antialiased=True)
    cbar = plt.colorbar(contour, orientation='horizontal', pad=0.15, shrink=0.3)
    cbar.ax.tick_params(labelsize=5)
    cbar.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    cbar.set_label('P Magnitude', fontsize=7)
    plt.xlim(x_[0] - 0.1, x_[-1] + 0.1)
    plt.ylim(y2_[-1] - 0.1, y1_[0] + 0.1)
    plt.gca().set_aspect('equal')
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.xlabel("X coord.", fontsize=7)
    plt.ylabel("Y coord.", fontsize=7)
    plt.title("Pressure Field", fontsize=7)
    plt.savefig(f"{folder_name}/pres_contour_theta{theta_plot}.png", bbox_inches='tight', dpi=300)

    ##### Turbulence Intensity using Bradshaw constant #####
    points_t = torch.tensor(points, dtype=torch.float32, requires_grad=True)
    prediction_t = net(points_t) 
    u_y = dde.grad.jacobian(prediction_t, points_t, i=0, j=1)
    v_x = dde.grad.jacobian(prediction_t, points_t, i=1, j=0)
    nu_tilde = prediction_t[:, 3:4]
    xi = nu_tilde * Re          
    fv1 = xi**3 / (xi**3 + cv1**3 + 1e-6)
    nu_t = nu_tilde * fv1

    nu_t_np = nu_t.detach().cpu().numpy().flatten()
    nu_t_np[~inside_mask] = np.nan
    norm_nu_t_np = nu_t_np * Re
    nu_t_mag = nu_t_np.reshape(X.shape)
    norm_nu_t_mag = norm_nu_t_np.reshape(X.shape)
    numax = np.nanmax(nu_t_mag)
    norm_numax = np.nanmax(norm_nu_t_mag)

    plt.figure(figsize=(10, 5), dpi=300)
    contour = plt.contourf(X, Y, nu_t_mag, levels=200, cmap='Spectral_r', antialiased=True)
    cbar = plt.colorbar(contour, orientation='horizontal', pad=0.15, shrink=0.3)
    cbar.ax.tick_params(labelsize=5)
    cbar.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    cbar.set_label(r'$\nu_{t}$ Magnitude', fontsize=7)
    plt.xlim(x_[0] - 0.1, x_[-1] + 0.1)
    plt.ylim(y2_[-1] - 0.1, y1_[0] + 0.1)
    plt.gca().set_aspect('equal')
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.xlabel("X coord.", fontsize=7)
    plt.ylabel("Y coord.", fontsize=7)
    plt.title(rf"$\nu_{{t}}$ Field (theta={theta_plot} deg, max {numax})", fontsize=7)
    plt.savefig(f"{folder_name}/nu_t.png", bbox_inches='tight', dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5), dpi=300)
    contour = plt.contourf(X, Y, norm_nu_t_mag, levels=200, cmap='Spectral_r', antialiased=True)
    cbar = plt.colorbar(contour, orientation='horizontal', pad=0.15, shrink=0.3)
    cbar.ax.tick_params(labelsize=5)
    cbar.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1e'))
    cbar.set_label(r"$\nu_{t}/ \nu$ Magnitude", fontsize=7)
    plt.xlim(x_[0] - 0.1, x_[-1] + 0.1)
    plt.ylim(y2_[-1] - 0.1, y1_[0] + 0.1)
    plt.gca().set_aspect('equal')
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.xlabel("X coord.", fontsize=7)
    plt.ylabel("Y coord.", fontsize=7)
    plt.title(rf"Normalized Eddy viscosity field $\nu_{{t}} / \nu $ (deg, max {norm_numax})", fontsize=7)
    plt.savefig(f"{folder_name}/nu_t_normalized.png", bbox_inches='tight', dpi=300)
    plt.close()

    R12 = nu_t * (u_y + v_x)
    #Bradshaw constant = 0.3  
    k = torch.abs(R12) / 0.3
    U_ref = 1
    TI = torch.sqrt(2/3 * k) / U_ref
    TI = TI * 100

    TI_np = TI.detach().cpu().numpy().flatten()
    TI_np[~inside_mask] = np.nan
    TI_mag = TI_np.reshape(X.shape)
    TImax = np.nanmax(TI_mag)

    plt.figure(figsize=(10, 5), dpi=300)
    contour = plt.contourf(X, Y, TI_mag, levels=200, cmap='Spectral_r', antialiased=True)
    cbar = plt.colorbar(contour, orientation='horizontal', pad=0.15, shrink=0.3)
    cbar.ax.tick_params(labelsize=5)
    cbar.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    cbar.set_label('Turbulence Intensity %', fontsize=7)
    plt.xlim(x_[0] - 0.1, x_[-1] + 0.1)
    plt.ylim(y2_[-1] - 0.1, y1_[0] + 0.1)
    plt.gca().set_aspect('equal')
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.xlabel("X coord.", fontsize=7)
    plt.ylabel("Y coord.", fontsize=7)
    plt.title(f"Turbulence Intensity % (theta={theta_plot} deg, max {TImax})", fontsize=7)
    plt.savefig(f"{folder_name}/Turbulence_Intensity.png", bbox_inches='tight', dpi=300)
    plt.close()
    ##########################################################

    # sample u very close to the wall, downstream of the step
    y_wall = y2_[-1] + 1e-4 
    x_scan = np.linspace(x_[4], Ld, 100)
    pts = np.column_stack((x_scan, np.full_like(x_scan, y_wall)))
    points_t = torch.tensor(pts, dtype=torch.float32)
    with torch.no_grad():
        prediction = net(points_t).cpu().numpy()
    u_wall = prediction[:, 0]
    # reattachment point ≈ where u_wall crosses from negative to positive
    sign_change = np.where(np.diff(np.sign(u_wall)) > 0)[0]
    # x_reattach = x_scan[sign_change[0]] if len(sign_change) else None
    # slope of a line where y=0, and find x where it crosses 0
    if len(sign_change) > 0:
        i = sign_change[0]  
        x1, x2 = x_scan[i], x_scan[i+1]
        u1, u2 = u_wall[i], u_wall[i+1]
        x_reattach = x1 - u1*(x2 - x1)/(u2-u1)
    else:
        x_reattach = None
    # with open(f"{folder_name}/details_theta{theta_plot}.txt", "w") as f:
    f.write(f"Reattachment length = {x_reattach}")
    # f.close()

    if Re == 36000:
        #plot velocity profiles
        plt.figure(figsize=(10, 8))
        step_h = 0.41667
        step_x = 0.333334
        k_ = 1
        locs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        colors = plt.cm.tab20(np.linspace(0, 1, 11))
        total_num_exp = 0.0
        total_den_exp = 0.0
        total_num_fem = 0.0
        total_den_fem = 0.0
        for i, loc in enumerate(locs):

            y_fine = np.linspace(y2_[6], y1_[0], 200)   
            x_fine = np.full(y_fine.shape[0], step_x + step_h * loc)
            grid_fine = np.column_stack((x_fine, y_fine))
            points_t = torch.tensor(grid_fine, dtype=torch.float32)
            with torch.no_grad():
                prediction = net(points_t).cpu().numpy()
            u_pred_fine = prediction[:, 0]     
            # y_plot_scaled = (y_fine - y2_[6]) * 2.4
            x_plot_pred = loc + u_pred_fine     
            plt.plot(x_plot_pred, y_fine, color=colors[i], label=f"PINN x/H={loc}")

            # experiment########################
            file_path = f'makiola1/experiment/loc{loc}.csv'
            my_data = np.genfromtxt(file_path, delimiter=',')
            x_global = my_data[:, 0]
            y_data = my_data[:, 1]
            u_over_U0 = my_data[:,2]
            # u_over_U0 = (x_global - loc) / k
            x_plot_data = loc + u_over_U0
            y_plot_scaled = (y_data * 0.4165) + y2_[6]
            plt.axvline(x=loc, color=colors[i], linestyle='--', alpha=0.5)
            label = "Makiola et al. Experiment" if i == 0 else ""
            plt.scatter(x_plot_data, y_plot_scaled, label=label, color='gray', marker='o')

            x_coord_new = np.full(y_data.shape[0], step_x + step_h * loc)
            onedgrid = np.column_stack((x_coord_new, y_plot_scaled))
            points_t = torch.tensor(onedgrid, dtype=torch.float32)
            with torch.no_grad():
                prediction = net(points_t).cpu().numpy()
            u_pred_for_exp = prediction[:, 0]

            diff = u_pred_for_exp - u_over_U0
            l2_num_exp = np.sqrt(np.sum(diff**2))
            l2_den_exp = np.sqrt(np.sum(u_over_U0**2))
            l2_rel_exp = (l2_num_exp / l2_den_exp) * 100
            total_num_exp += np.sum(diff**2)
            total_den_exp += np.sum(u_over_U0**2)
            #######################################

            # fem########################
            file_path = f'makiola1/fem/loc{loc}.csv'
            my_data = np.genfromtxt(file_path, delimiter=',')
            x_global = my_data[:, 0]
            y_data = my_data[:, 1]

            u_over_U0 = (x_global - loc) / k_
            x_plot_data = loc + u_over_U0
            y_plot_scaled = (y_data * 0.4165) + y2_[6]
            plt.axvline(x=loc, color=colors[i], linestyle='--', alpha=0.5)
            label = "FEM Louda et al." if i == 0 else ""
            plt.plot(x_plot_data, y_plot_scaled, label=label, color='steelblue', marker='^')

            x_coord_new = np.full(y_data.shape[0], step_x + step_h * loc)
            onedgrid = np.column_stack((x_coord_new, y_plot_scaled))
            points_t = torch.tensor(onedgrid, dtype=torch.float32)
            with torch.no_grad():
                prediction = net(points_t).cpu().numpy()
            u_pred_for_fem = prediction[:, 0]

            diff = u_pred_for_fem - u_over_U0
            l2_num_fem = np.sqrt(np.sum(diff**2))
            l2_den_fem = np.sqrt(np.sum(u_over_U0**2))
            l2_rel_fem = (l2_num_fem / l2_den_fem) * 100
            total_num_fem += np.sum(diff**2)
            total_den_fem += np.sum(u_over_U0**2)
            #######################################

            f.write(f"Station x/S={loc}: EXP abs L2={l2_num_exp:.5f}, rel L2={l2_rel_exp:.5f} % | FEM abs L2={l2_num_fem:.5f}, rel L2={l2_rel_fem:.5f} %\n")

        overall_rel_l2_exp = ( np.sqrt(total_num_exp) / np.sqrt(total_den_exp) ) * 100
        overall_rel_l2_fem = ( np.sqrt(total_num_fem) / np.sqrt(total_den_fem) ) * 100
        f.write(f"Overall relative L2 (experiment): {overall_rel_l2_exp:.5f} %\n")
        f.write(f"Overall relative L2 (FEM): {overall_rel_l2_fem:.5f} %\n")
        f.close()

        plt.xlabel('x/H')
        plt.ylabel('y/H')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title("Velocity Profiles")
        plt.savefig(f"{folder_name}/velocity_profile_Re{Re}.png", bbox_inches='tight')
    #############################################################
    #3#Loss Curve
    plt.figure(figsize=(10, 6))
    keys = ["_step"] + [f"train/{lbl}" for lbl in labels]  
    # keys = [_step, xmom, ymom, etc..]
    history = run.history(keys=keys, samples=10000)  
    adam_end = itr_1
    plt.figure(figsize=(10, 6))  
    plt.gca().set_prop_cycle(color=plt.cm.tab20(np.linspace(0, 1, 14)))  
    for lbl in labels:
        steps = history["_step"]
        vals = history[f"train/{lbl}"]
        vals = vals / vals.iloc[0]
        mask = steps <= adam_end
        plt.semilogy(steps[mask], vals[mask] + 1e-10, label=lbl)   
        plt.semilogy(steps[~mask], vals[~mask] + 1e-10, linestyle="--")  

    plt.axvline(x=adam_end, color="black", linestyle=":", alpha=0.5)  
    plt.xlabel("Iterations")  
    plt.ylabel("Normalized Train Loss ($L / L_{initial}$)")  
    plt.title("Train Loss (solid=Adam, dashed=L-BFGS)")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))  
    plt.grid(True, which="both", ls="-", alpha=0.3)  
    plt.tight_layout()
    plt.savefig(f"{folder_name}/training_loss_wandb.png", dpi=150, bbox_inches="tight")  
    plt.close()

    plt.figure(figsize=(10, 6))
    keys = ["_step"] + [f"test/{lbl}" for lbl in labels]  
    history = run.history(keys=keys, samples=10000)  
    adam_end = itr_1
    plt.figure(figsize=(10, 6))  
    plt.gca().set_prop_cycle(color=plt.cm.tab20(np.linspace(0, 1, 14)))  
    for lbl in labels:
        steps = history["_step"]
        vals = history[f"test/{lbl}"]
        vals = vals / vals.iloc[0]
        mask = steps <= adam_end
        plt.semilogy(steps[mask], vals[mask] + 1e-10, label=lbl)   
        plt.semilogy(steps[~mask], vals[~mask] + 1e-10, linestyle="--")  

    plt.axvline(x=adam_end, color="black", linestyle=":", alpha=0.5)  
    plt.xlabel("Iterations")  
    plt.ylabel("Normalized Test Loss ($L / L_{initial}$)")  
    plt.title("Test Loss (solid=Adam, dashed=L-BFGS)")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))  
    plt.grid(True, which="both", ls="-", alpha=0.3)  
    plt.tight_layout()
    plt.savefig(f"{folder_name}/test_loss_wandb.png", dpi=150, bbox_inches="tight")  
    plt.close()
    #############################################################
    #############################################################

