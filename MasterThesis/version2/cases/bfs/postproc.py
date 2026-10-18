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


def input_transform(x):
    x_scaled = (x[:,0:1]-x_min)/(x_max-x_min)
    y_scaled = (x[:,1:2]-y_min)/(y_max-y_min)
    xy = torch.cat([x_scaled, y_scaled], dim=1)
    proj = xy @ B.T
    # proj returns N,128 -> N, 2 and 2, 128
    # returns N, 256 due to N,128 and N, 128 stacked side by side , hence input is 256 for each point
    return torch.cat([torch.cos(proj), torch.sin(proj)], dim=1) 
    # return torch.cat([x_scaled, y_scaled], dim=1) 

def output_transform(x, y):
    m=2
    u = y[:, 0:1]
    v = y[:, 1:2]
    p = y[:, 2:3]
    xcoord = x[:, 0:1]
    ycoord = x[:, 1:2]

    if bc_enforcement:
        phi_top = linseg_adf(x,y,0.0,2.061,30.0,2.061,30.0)
        phi_bottom = linseg_adf(x,y,2.0,0.0,30.0,0.0,28.0)
        phi_step_hor = linseg_adf(x,y,0.0,1.0,2.0,1.0,2.0)
        phi_step_ver = linseg_adf(x,y,2.0,1.0,2.0,0.0,1.0)
        d = 1 / ( (1/phi_top**m + 1/phi_bottom**m + 1/phi_step_hor**m + 1/phi_step_ver**m) )**(1/m)
        # d = torch.tanh(d / 0.02)
        # d = torch.where(d > 0.02, torch.ones_like(d), torch.zeros_like(d))
        u = d * u
        v = d * v
        return torch.cat([u, v, p], dim=1)
    else:
        return torch.cat([u, v, p], dim=1)

def linseg_adf(x, y, x1, y1, x2, y2, L, eps=1e-8):
    L = torch.tensor(L, dtype=x.dtype, device=x.device)
    xc, yc = (x1+x2)/2, (y1+y2)/2
    f = ((x-x1)*(y2-y1) - (y-y1)*(x2-x1)) / L
    t = (1/L) * ((L/2)**2 - torch.sqrt((x-xc)**2 + (y-yc)**2))
    varphi = torch.sqrt(t**2 + f**4 + eps)
    phi = torch.sqrt(f**2 + 0.25*(varphi - t)**2 + eps)

    return phi
######################################################################
#################RESTORE###############################################################################################################################################
pt_path = "20260826_141044/net_finish.pt"
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
Re = 183.3

if fourier_enabled:
    # change the amplitude to 0.5 or 1 if required in fourier lift
    B = torch.load(f"runs/{base_dir}/fourier_B.pt")

geom = dde.geometry.Polygon([
        [0.0, 2.061], [30.0, 2.061], [30.0, 0.0], [2.0, 0.0], [2.0, 1.0],
        [0.0, 1.0]
])

net = torch.load(f"runs/{base_dir}/net_finish.pt", weights_only=False) 
net.eval()

with open(os.path.join(f"runs/{base_dir}", "wandb_run_info.json")) as f:      
    run_info = json.load(f)
API = wandb.Api()  
run = API.run(f"{run_info['wandb_project']}/{run_info['wandb_run_id']}")
#################RESTORE###############################################################################################################################################
Relist = [183.3]
x_dense = np.linspace(0.0, 30.0, 500)
y_dense = np.linspace(0.0, 2.061, 200)
X, Y = np.meshgrid(x_dense, y_dense)
points = np.vstack((X.ravel(), Y.ravel())).T
for Re in Relist:
    f = open(f"{folder_name}/details.txt", 'w')
    # Re_column = np.full((points.shape[0], 1), float(Re))
    # X_3D = np.hstack((points, Re_column))
    points_t = torch.tensor(points, dtype=torch.float32)
    with torch.no_grad():
        prediction = net(points_t).cpu().numpy()
    if isinstance(prediction, tuple) or isinstance(prediction, list):
        prediction = prediction[0]
    u_component = prediction[:, 0]
    v_component = prediction[:, 1]
    p = prediction[:,2]
    u_mag_flat = np.sqrt(u_component**2 + v_component**2)
    U_mag = u_mag_flat.reshape(X.shape)
    U_mag[(X < 2.0) & (Y < 1.0)] = np.nan
    plt.figure(figsize=(10, 5), dpi=300)
    contour = plt.contourf(X, Y, U_mag, levels=200, cmap='Spectral_r', antialiased=True)
    cbar = plt.colorbar(contour, orientation='horizontal', pad=0.15, shrink=0.3)
    cbar.ax.tick_params(labelsize=5)
    cbar.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    cbar.set_label('U Magnitude', fontsize=7)
    plt.xlim(-0.1, 30.1)
    plt.ylim(-0.1, 2.1)
    plt.gca().set_aspect('equal')
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.xlabel("X coord.", fontsize=7)
    plt.ylabel("Y coord.", fontsize=7)
    plt.title("Velocity Field", fontsize=7)
    # plt.box(False)
    plt.savefig(f"{folder_name}/vel_contour_Re{Re}.png", bbox_inches='tight', dpi=300)

    plt.figure(figsize=(10, 5), dpi=600)
    P_mag = p.reshape(X.shape)
    P_mag[(X < 2.0) & (Y < 1.0)] = np.nan
    contour = plt.contourf(X, Y, P_mag, levels=200, cmap='Spectral_r', antialiased=True)
    cbar = plt.colorbar(contour, orientation='horizontal', pad=0.15, shrink=0.3)
    cbar.ax.tick_params(labelsize=5)
    cbar.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    cbar.set_label('P Magnitude', fontsize=7)
    plt.xlim(-0.1, 30.1)
    plt.ylim(-0.1, 2.1)
    plt.gca().set_aspect('equal')
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.xlabel("X coord.", fontsize=7)
    plt.ylabel("Y coord.", fontsize=7)
    plt.title("Pressure Field", fontsize=7)
    plt.savefig(f"{folder_name}/pres_contour_Re{Re}.png", bbox_inches='tight', dpi=600)

    if Re == 183.3:
        #plot velocity profiles
        plt.figure(figsize=(10, 8))
        step_h = 1
        step_x = 2
        # Umax = 0.879542893
        # U0 = (2/3) * Umax
        # locs = [0, 3.06, 6.12, 9.74, 11.84]
        locs = [0, 3.06, 5.41, 7.14, 9.74, 11.84]
        colors = ['b', 'g', 'r', 'c', 'm', 'tab:orange']
        total_num = 0.0
        total_den = 0.0
        for i, loc in enumerate(locs):
            y_coord_new = np.linspace(0, 2.061, 22) 
            x_coord_new = np.full(y_coord_new.shape[0], step_x + step_h * loc)
            # re_line_column = np.full(y_coord_new.shape[0], float(Re))
            onedgrid = np.column_stack((x_coord_new, y_coord_new))
            points_t = torch.tensor(points, dtype=torch.float32)
            with torch.no_grad():
                prediction = net(points_t).cpu().numpy()
            u_pred = prediction[:,0]
            y_plot_scaled = y_coord_new * 4.9
            # x_plot_data = loc + (u_pred / U0)
            x_plot_data = loc + u_pred
            plt.plot(x_plot_data, y_plot_scaled, color=colors[i], label=f"PINN z'/S = {loc}")
            file_path = f'armaly1/loc{loc}.csv'
            my_data = np.genfromtxt(file_path, delimiter=',')
            armaly_x = my_data[:,0]
            armaly_y = my_data[:,1]
            armaly_x_normalized = armaly_x / (87.954 * (2/3))
            armaly_x_final = loc + armaly_x_normalized
            armaly_y_scaled = armaly_y / 4.9
            plt.axvline(x=loc, color=colors[i], linestyle='--', alpha=0.5)
            armaly_label = "Armaly et.al (1983)" if i == 0 else ""
            plt.scatter(armaly_x_final, armaly_y, label=armaly_label, color='gray', marker='o')

            x_coord_new = np.full(armaly_y_scaled.shape[0], step_x + step_h * loc)
            onedgrid = np.column_stack((x_coord_new, armaly_y_scaled))
            points_t = torch.tensor(onedgrid, dtype=torch.float32)
            with torch.no_grad():
                prediction = net(points_t).cpu().numpy()
            u_pred = prediction[:, 0]
         
            diff = u_pred - armaly_x_normalized
            l2_num = np.sqrt(np.sum(diff**2))
            l2_den = np.sqrt(np.sum(armaly_x_normalized**2))
            l2_rel = (l2_num / l2_den) * 100
         
            total_num += np.sum(diff**2)
            total_den += np.sum(armaly_x_normalized**2)
         
            f.write(f"Station x/S={loc}: abs L2 = {l2_num:.5f}, relative L2 = {l2_rel:.5f} %\n")

        overall_rel_l2 = ( np.sqrt(total_num) / np.sqrt(total_den) ) * 100
        f.write(f"Overall relative L2 error: {overall_rel_l2:.5f} %\n")
     

        plt.xlabel("Adimensionalized x/S coordinate")
        plt.ylabel("y coordinate (mm)")
        plt.xlim(0, 15)
        plt.ylim(0, 10.2)  # Fits your geometry perfectly from floor to ceiling
        plt.grid(True, linestyle=':')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title("Velocity Profiles")
        plt.savefig(f"{folder_name}/velocity_profile_Re{Re}.png", dpi=400)

    # sample u very close to the wall, downstream of the step
    y_wall = 1e-4  
    x_scan = np.linspace(3.0, 30.0, 500)
    pts = np.column_stack((x_scan, np.full_like(x_scan, y_wall)))
    points_t = torch.tensor(onedgrid, dtype=torch.float32)
    with torch.no_grad():
        prediction = net(points_t).cpu().numpy()
    u_wall = prediction[:, 0]
    # reattachment point ≈ where u_wall crosses from negative to positive
    sign_change = np.where(np.diff(np.sign(u_wall)) > 0)[0]
    if len(sign_change) > 0:
        i = sign_change[0]   
        x1, x2 = x_scan[i], x_scan[i+1]
        u1, u2 = u_wall[i], u_wall[i+1]
        x_reattach = x1 - u1*(x2 - x1)/(u2-u1)
    else:
        x_reattach = None
    f.write(f"lr={lr}, neurons={neurons}, layers={layers}, act={act}, iterations={itr}, Reattachment length = {x_reattach}")
    f.close()

    # #############################################################
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

