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
    if activation == "tanh":
        x_scaled = 2 * (x[:,0:1]-x_min)/(x_max-x_min) - 1
        y_scaled = 2 * (x[:,1:2]-y_min)/(y_max-y_min) - 1
    else:
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
    x_norm = 2 * (x_tensor - x_min) / (x_max - x_min) - 1
    y_norm = 2 * (y_tensor - y_min) / (y_max - y_min) - 1

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
        nu_tilde = NU_TILDE_SCALE * torch.sigmoid(nu_tilde_raw)
        # nu_tilde = nu_tilde_raw**2
        # nu_tilde = torch.abs(nu_tilde_raw)
        # softplus has no upper bound - so try this
        # nu_tilde = torch.nn.functional.softplus(nu_tilde_raw)
        return torch.cat([u, v, p, nu_tilde], dim=1)

######################################################################
#################RESTORE###############################################################################################################################################
pt_path = "20260901_120534/net_finish.pt"
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
# labels = [lbl.replace("_", " ") for lbl in labels_keys]  

# Case variables 
Re = 36000
theta = 45

if fourier_enabled:
    # change the amplitude to 0.5 or 1 if required in fourier lift
    B = torch.load(f"runs/{base_dir}/fourier_B.pt")

#Top wall coords
top_coords = np.load('sduct_topwall.npy')
#Bottom wall coords
bottom_coords = np.flipud(np.load('sduct_bottomwall.npy'))
complete_coords = np.vstack([top_coords, bottom_coords])
x_min, x_max = complete_coords[:, 0].min(), complete_coords[:, 0].max()
y_min, y_max = complete_coords[:, 1].min(), complete_coords[:, 1].max()
geom = dde.geometry.Polygon(complete_coords)

net = torch.load(f"runs/{base_dir}/net_finish.pt", weights_only=False) 
net.eval()

with open(os.path.join(f"runs/{base_dir}", "wandb_run_info.json")) as f:      
    run_info = json.load(f)
API = wandb.Api()  
run = API.run(f"{run_info['wandb_project']}/{run_info['wandb_run_id']}")
#################RESTORE###############################################################################################################################################
cv1 = 7.1
Relist = [2.5e6]
x_min, x_max = complete_coords[:, 0].min(), complete_coords[:, 0].max()
y_min, y_max = complete_coords[:, 1].min(), complete_coords[:, 1].max()
x_range = np.linspace(x_min, x_max, 1200)
y_range = np.linspace(y_min, y_max, 1200)
xx, yy = np.meshgrid(x_range, y_range)
points = np.vstack((xx.ravel(), yy.ravel())).T

#Predict P at inlet and use that for C_p calculation
y_coord_new = 0.5000252
x_coord_new = 0.0016813294762442867
point = np.column_stack((x_coord_new, y_coord_new))
points_t = torch.tensor(point, dtype=torch.float32)
with torch.no_grad():
    prediction = net(points_t).cpu().numpy()
p_ref = prediction[:, 2]
print(f"p_ref : {p_ref}")

outlet_y = np.linspace(0.05999999999999983, 0.46, 20)
outlet_x = np.full_like(outlet_y, 1.888)
outlet_pts = np.column_stack((outlet_x, outlet_y))
points_t = torch.tensor(outlet_pts, dtype=torch.float32)
with torch.no_grad():
    prediction = net(points_t).cpu().numpy()
p_outlet = prediction[:,2]
print(p_outlet[0])

for Re in Relist:
    f = open(f"{folder_name}/details_Re{Re}.txt", 'w')
    # Re_column = np.full((points.shape[0], 1), float(Re))
    # X_3D = np.hstack((points, Re_column))
    inside_mask = geom.inside(points)
    points_t = torch.tensor(points, dtype=torch.float32)
    with torch.no_grad():
        prediction = net(points_t).cpu().numpy()
    u_component = prediction[:, 0]
    v_component = prediction[:, 1]
    p = prediction[:, 2]
    u_mag_flat = np.sqrt(u_component**2 + v_component**2)
    u_mag_flat[~inside_mask] = np.nan
    U_mag = u_mag_flat.reshape(xx.shape)
    plt.figure(figsize=(10, 5), dpi=300)
    contour = plt.contourf(xx, yy, U_mag, levels=200, cmap='Spectral_r', antialiased=True)
    cbar = plt.colorbar(contour, orientation='horizontal', pad=0.15, shrink=0.3)
    cbar.ax.tick_params(labelsize=5)
    cbar.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    cbar.set_label('U Magnitude', fontsize=7)
    plt.xlim(0.0, 1.9)
    plt.ylim(0.04, 0.71)
    plt.gca().set_aspect('equal')
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.xlabel("X coord.", fontsize=7)
    plt.ylabel("Y coord.", fontsize=7)
    plt.title(rf"Dimensionless Velocity Magnitude $\vert u \vert / U_0$", fontsize=7)
    plt.savefig(f"{folder_name}/vel_contour_Re{Re:.2e}.png", bbox_inches='tight', dpi=300)

    u_component[~inside_mask] = np.nan
    v_component[~inside_mask] = np.nan
    u_component = u_component.reshape(xx.shape)
    v_component = v_component.reshape(xx.shape)
    plt.figure(figsize=(10, 5), dpi=300)
    strm = plt.streamplot(
        xx, yy, u_component, v_component,
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
    plt.xlim(0.0, 1.9)
    plt.ylim(0.04, 0.71)
    plt.gca().set_aspect('equal')
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.xlabel("X coord.", fontsize=7)
    plt.ylabel("Y coord.", fontsize=7)
    umax = np.nanmax(U_mag)
    plt.title(f"Velocity Field {umax}", fontsize=7)
    # plt.box(False)
    plt.savefig(f"{folder_name}/velStream_Re{Re:.2e}.png", bbox_inches='tight', dpi=300)

    #C_p
    Cp = 2 * (p - p_outlet[0])
    plt.figure(figsize=(10, 5), dpi=300)
    Cp[~inside_mask] = np.nan
    Cp_mag = Cp.reshape(xx.shape)
    contour = plt.contourf(xx, yy, Cp_mag, levels=25, cmap='Spectral_r', antialiased=True)
    cbar = plt.colorbar(contour, orientation='horizontal', pad=0.15, shrink=0.3)
    cbar.ax.tick_params(labelsize=5)
    cbar.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    cbar.set_label(r"$C_{p}$", fontsize=7)
    plt.xlim(0.0, 1.9)
    plt.ylim(0.04, 0.71)
    plt.gca().set_aspect('equal')
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.xlabel("X coord.", fontsize=7)
    plt.ylabel("Y coord.", fontsize=7)
    plt.title(rf"Pressure Coefficient $C_{{p}}$", fontsize=7)
    plt.savefig(f"{folder_name}/pres_contour_Re{Re:.2e}.png", bbox_inches='tight', dpi=300)

    ##### Turbulence Intensity using Bradshaw constant #####
    points_t = torch.tensor(points, dtype=torch.float32, requires_grad=True)
    prediction_t = net(points_t) 
    # u_y = dde.grad.jacobian(prediction_t, points_t, i=0, j=1)
    # v_x = dde.grad.jacobian(prediction_t, points_t, i=1, j=0)
    v_y = dde.grad.jacobian(prediction_t, points_t, i=1, j=1)
    u_x = dde.grad.jacobian(prediction_t, points_t, i=0, j=0)
    nu_tilde = prediction_t[:, 3:4]
    xi = nu_tilde * Re          
    fv1 = xi**3 / (xi**3 + cv1**3 + 1e-6)
    nu_t = nu_tilde * fv1

    nu_t_np = nu_t.detach().cpu().numpy().flatten()
    nu_t_np[~inside_mask] = np.nan
    norm_nu_t_np = nu_t_np * Re
    nu_t_mag = nu_t_np.reshape(xx.shape)
    norm_nu_t_mag = norm_nu_t_np.reshape(xx.shape)
    numax = np.nanmax(nu_t_mag)
    norm_numax = np.nanmax(norm_nu_t_mag)

    plt.figure(figsize=(10, 5), dpi=300)
    contour = plt.contourf(xx, yy, nu_t_mag, levels=200, cmap='Spectral_r', antialiased=True)
    cbar = plt.colorbar(contour, orientation='horizontal', pad=0.15, shrink=0.3)
    cbar.ax.tick_params(labelsize=5)
    cbar.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    cbar.set_label(r"$\nu_{t}$ Magnitude", fontsize=7)
    plt.xlim(0.0, 1.9)
    plt.ylim(0.04, 0.71)
    plt.gca().set_aspect('equal')
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.xlabel("X coord.", fontsize=7)
    plt.ylabel("Y coord.", fontsize=7)
    plt.title(rf"$\nu_{{t}}$ Field (deg, max {numax})", fontsize=7)
    plt.savefig(f"{folder_name}/nu_t.png", bbox_inches='tight', dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5), dpi=300)
    contour = plt.contourf(xx, yy, norm_nu_t_mag, levels=200, cmap='Spectral_r', antialiased=True)
    cbar = plt.colorbar(contour, orientation='horizontal', pad=0.15, shrink=0.3)
    cbar.ax.tick_params(labelsize=5)
    cbar.locator = ticker.MaxNLocator(nbins=8)
    cbar.update_ticks()
    cbar.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1e'))
    cbar.set_label(r"$\nu_{t}/ \nu$ Magnitude", fontsize=7)
    plt.xlim(0.0, 1.9)
    plt.ylim(0.04, 0.71)
    plt.gca().set_aspect('equal')
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.xlabel("X coord.", fontsize=7)
    plt.ylabel("Y coord.", fontsize=7)
    plt.title(rf"Normalized Eddy viscosity field $\nu_{{t}} / \nu $ (deg, max {norm_numax})", fontsize=7)
    plt.savefig(f"{folder_name}/nu_t_normalized.png", bbox_inches='tight', dpi=300)
    plt.close()

    # R12 = nu_t * (u_y + v_x)
    #Bradshaw constant = 0.3  
    # k = torch.abs(R12) / 0.3
    u_u = 2 * nu_t * (u_x)
    v_v = 2 * nu_t * (v_y)
    k = 0.5 * (u_u + v_v)
    U_ref = 1
    TI = torch.sqrt(2/3 * k) / U_ref
    TI = TI * 100

    TI_np = TI.detach().cpu().numpy().flatten()
    TI_np[~inside_mask] = np.nan
    TI_mag = TI_np.reshape(xx.shape)
    TImax = np.nanmax(TI_mag)

    plt.figure(figsize=(10, 5), dpi=300)
    contour = plt.contourf(xx, yy, TI_mag, levels=200, cmap='Spectral_r', antialiased=True)
    cbar = plt.colorbar(contour, orientation='horizontal', pad=0.15, shrink=0.3)
    cbar.ax.tick_params(labelsize=5)
    cbar.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    cbar.set_label('Turbulence Intensity %', fontsize=7)
    plt.xlim(0.0, 1.9)
    plt.ylim(0.04, 0.71)
    plt.gca().set_aspect('equal')
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.xlabel("X coord.", fontsize=7)
    plt.ylabel("Y coord.", fontsize=7)
    plt.title(f"Turbulence Intensity % (deg, max {TImax})", fontsize=7)
    plt.savefig(f"{folder_name}/Turbulence_Intensity.png", bbox_inches='tight', dpi=300)
    plt.close()
    ##########################################################

    # #plot Cp and u profiles
    locs = [0.28003528, 0.2000252, 0.12001512]
    colors = ['b', 'g', 'r']
    label = ["U", "M", "D"]
    res = ["bruns", "garbaruk", "cnn"]
    labelt = ["Bruns et al.", "Garbaruk et al.", "Bokil et al. (RANS-CNN)"]
    res1 = ["bruns", "cnn"]
    labelt1 = ["Bruns et al.", "Bokil et al. (RANS-CNN)"]
    marktype = ["o", "", ""]
    total_num = {label: 0.0 for label in labelt}     
    total_den = {label: 0.0 for label in labelt}
    total_num_v = {label: 0.0 for label in labelt1}    
    total_den_v = {label: 0.0 for label in labelt1}
    for i, loc in enumerate(locs):
        bottom_coords_x = bottom_coords[:, 0:1]
        bottom_coords_y = bottom_coords[:, 1:2] + loc
        mid_curve = np.column_stack([bottom_coords_x, bottom_coords_y])
        points_t = torch.tensor(mid_curve, dtype=torch.float32)
        with torch.no_grad():
            prediction = net(points_t).cpu().numpy()
        u_pred = prediction[:, 0]
        v_pred = prediction[:, 1]
        p_pred = prediction[:, 2]
        Cp = 2 * (p_pred - p_outlet[0])
        # new_bottom_coords_x = (bottom_coords_x - x_inlet_pinn)/(x_outlet_pinn - x_inlet_pinn)
        new_bottom_coords_x = bottom_coords_x / 0.4
        plt.figure(figsize=(12, 8))
        plt.plot(new_bottom_coords_x, Cp, color='k', label=f"PINN")

        for j, type in enumerate(res):
            file_path = f'data/cp_{label[i]}_{type}.csv'
            my_data = np.genfromtxt(file_path, delimiter=',')
            x = my_data[:, 0]
            norm_x = x
            cp = my_data[:, 1]
            if j == 0:
                plt.scatter(norm_x, cp, label=f"{labelt[j]}", color=colors[j], marker='o')
            elif j == 1:
                plt.plot(norm_x, cp, color=colors[j], linestyle="-.", label=f"{labelt[j]}")
            else:
                plt.plot(norm_x, cp, color=colors[j], linestyle="--", label=f"{labelt[j]}")
            plt.xlabel("Normalized x coordinate")
            plt.ylabel(r"$C_{p}$")
            # plt.grid(True, linestyle=':')
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.title(rf"Pressure Coefficient $C_{{p}}$ at section {label[i]}")
            plt.tight_layout()
            plt.savefig(f"{folder_name}/cp_profile_{label[i]}.png", dpi=150)

            x_query = norm_x * 0.4
            y_at_query = np.interp(x_query, bottom_coords_x.flatten(), bottom_coords_y.flatten()) 
            onedgrid = np.column_stack((x_query, y_at_query))
            points_t = torch.tensor(onedgrid, dtype=torch.float32)
            with torch.no_grad():
                prediction = net(points_t).cpu().numpy()
            p_pred_ = prediction[:, 2]
            Cp_ = 2 * (p_pred_ - p_outlet[0])

            diff = Cp_ - cp
            l2_num = np.sqrt(np.sum(diff**2))
            l2_den = np.sqrt(np.sum(cp**2))
            l2_rel = (l2_num / l2_den) * 100
            total_num[labelt[j]] += np.sum(diff**2)
            total_den[labelt[j]] += np.sum(cp**2)

            f.write(f"Station loc={label[i]} against {labelt[j]} Cp: abs L2={l2_num:.5f}, rel L2={l2_rel:.5f} %\n")
      
        plt.close()

        #u profile
        plt.figure(figsize=(10, 8))
        u_mag_flat = np.sqrt(u_pred**2 + v_pred**2)
        plt.plot(new_bottom_coords_x, u_mag_flat, color='k', label=f"PINN")
        for k, type in enumerate(res1):
            file_path = f'data/u_{label[i]}_{type}.csv'
            my_data = np.genfromtxt(file_path, delimiter=',')
            x = my_data[:, 0]
            # norm_x = (x - x_inlet_exp) / (x_outlet_exp - x_inlet_exp)
            norm_x = x
            umag = my_data[:, 1]
            if k == 0:
                plt.scatter(norm_x, umag, label=f"{labelt1[k]}", color=colors[k], marker='o')
            elif k == 1:
                plt.plot(norm_x, umag, label=f"{labelt1[k]}", color=colors[k], linestyle="--")
            plt.xlabel("Normalized x coordinate")
            plt.ylabel(r"$\vert u \vert / U_0$")
            # plt.grid(True, linestyle=':')
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.title(rf"Dimensionless Velocity Magnitude $\vert u \vert / U_0$ at section {label[i]}")
            plt.tight_layout()
            plt.savefig(f"{folder_name}/u_profile_{label[i]}.png", dpi=150)

            x_query = norm_x * 0.4
            y_at_query = np.interp(x_query, bottom_coords_x.flatten(), bottom_coords_y.flatten()) 
            onedgrid = np.column_stack((x_query, y_at_query))
            points_t = torch.tensor(onedgrid, dtype=torch.float32)
            with torch.no_grad():
                prediction = net(points_t).cpu().numpy()
            u_pred_ = prediction[:,0]
            v_pred_ = prediction[:,1]
            u_mag_flat_ = np.sqrt(u_pred_**2 + v_pred_**2)

            diff = u_mag_flat_ - umag
            l2_num = np.sqrt(np.sum(diff**2))
            l2_den = np.sqrt(np.sum(umag**2))
            l2_rel = (l2_num / l2_den) * 100
            total_num_v[labelt1[k]] += np.sum(diff**2)
            total_den_v[labelt1[k]] += np.sum(umag**2)

            f.write(f"Station loc={label[i]} against {labelt1[k]} U: abs L2={l2_num:.5f}, rel L2={l2_rel:.5f}\n")
        
        plt.close()

    overall_rel_l2_bruns = (np.sqrt(total_num["Bruns et al."]) / np.sqrt(total_den["Bruns et al."])) * 100 
    overall_rel_l2_garbaruk = (np.sqrt(total_num["Garbaruk et al."]) / np.sqrt(total_den["Garbaruk et al."])) * 100
    overall_rel_l2_cnn = (np.sqrt(total_num["Bokil et al. (RANS-CNN)"]) / np.sqrt(total_den["Bokil et al. (RANS-CNN)"])) * 100
    f.write(f"Overall relative L2 Cp (Bruns et al.): {overall_rel_l2_bruns:.5f} %\n")
    f.write(f"Overall relative L2 Cp (Garbaruk et al.): {overall_rel_l2_garbaruk:.5f} %\n")
    f.write(f"Overall relative L2 Cp (Bokil et al.): {overall_rel_l2_cnn:.5f} %\n")

    overall_rel_l2_bruns = (np.sqrt(total_num_v["Bruns et al."]) / np.sqrt(total_den_v["Bruns et al."])) * 100    
    overall_rel_l2_cnn = (np.sqrt(total_num_v["Bokil et al. (RANS-CNN)"]) / np.sqrt(total_den_v["Bokil et al. (RANS-CNN)"])) * 100
    f.write(f"Overall relative L2 |u|/U0 (Bruns et al.): {overall_rel_l2_bruns:.5f}\n")
    f.write(f"Overall relative L2 |u|/U0 (Bokil et al.): {overall_rel_l2_cnn:.5f}\n")

    f.close()

    # #############################################################
    #3#Loss Curve
    plt.figure(figsize=(10, 6))
    keys = ["_step"] + [f"train/{lbl}" for lbl in labels_keys]  
    # keys = [_step, xmom, ymom, etc..]
    history = run.history(keys=keys, samples=10000)  
    adam_end = itr_1
    plt.figure(figsize=(10, 6))  
    plt.gca().set_prop_cycle(color=plt.cm.tab20(np.linspace(0, 1, 14)))  
    for lbl in labels_keys:
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
    keys = ["_step"] + [f"test/{lbl}" for lbl in labels_keys]  
    history = run.history(keys=keys, samples=10000)  
    adam_end = itr_1
    plt.figure(figsize=(10, 6))  
    plt.gca().set_prop_cycle(color=plt.cm.tab20(np.linspace(0, 1, 14)))  
    for lbl in labels_keys:
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

