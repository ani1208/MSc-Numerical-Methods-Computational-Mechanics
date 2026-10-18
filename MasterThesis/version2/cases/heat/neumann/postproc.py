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

def output_transform(x, y):
    # x[:, 0:1] is x-coordinate
    # x[:, 1:2] is y-coordinate
    
    # This term is zero if x=0 or x=L
    x_boundary = x[:, 0:1] * (L - x[:, 0:1])
    
    # This term is zero if y=0 or y=W
    y_boundary = x[:, 1:2] * (W - x[:, 1:2])
    
    # Combine them. Since it's a product, if ANY edge is hit, the result is 0.
    return x_boundary * y_boundary * y
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

if fourier_enabled:
    # change the amplitude to 0.5 or 1 if required in fourier lift
    B = torch.load(f"runs/{base_dir}/fourier_B.pt")

#Parameters for the heat equation
a = 0.4
L = 1
W = 1
n = m = 1
T = 1

#Geometry Definition
geom = dde.geometry.Rectangle([0, 0], [L, W])
time_domain = dde.geometry.TimeDomain(0, T)
geomtime = dde.geometry.GeometryXTime(geom, time_domain)

net = torch.load(f"runs/{base_dir}/net_finish.pt", weights_only=False) 
net.eval()

with open(os.path.join(f"runs/{base_dir}", "wandb_run_info.json")) as f:      
    run_info = json.load(f)
API = wandb.Api()  
run = API.run(f"{run_info['wandb_project']}/{run_info['wandb_run_id']}")
#################RESTORE###############################################################################################################################################
#############################################################
# #Save Plots for Analytical vs PINNs Solutions
# 1 #PINNs solution
#############################################################
# 1. Corrected 3D Surface Time Slices Plotting
#############################################################
t_steps = np.linspace(0, T, 100)
x_steps = np.linspace(0, L, 100)
y_steps = np.linspace(0, W, 100)

# Create distinct meshgrids for spatial evaluation and time tracking
xx_space, yy_space = np.meshgrid(x_steps, y_steps, indexing='ij')

time_slices = [0.000, 0.250, 0.500, 0.750, 1.000]
# num_plots = len(time_slices)
# cols = 2
# rows = (num_plots + cols - 1) // cols

# fig = plt.figure(figsize=(15, 5 * rows))
# plt.rcParams.update({'font.size': 12, 'axes.labelsize': 12, 'axes.titlesize': 12})
# axs = fig.subplots(rows, cols, subplot_kw={'projection': '3d'}).flatten()

# Pre-calculate global min/max bounds for consistent Z-axis scaling across plots
u_global_predictions = []
for t_val in time_slices:
    X_eval = np.vstack((xx_space.ravel(), yy_space.ravel(), np.full(xx_space.size, t_val))).T
    points_t = torch.tensor(X_eval, dtype=torch.float32)
    with torch.no_grad():
        prediction = net(points_t).cpu().numpy()
    u_global_predictions.append(prediction.reshape(xx_space.shape))

u_min_global = min(u.min() for u in u_global_predictions)
u_max_global = max(u.max() for u in u_global_predictions)

for i, t_val in enumerate(time_slices):
    u_slice = u_global_predictions[i]
    # u_min_local = u_slice.min()
    u_min_local = 0.0
    u_max_local = u_slice.max()
    # surf = axs[i].plot_surface(xx_space, yy_space, u_slice, cmap='Spectral_r', 
    #                            alpha=0.8, antialiased=True, vmin=u_min_local, vmax=u_max_local)

    # axs[i].set_xlim(0, L)
    # axs[i].set_ylim(0, W)
    # axs[i].set_zlim(u_min_global, u_max_global)
    # axs[i].zaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # axs[i].set_xlabel('x')
    # axs[i].set_ylabel('y')
    # axs[i].set_zlabel('Temperature (u)')
    # axs[i].set_title(f'3D Surface at t = {t_val:.2f}')
    # fig.colorbar(surf, ax=axs[i], shrink=0.5, aspect=5, format='%.2f')

    fig, ax = plt.subplots(figsize=(8, 6))

    surf = ax.contourf(xx_space, yy_space, u_slice, cmap="Spectral_r", levels=25,vmin=u_min_local, vmax=u_max_local)

    ax.set_aspect("equal")
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_title(f"Time: {t_val:.3f} s - PINN solution")
    cbar = fig.colorbar(surf, ax=ax, shrink=0.7)
    cbar.formatter = ticker.ScalarFormatter(useMathText=True)
    cbar.formatter.set_powerlimits((0, 0)) 
    cbar.update_ticks()
    plt.tight_layout()
    plt.savefig(f"step_pinn_{t_val:.3f}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    my_data = genfromtxt(f'dirichlet/U_vals_{t_val:.3f}.csv', delimiter=',', skip_header=1)
    x = my_data[:, 0:1]      
    y = my_data[:, 1:2]
    u = my_data[:, 2:3]
    u_true_reshaped = u.reshape(len(y_steps), len(x_steps)).T
    surf = ax.contourf(xx_space, yy_space, np.abs(u_true_reshaped - u_slice), cmap="inferno", levels=50)

    ax.set_aspect("equal")
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_title(f"Time: {t_val:.3f} s - Absolute Error")
    cbar = fig.colorbar(surf, ax=ax, shrink=0.7)
    cbar.formatter = ticker.ScalarFormatter(useMathText=True)
    cbar.formatter.set_powerlimits((0, 0))  
    cbar.update_ticks()
    plt.tight_layout()
    plt.savefig(f"step_absolute_error_{t_val:.3f}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    with open(f"U_vals_{t_val:.3f}.csv", "w") as f:
        f.write("x,y,u\n")
        x_flat = xx_space.ravel(order='F')
        y_flat = yy_space.ravel(order='F')
        u_flat = u_slice.ravel(order='F')

        for x_v, y_v, u_v in zip(x_flat, y_flat, u_flat):
            f.write(f"{x_v:.6f},{y_v:.6f},{u_v:.6f}\n")

    ###########L2 Norm in a .txt file ###########################
    l2 = (np.linalg.norm(u_true_reshaped - u_slice) / np.linalg.norm(u_true_reshaped)) * 100
    with open(f"L2error_{t_val:.3f}.txt", "w") as f:
        f.write(f"L2 Error: {l2:.4f}%\n")
    #############################################################  

#############################################################
##############################################################
# 2 #Residuals
# X_train = data.train_x # From data object
# tol = 1e-6
# # 1. Domain points: not on boundary and not on initial time
# is_initial = np.isclose(X_train[:, 2], 0.0, atol=tol)  # t ≈ 0
# is_boundary = (
#     np.isclose(X_train[:, 0], 0.0, atol=tol) |  # x ≈ 0
#     np.isclose(X_train[:, 0], L, atol=tol) |    # x ≈ L
#     np.isclose(X_train[:, 1], 0.0, atol=tol) |  # y ≈ 0
#     np.isclose(X_train[:, 1], W, atol=tol)      # y ≈ W
# )
# is_domain = ~(is_initial | is_boundary)

# #extract
# X_domain = X_train[is_domain]           # Interior domain points
# X_boundary = X_train[is_boundary]       # Boundary points
# X_initial = X_train[is_initial]         # Initial condition points

# X_all = np.vstack((X_domain, X_boundary, X_initial))
# f = model.predict(X_all, operator=pde)
# t_vals = X_all[:, 2]  # time
# residual_magnitude = np.abs(f).flatten()

# plt.figure(figsize=(10, 6))
# plt.scatter(t_vals, residual_magnitude, s=1, alpha=0.6)
# plt.xlabel('Time t')
# plt.ylabel('|PDE Residual|')
# plt.title('PDE Residual Magnitude Across Training Domain')
# plt.grid(True, alpha=0.3)
# plt.savefig("pde_residuals.png", dpi=150, bbox_inches="tight")
# 2. Custom Learning Rate Decay Plot
plt.figure(figsize=(6, 4))
plt.semilogy(lr_tracker.steps, lr_tracker.lr_history, color='darkorange', linewidth=2)
plt.title("Learning Rate Decay", fontsize=12)
plt.xlabel("Iterations")
plt.ylabel("Learning Rate")
plt.grid(True, linestyle=":", alpha=0.6)
plt.savefig("learningrate.png", dpi=150, bbox_inches="tight")
#############################################################

#############################################################
# 2. Corrected Local Snapshots: 2D Spatial Physics Error (x, y, error)
#############################################################
w_pde = model.loss_weights[0] if model.loss_weights is not None else 1.0
weighted_time_summary = []

# Subplots for 2D Error Contours across time slices
# fig, axs = plt.subplots(num_plots, 1, figsize=(8, 4 * num_plots))

for i, t_val in enumerate(time_slices):
    fig, ax = plt.subplots(figsize=(8, 6))
    X_slice = np.vstack((xx_space.ravel(), yy_space.ravel(), np.full(xx_space.size, t_val))).T
    points_t = torch.tensor(X_slice, dtype=torch.float32, requires_grad=True)  
    prediction = net(points_t)
    residuals = pde(points_t, prediction)  
    err_eq = sum(torch.abs(r) for r in residuals)  
    pde_residual_slice = err_eq.detach().cpu().numpy().flatten()
    local_pde_error_grid = np.abs(pde_residual_slice).reshape(xx_space.shape)
    
    contour = ax.contourf(xx_space, yy_space, local_pde_error_grid, 
                               levels=50, cmap='inferno')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Local Physics Error at t = {t_val:.2f}')
    
    # Create colorbar matching your requested naming convention label
    cbar = fig.colorbar(contour, ax=ax, format=ticker.ScalarFormatter(useMathText=True))
    cbar.formatter.set_powerlimits((0, 0))
    cbar.set_label('Local PDE Residual')
    cbar.update_ticks()
    # Compute metrics for bar summary tracking chart
    pde_mse_slice = np.mean(np.square(pde_residual_slice))
    weighted_pde_loss = w_pde * pde_mse_slice
    
    if np.isclose(t_val, 0.0):
        w_ic = model.loss_weights[1] if model.loss_weights is not None else 1.0
        points_t = torch.tensor(X_slice, dtype=torch.float32)
        with torch.no_grad():
            prediction = net(points_t).cpu().numpy()
        y_pred_t0 = prediction.flatten()
        y_true_t0 = (np.sin(n * np.pi * X_slice[:, 0] / L) * np.sin(m * np.pi * X_slice[:, 1] / W))
        ic_mse_slice = np.mean(np.square(y_pred_t0 - y_true_t0))
        weighted_pde_loss += (w_ic * ic_mse_slice)

    weighted_time_summary.append(weighted_pde_loss)

    plt.tight_layout(pad=2.0)
    plt.savefig(f"Time_Slices_Comparison_{t_val:.3f}.png", dpi=150, bbox_inches="tight")

#############################################################
#. TIME SUMMARY: Weighted Loss Across Timestamps
#############################################################
plt.figure(figsize=(8, 5))

# Plotting the total weighted landscape contribution per slice
bars = plt.bar([str(t) for t in time_slices], weighted_time_summary, color='purple', alpha=0.7, edgecolor='k')

# Add values on top of the bars for crisp documentation metrics
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, height * 1.2, f'{height:.2e}', ha='center', va='bottom', fontsize=9)

plt.yscale('log')  # Crucial because t=0 (trained region) will be significantly lower than t > 0
plt.xlabel("Time Slice ($t$)")
plt.ylabel("Weighted Physics Loss ($w \cdot L$)")
plt.title("Weighted Loss Over Time Slices")
plt.grid(True, which="both", axis='y', ls="--", alpha=0.3)
plt.tight_layout()

plt.savefig("Time_Summary_Weighted_Loss.png", dpi=150, bbox_inches="tight")
print("Saved: Time_Summary_Weighted_Loss.png")
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

