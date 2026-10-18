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

#Parameters for the burgers equation
L = 1
n = 1
T = 1

#Geometry Definition
geom = dde.geometry.Interval(-1, 1)
time_domain = dde.geometry.TimeDomain(0, T)
geomtime = dde.geometry.GeometryXTime(geom, time_domain)


net = torch.load(f"runs/{base_dir}/net_finish.pt", weights_only=False) 
net.eval()

with open(os.path.join(f"runs/{base_dir}", "wandb_run_info.json")) as f:      
    run_info = json.load(f)
API = wandb.Api()  
run = API.run(f"{run_info['wandb_project']}/{run_info['wandb_run_id']}")
#################RESTORE###############################################################################################################################################
#Save Plots for Analytical vs PINNs Solutions
# 1#Analytical vs PINNs solution
t = np.linspace(0, T, 100)
x = np.linspace(-1, 1, 256)
xx, tt = np.meshgrid(x,t)
X = np.vstack((xx.ravel(), tt.ravel())).T

points_t = torch.tensor(X, dtype=torch.float32)
with torch.no_grad():
    prediction = net(points_t).cpu().numpy()
y_pred = prediction
y_pred_reshaped = y_pred.reshape(len(t), len(x))

plt.figure(figsize=(16, 6))
plt.rcParams.update({'font.size': 14, 'axes.labelsize': 14, 'axes.titlesize': 14})
plt.subplot(1, 3, 1)
plt.contourf(xx, tt, exact, levels=50, cmap='Spectral_r')
plt.colorbar(label="u")
plt.xlabel('x')
plt.ylabel('t')
plt.title("Analytical Solution for 1D Burgers Equation")
# plt.savefig("analytical.png", dpi=150, bbox_inches="tight")

plt.subplot(1, 3, 2)
plt.contourf(xx, tt, y_pred_reshaped, levels=50, cmap='Spectral_r')
plt.colorbar(label="u")
plt.xlabel('x')
plt.ylabel('t')
plt.title("PINNs Solution for 1D Burgers Equation")
# plt.savefig("pinns.png", dpi=150, bbox_inches="tight")

# fig = plt.figure(figsize=(11,6))
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(tt, xx, y_pred_reshaped, cmap='Spectral_r');
# ax.view_init(35,35)
# ax.set_xlabel('$t$')
# ax.set_ylabel('$x$')
# ax.set_zlabel('$u(x,t)$', labelpad=10)
# ax.set_box_aspect(None, zoom=0.85) 
# ax.set_title('Solution of Burgers equation');
# plt.tight_layout()
# plt.savefig("pinns_3d.png", dpi=150, bbox_inches="tight")
plt.subplot(1, 3, 3)
plt.contourf(xx, tt, np.abs(y_true - y_pred_reshaped), levels=50, cmap='inferno')
plt.colorbar(label="absolute error u", format='%.1e')
plt.xlabel('x')
plt.ylabel('t')
plt.title("∣ analytical - prediction ∣")
plt.savefig("pinns.png", dpi=150, bbox_inches="tight")
#############################################################
# 2#Time slices 
time_slices = [0.0, 0.25, 0.5, 0.75, 1.0]
num_plots = len(time_slices)

# Extract the PDE loss weight dynamically (defaults to 1.0 if not explicitly set)
w_pde = model.loss_weights[0] if model.loss_weights is not None else 1.0

# Array to store the weighted summary metric for each slice
weighted_time_summary = []

# Figure with 2 columns: Left for predictions vs exact, Right for local raw PDE error
fig, axs = plt.subplots(num_plots, 2, figsize=(14, 4 * num_plots))

for i, t_val in enumerate(time_slices):
    # --- Left Column: Profile Comparison ---
    idx = np.argmin(np.abs(t - t_val))
    y_analytical = exact[idx, :]
    y_pinn = y_pred_reshaped[idx, :]

    axs[i, 0].plot(x, y_analytical, 'r-', label='Analytical Solution')
    axs[i, 0].plot(x, y_pinn, 'b--', label='PINN Prediction')
    axs[i, 0].set_autoscale_on(False)
    axs[i, 0].set_xlabel('x')
    axs[i, 0].set_ylabel('u')
    axs[i, 0].set_title(f'Solution Profile at t = {t_val}')
    axs[i, 0].legend()
    axs[i, 0].set_ylim(-1.1, 1.1)
    axs[i, 0].set_xlim(-1.1, 1.1)
    axs[i, 0].grid(True, alpha=0.3)

    # --- Right Column: Local Spatial Error (Raw operator=pde) ---
    X_slice = np.vstack((x, np.full_like(x, t_val))).T
    points_t = torch.tensor(X_slice, dtype=torch.float32, requires_grad=True)  
    prediction = net(points_t)
    residuals = pde(points_t, prediction)  
    err_eq = sum(torch.abs(r) for r in residuals)
    pde_residual_slice = err_eq.detach().cpu().numpy().flatten()
    local_pde_error = np.abs(pde_residual_slice)
    
    axs[i, 1].plot(x, local_pde_error, 'g-', label='Local PDE Residual (Raw)')
    axs[i, 1].set_yscale('log')  # Log scale to easily see small structural variations
    axs[i, 1].set_xlabel('x')
    axs[i, 1].set_ylabel('Absolute Residual')
    axs[i, 1].set_title(f'Local Physics Error at t = {t_val}')
    axs[i, 1].legend()
    axs[i, 1].grid(True, which="both", alpha=0.3)
    
    # Compute the unweighted Mean Squared Error for this slice, then apply the train weight
    pde_mse_slice = np.mean(np.square(pde_residual_slice))
    weighted_pde_loss = w_pde * pde_mse_slice
    
    # If checking the initial condition slice (t=0), add the weighted IC loss component
    if np.isclose(t_val, 0.0):
        w_ic = model.loss_weights[1] if model.loss_weights is not None else 1.0
        points_t = torch.tensor(X_slice, dtype=torch.float32)
        with torch.no_grad():
            prediction = net(points_t).cpu().numpy()
        y_pred_t0 = prediction
        y_true_t0 = np.sin(n * np.pi * x / L)
        ic_mse_slice = np.mean(np.square(y_pred_t0.flatten() - y_true_t0.flatten()))
        weighted_pde_loss += (w_ic * ic_mse_slice)

    weighted_time_summary.append(weighted_pde_loss)

plt.tight_layout(pad=2.0)
plt.savefig("Time_Slices_Comparison.png", dpi=150, bbox_inches="tight")
print("Saved: Time_Slices_Comparison.png")

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

