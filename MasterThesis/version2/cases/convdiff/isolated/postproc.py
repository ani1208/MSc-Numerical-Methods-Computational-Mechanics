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


# this func exploits symmetry, it treats the field as centered around y=1 
def input_transform(x):
    x_coord = x[:, 0:1] # 0.2
    y_coord = x[:, 1:2] # 0.5 and 1.5
    y_symmetric = torch.square(y_coord - 1.0) # Both become 0.25!
    return torch.concat([x_coord, y_symmetric], axis=1)

def output_transform(x, y):
    x_coord = x[:, 0:1]
    raw_NN_output = y
    
    # Smooth distance function:
    return x_coord + x_coord * (1.0 - x_coord) * raw_NN_output

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
Re = data["training"]["Re"]
Pr = data["training"]["Pr"]

if fourier_enabled:
    # change the amplitude to 0.5 or 1 if required in fourier lift
    B = torch.load(f"runs/{base_dir}/fourier_B.pt")

#Geometry Definition
geom = dde.geometry.Rectangle([0, 0], [L, W])

net = torch.load(f"runs/{base_dir}/net_finish.pt", weights_only=False) 
net.eval()

with open(os.path.join(f"runs/{base_dir}", "wandb_run_info.json")) as f:      
    run_info = json.load(f)
API = wandb.Api()  
run = API.run(f"{run_info['wandb_project']}/{run_info['wandb_run_id']}")
#################RESTORE###############################################################################################################################################
# #Save Plots for Analytical vs PINNs Solutions
# 1 #PINNs solution
x1 = np.linspace(0, L, 21)
y1 = np.linspace(0, W, 41)
xx, yy = np.meshgrid(x1, y1, indexing='ij')
X = np.vstack((xx.ravel(), yy.ravel())).T
y_pred = model.predict(X)
y_pred_reshaped = y_pred.reshape(len(x1), len(y1))
plt.figure(figsize=(20, 8))
plt.subplot(1, 3, 1)
ctr = plt.contourf(xx, yy, y_pred_reshaped, levels=50, cmap='Spectral_r')
cbar = plt.colorbar(ctr, label="u PINN", ax=plt.gca(), format=ticker.ScalarFormatter(useMathText=True))
cbar.formatter.set_powerlimits((0, 0))
plt.xlabel('x')
plt.ylabel('y')
plt.title("PINNs Solution for 2D Convection-Diffusion Equation")
# plt.savefig(f"{folder_name}/pinns.png", dpi=150, bbox_inches="tight")

plt.subplot(1, 3, 2)
x1 = np.linspace(0, 1, 21)
y1 = np.linspace(0, 2, 41)
# print(my_data)
X = my_data[:, 0:2]      
y = my_data[:, 2:3]
xx, yy = np.meshgrid(x1, y1, indexing='ij')
exact = y.reshape(len(x1), len(y1))
ctr = plt.contourf(xx, yy, exact, levels=50, cmap='Spectral_r')
cbar = plt.colorbar(ctr, label="u FEM", ax=plt.gca(), format=ticker.ScalarFormatter(useMathText=True))
cbar.formatter.set_powerlimits((0, 0))
plt.xlabel('x')
plt.ylabel('y')
plt.title("Reference Solution for 2D Convection-Diffusion Equation")
# plt.savefig("ReferenceRe.png", dpi=150, bbox_inches="tight")

plt.subplot(1, 3, 3)
my_data = genfromtxt(f'Re{Re}Pr{Pr}.csv', delimiter=',')
x = my_data[:,0]
y = my_data[:,1]
y_true = my_data[:,2]
# xx, yy = np.meshgrid(x, y, indexing='ij')
y_true_reshaped = y_true.reshape(len(x1), len(y1))
ctr = plt.contourf(xx, yy, np.abs(y_true_reshaped - y_pred_reshaped), levels=50, cmap='inferno')
cbar = plt.colorbar(ctr, label="Absolute error", ax=plt.gca(), format=ticker.ScalarFormatter(useMathText=True))
cbar.formatter.set_powerlimits((0, 0))
plt.xlabel('x')
plt.ylabel('y')
plt.title("∣ FEM - prediction ∣")
plt.savefig(f"{folder_name}/pinns.png", dpi=150, bbox_inches="tight")
###########L2 Norm in a .txt file ###########################
l2 = ( np.linalg.norm(y_true_reshaped - y_pred_reshaped) / np.linalg.norm(y_true_reshaped) ) * 100
with open(f"{folder_name}/L2error.txt", "w") as f:
    f.write(f"L2 Error : {l2:.4f}%")
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

