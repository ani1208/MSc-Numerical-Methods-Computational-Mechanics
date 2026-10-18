######################################################################
import os
import json
import glob
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
warnings.filterwarnings("ignore", category=DeprecationWarning)
import wandb
######################################################################
from config import get_base_config  
config = get_base_config()
######################################################################
# set seed 
SEED = config.training.seed
np.random.seed(SEED)
torch.manual_seed(SEED)
dde.config.set_random_seed(SEED)

###########################################################################
# GradNorm scheme - Wang et al.
class GradNormCallback(dde.callbacks.Callback):
    def __init__(self, period=1000, momentum=0.9, n_pde_terms=3, lr=1e-4):
        super().__init__()
        self.period = period
        self.momentum = momentum
        self.n_pde_terms = n_pde_terms  
        self.lr = lr

    def on_epoch_end(self):
        global loss_weights

        iteration = self.model.train_state.epoch
        if iteration % self.period != 0 or iteration == 0:
            return

        net = self.model.net
        x_batch = self.model.data.train_x_all
        x_t = torch.tensor(x_batch, dtype=torch.float32, requires_grad=True)

        grad_norms = []

        for i in range(self.n_pde_terms):
            net.zero_grad()

            y_t = net(x_t)
            residuals = navier_stokes_with_SA(x_t, y_t)  
            R_i = residuals[i]
            L_i = torch.mean(R_i ** 2)                     
            L_i.backward(retain_graph=False)

            total_norm_sq = sum(p.grad.norm()**2 for p in net.parameters() if p.grad is not None)
            grad_norms.append(torch.sqrt(total_norm_sq).item())

        bc_list = [inlet_x, inlet_y, outlet_p, wallt_x, wallt_y, wallb_x, wallb_y, wallsh_x, wallsh_y, wallsv_x, wallsv_y]

        for bc in bc_list:
            net.zero_grad()
            try:
                bc_pts = bc.filter(geom.random_boundary_points(1000))
            except IndexError:
                bc_pts = np.empty((0, 2))
            if bc_pts.shape[0] == 0:
                grad_norms.append(1e-8)
                continue
            # print(f"  {bc} -> {bc_pts.shape[0]} points")
            bc_pts_t = torch.tensor(bc_pts, dtype=torch.float32, requires_grad=True)
            y_bc = net(bc_pts_t)
            err = bc.error(bc_pts, bc_pts_t, y_bc, 0, bc_pts.shape[0])
            L_bc = torch.mean(err ** 2)
            L_bc.backward(retain_graph=False)
            total_norm_sq = sum(p.grad.norm()**2 for p in net.parameters() if p.grad is not None)
            grad_norms.append(torch.sqrt(total_norm_sq).item())

        net.zero_grad()
        sum_norms = sum(grad_norms) + 1e-8
        lambda_hat = [sum_norms / (g + 1e-8) for g in grad_norms]
        #maybe this clip can be removed if needed
        #adding explicit scaling for nu transport equation - can be used to scale mass conservation equation too as gaurav suggested it can alter the reattachment length
        lambda_hat = [min(l, loss_clip) for l in lambda_hat]
        
        if loss_explicit_enabled:
            for i in loss_indices:
                lambda_hat[i] *= loss_explicit_weight_scale

        for i in range(len(grad_norms)):
            self.model.loss_weights[i] = self.momentum * self.model.loss_weights[i] +  (1 - self.momentum) * lambda_hat[i]

        print(f"[iter {iteration}] grad_norms={grad_norms}  new_lambdas={model.loss_weights}")

###########################################################################

###########################################################################
class LossPlotCallback(dde.callbacks.Callback):
    def __init__(self, period=3000, labels=None):
        super().__init__()
        self.period = period
        self.labels = labels
 
    def on_epoch_end(self):
        epoch = self.model.train_state.epoch
        if epoch % self.period == 0 and epoch > 0:
            plot_loss_curves(self.model.losshistory, epoch, self.labels)
 
 
def plot_loss_curves(losshistory, epoch, labels):
    train_loss = np.array(losshistory.loss_train)
    test_loss = np.array(losshistory.loss_test)
    steps = np.array(losshistory.steps)
 
    # Train
    plt.figure(figsize=(10, 6))
    plt.gca().set_prop_cycle(color=plt.cm.tab20(np.linspace(0, 1, 14)))
    for i in range(train_loss.shape[1]):
        ref_loss = train_loss[0, i]
        normalized_loss = train_loss[:, i] / ref_loss
        plt.semilogy(steps, normalized_loss + 1e-10, label=labels[i])
    plt.xlabel("Iterations")
    plt.ylabel("Normalized Loss ($L / L_{initial}$)")
    plt.title(f"Training Loss (epoch {epoch})")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{folder_name}/training_loss_epoch{epoch}.png", dpi=150, bbox_inches="tight")
    plt.close()
###########################################################################

###########################################################################
# Log to Wandb
class WandbLossCallback(dde.callbacks.Callback):
    def __init__(self, labels, period=10):
        super().__init__()
        self.labels = labels
        self.period = period

    def on_epoch_end(self):
        epoch = self.model.train_state.epoch
        if epoch % self.period != 0:
            return
        train_loss = self.model.train_state.loss_train
        test_loss = self.model.train_state.loss_test
        log_dict = {f"train/{lbl}": v for lbl, v in zip(self.labels, train_loss)}
        log_dict.update({f"test/{lbl}": v for lbl, v in zip(self.labels, test_loss)})
        log_dict["Re"] = Re
        wandb.log(log_dict, step=epoch)
###########################################################################

###########################################################################
# Copy previous iteration NN for evaluation later at current iteration
class UpdatePrevNet(dde.callbacks.Callback):
    def __init__(self, period=10):
        super().__init__()
        self.period = period
 
    def on_epoch_end(self):
        global prev_net
        epoch = self.model.train_state.epoch
        if epoch % self.period == 0 and epoch > 0:
            prev_net = copy.deepcopy(self.model.net)

###########################################################################

###########################################################################
# Plot contours at checkpoints
class PlotCallback(dde.callbacks.Callback):
    def __init__(self, period=100):
        super().__init__()
        self.period = period

    def on_epoch_end(self):
        epoch = self.model.train_state.epoch
        # self.on_epoch_end runs at every iteration step
        if epoch % self.period == 0:
            contour_plotting_at_checkpoints(self.model, epoch)

def contour_plotting_at_checkpoints(model, epoch, theta_plot=45.0):
    Re = 183.3
    x_dense = np.linspace(0.0, 30.0, 500)
    y_dense = np.linspace(0.0, 2.061, 200)
    X, Y = np.meshgrid(x_dense, y_dense)
    points = np.vstack((X.ravel(), Y.ravel())).T
    # Re_column = np.full((points.shape[0], 1), float(Re))
    # X_3D = np.hstack((points, Re_column))
    prediction = model.predict(points)
    if isinstance(prediction, tuple) or isinstance(prediction, list):
        prediction = prediction[0]
    u_component = prediction[:, 0]
    v_component = prediction[:, 1]
    p = prediction[:,2]
    u_mag_flat = np.sqrt(u_component**2 + v_component**2)
    U_mag = u_mag_flat.reshape(X.shape)
    U_mag[(X < 2.0) & (Y < 1.0)] = np.nan
    plt.figure(figsize=(10, 5), dpi=600)
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
    plt.savefig(f"{folder_name}/iteration_{epoch}_comparison_Re{Re}.png", bbox_inches='tight', dpi=600)
    plt.close()

###########################################################################

###########################################################################
# Plot residual contours at checkpoints
class ResidualPlotCallback(dde.callbacks.Callback):
    def __init__(self, period=100):
        super().__init__()
        self.period = period

    def on_epoch_end(self):
        epoch = self.model.train_state.epoch
        # self.on_epoch_end runs at every iteration step
        if epoch % self.period == 0:
            residual_plotting_at_checkpoints(self.model, epoch)

def residual_plotting_at_checkpoints(model, epoch):
    Re = 183.3
    x_dense = np.linspace(0.0, 30.0, 500)
    y_dense = np.linspace(0.0, 2.061, 200)
    X, Y = np.meshgrid(x_dense, y_dense)
    points_r = np.vstack((X.ravel(), Y.ravel())).T
    inside_mask_r = geom.inside(points_r)
    residuals = model.predict(points_r, operator=navier_stokes_with_SA)
    labels = ["continuity", "x-mom", "y-mom"]
    for i, label in enumerate(labels):
        res = residuals[i].flatten()
        res_abs = np.abs(res)
        res_abs[~inside_mask_r] = np.nan
        res_grid = res_abs.reshape(xxr.shape)

        plt.figure(figsize=(10, 5), dpi=300)
        contour = plt.contourf(xxr, yyr, res_grid, levels=100, cmap='inferno', antialiased=True)
        cbar = plt.colorbar(contour, orientation='horizontal', pad=0.15, shrink=0.3)
        cbar.ax.tick_params(labelsize=5)
        cbar.set_label(f'|{label} residual|', fontsize=8)
        plt.xlim(0.0, 1.9)
        plt.ylim(0.04, 0.71)
        plt.gca().set_aspect('equal')
        plt.title(f"{label} residual (epoch {epoch})", fontsize=9)
        plt.savefig(f"{folder_name}/residual_{label}_epoch{epoch}.png", bbox_inches='tight', dpi=300)
        plt.close()


###########################################################################

###########################################################################
# Re ramp for curriculum training 
# Wang et al. , Aditi et al.
class ReRampCallback(dde.callbacks.Callback):
    def __init__(self, re_schedule):
        super().__init__()
        self.re_schedule = re_schedule

    def on_epoch_end(self):
        global Re
        epoch = self.model.train_state.epoch
        for start_epoch, re_value in self.re_schedule:
            if epoch >= start_epoch:
                Re = re_value
###########################################################################

###############################################################################
# Boundary definitions
def wall_top_boundary(x, on_boundary):
    """Checks for points on top wall boundary"""
    return on_boundary and np.isclose(x[1], 2.061)


def wall_bottom_boundary(x, on_boundary):
    """Checks for points on bottom wall boundary"""
    return on_boundary and np.isclose(x[1], 0.0)


def wall_mid_horizontal_boundary(x, on_boundary):
    """Check for points on step horizontal boundary"""
    return on_boundary and (np.isclose(x[1], 1) and x[0] < 2.0)


def wall_mid_vertical_boundary(x, on_boundary):
    """Check for points on step horizontal boundary"""
    return on_boundary and (x[1] < 1 and np.isclose(x[0], 2.0))


def outlet_boundary(x, on_boundary):
    """Implements the outlet boundary with zero y-velocity component"""
    return on_boundary and np.isclose(x[0], 30.0)


def inlet_boundary(x, on_boundary):
    """Implements the inlet boundary with parabolic x-velocity component"""
    return on_boundary and np.isclose(x[0], 0.0)

def parabolic_velocity(x):
    #changed velocity profile acc. to CFD training
    y1, y2 = 1.0, 2.061
    H = y2-y1
    # Umax = 0.879542893
    # C = 4 * Umax / H**2  #gives mean velocity 0.58 - wrong
    # try 3.553 to make Umax 1 and check
    C = 6 / H**2 #gives mean velocity 1 
    return (C * (x[:, 1:2] - 1) * (2.061 - x[:, 1:2])).reshape(-1, 1)


def zero_velocity(x):
    """Zero velocity"""
    return np.zeros((x.shape[0], 1))

###############################################################################
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

def navier_stokes_with_SA(x, y):

    x_coord, y_coord = x[:, 0:1], x[:, 1:2]
    u, v, p = y[:, 0:1], y[:, 1:2], y[:, 2:3]

    ########pseudo time stepping##################
    time_step = 0.1
    if pseudo_time_enabled:
        with torch.no_grad():
            prev_predicition = prev_net(x) 
        u_prev = prev_predicition[:,0:1]
        v_prev = prev_predicition[:,1:2]
        p_prev = prev_predicition[:,2:3]
        term1 = (u - u_prev)/time_step
        term2 = (v - v_prev)/time_step
        term3 = (p - p_prev)/time_step
    else: 
        term1 = torch.zeros_like(u)
        term2 = torch.zeros_like(v)
        term3 = torch.zeros_like(p)
    ##############################################

    u_x = dde.grad.jacobian(y, x, i=0, j=0) 
    u_y = dde.grad.jacobian(y, x, i=0, j=1) 
    v_x = dde.grad.jacobian(y, x, i=1, j=0) 
    v_y = dde.grad.jacobian(y, x, i=1, j=1) 
    p_x = dde.grad.jacobian(y, x, i=2, j=0)
    p_y = dde.grad.jacobian(y, x, i=2, j=1)  
    u_xx = dde.grad.hessian(y, x, component=0, i=0, j=0) 
    u_yy = dde.grad.hessian(y, x, component=0, i=1, j=1) 
    v_xx = dde.grad.hessian(y, x, component=1, i=0, j=0) 
    v_yy = dde.grad.hessian(y, x, component=1, i=1, j=1)

    continuity = term3 + u_x + v_y 
    x_momentum = term1 + u * u_x + v * u_y + p_x - (1.0 / Re) * (u_xx + u_yy)
    y_momentum = term2 + u * v_x + v * v_y + p_y - (1.0 / Re) * (v_xx + v_yy)
    continuity = continuity 
    x_momentum = x_momentum 
    y_momentum = y_momentum

    return [continuity, x_momentum, y_momentum]
    

#######################################################################################

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


############################################################################################
# Parse data from config

# W&B
wandb_project = config.wandb.project
wandb_name = config.wandb.name
wandb_tag = config.wandb.tag
case_folder = config.wandb.folder

# Arch
num_neurons = config.arch.hidden_dim
num_layers = config.arch.num_layers
num_out_dim = config.arch.out_dim
num_in_dim = config.arch.input_dim
activation = config.arch.activation
fourier_enabled = config.arch.fourier_enabled
fourier_amp = config.arch.fourier_emb.embed_scale
fourier_dim = config.arch.fourier_emb.embed_dim

# Optim
optimizer_1 = config.optim.optimizer_1
optimizer_2 = config.optim.optimizer_2
use_lbfgs = config.optim.use_lbfgs
lr_schedule = config.optim.lr_schedule
learning_rate = config.optim.learning_rate
decay_rate = config.optim.decay_rate
decay_steps = config.optim.decay_steps
separate_enabled = config.optim.separate_enabled
optim_schedule = config.optim.schedule

# Training
SEED = config.training.seed
itr_1 = config.training.max_steps_optimizer_1
itr_2 = config.training.max_steps_optimizer_2
sampling = config.training.sampling
resample_every = config.training.sampling_schedule.every
anchor_incl = config.training.anchors
domain_pts = config.training.sampling_points_domain
bc_pts = config.training.sampling_points_boundary
bc_enforcement = config.training.exact_bc_enforcement
rar_enabled = config.training.rar_enabled
rar_points = config.training.rar_points

# Loss weighting
loss_strategy_enabled = config.loss_weighting.gradnorm
loss_weights_dict = config.loss_weighting.loss_weights
loss_update_start = config.loss_weighting.update_schedule.start
loss_update_every = config.loss_weighting.update_schedule.every
loss_momentum = config.loss_weighting.momentum
loss_clip = config.loss_weighting.clip
loss_explicit_enabled = config.loss_weighting.explicit_enabled
loss_explicit_weight_scale = config.loss_weighting.explicit_scale
loss_indices = config.loss_weighting.explicit_indices

# Pseudo-time
pseudo_time_enabled = config.pseudo_time.enabled
pseudo_time_strategy = config.pseudo_time.strategy
pts_tau = config.pseudo_time.pts_tau
pts_momentum = config.pseudo_time.momentum
prev_net_update_every = config.pseudo_time.prev_net_update_every

# Re ramp
re_ramp_enabled = config.re_ramp.enabled
re_ramp_schedule = config.re_ramp.schedule

# Logging
log_every_steps = config.logging.log_every_steps
log_errors = config.logging.log_errors
log_lr = config.logging.log_lr
log_losses = config.logging.log_losses
log_raw_losses = config.logging.log_raw_losses
log_loss_weights = config.logging.log_loss_weights
log_pts_weights = config.logging.log_pts_weights

# Saving
ckpt_path = config.saving.ckpt_path
save_every_steps = config.saving.save_every_steps
num_keep_ckpts = config.saving.num_keep_ckpts

# Case variables 
Re = 183.3
_print_counter = {"n": 0}
############################################################################################

run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
folder_name = f"runs/{run_time}"
os.makedirs(folder_name, exist_ok=True)

if fourier_enabled:
    # change the amplitude to 0.5 or 1 if required in fourier lift
    B = torch.randn(fourier_dim, 2) * fourier_amp
    torch.save(B, f"{folder_name}/fourier_B.pt")
    
fh = open(f"{folder_name}/values.txt", 'a')

wandb.init(
    project=config.wandb.project,
    name=config.wandb.name,
    tags=[config.wandb.tag] if config.wandb.tag else None,
    config=config.to_dict(),
)
run_info = {"wandb_run_id": wandb.run.id, "wandb_project": config.wandb.project}
with open(os.path.join(folder_name, "wandb_run_info.json"), "w") as f:   
    json.dump(run_info, f, indent=2)
######################################################################################################

geom = dde.geometry.Polygon([
        [0.0, 2.061], [30.0, 2.061], [30.0, 0.0], [2.0, 0.0], [2.0, 1.0],
        [0.0, 1.0]
])

inlet_x = dde.icbc.DirichletBC(geom, parabolic_velocity, inlet_boundary, component=0)
inlet_y = dde.icbc.DirichletBC(geom, zero_velocity, inlet_boundary, component=1)
# outlet_u_x = dde.icbc.NeumannBC (geom, zero_velocity, outlet_boundary, component=0)
# outlet_v_x = dde.icbc.NeumannBC (geom, zero_velocity, outlet_boundary, component=1)
outlet_p = dde.icbc.DirichletBC(geom, zero_velocity, outlet_boundary, component=2)
wallt_x = dde.icbc.DirichletBC(geom, zero_velocity, wall_top_boundary, component=0)
wallt_y = dde.icbc.DirichletBC(geom, zero_velocity, wall_top_boundary, component=1)
wallb_x = dde.icbc.DirichletBC(geom, zero_velocity, wall_bottom_boundary, component=0)
wallb_y = dde.icbc.DirichletBC(geom, zero_velocity, wall_bottom_boundary, component=1)
wallsh_x = dde.icbc.DirichletBC(geom, zero_velocity, wall_mid_horizontal_boundary, component=0)
wallsh_y = dde.icbc.DirichletBC(geom, zero_velocity, wall_mid_horizontal_boundary, component=1)
wallsv_x = dde.icbc.DirichletBC(geom, zero_velocity, wall_mid_vertical_boundary, component=0)
wallsv_y = dde.icbc.DirichletBC(geom, zero_velocity, wall_mid_vertical_boundary, component=1)


if bc_enforcement:
    bc = [inlet_x, inlet_y, outlet_p]
else:
    bc = [inlet_x, inlet_y, outlet_p, wallt_x, wallt_y, wallb_x, wallb_y, wallsh_x, wallsh_y, wallsv_x, wallsv_y]
    
if anchor_incl:
    #add more points near step  
    x_d = np.linspace(2.0, 30.0, 200)
    y_d = np.linspace(0.0, 1.0, 100)
    X, Y = np.meshgrid(x_d, y_d)
    points = np.vstack((X.ravel(), Y.ravel())).T
else:
    points = None

data = dde.data.PDE(
    geom,
    navier_stokes,
    bc,
    num_domain=domain_pts,
    num_boundary=bc_pts,
    num_test=domain_pts,
    anchors=points
)

if fourier_enabled:
    layer_size = [fourier_dim*num_in_dim] + [num_neurons] * num_layers + [num_out_dim]
else:
    layer_size = [num_in_dim] + [num_neurons] * num_layers + [num_out_dim]

net = dde.nn.FNN(layer_size, activation, "Glorot uniform")
net.apply_feature_transform(input_transform)
net.apply_output_transform(output_transform)

labels = list(loss_weights_dict.keys())
loss_weights = list(loss_weights_dict.values())

model = dde.Model(data, net)
plot_contour = PlotCallback(period=1000)
# plot_residual = ResidualPlotCallback(period=10000)
plot_loss = LossPlotCallback(period=5000, labels=labels)
stage_callbacks = [plot_contour, plot_loss]
stage_callbacks_2 = [plot_contour]
if pseudo_time_enabled:
    prev_net = copy.deepcopy(net)
    update_prev_net = UpdatePrevNet(period=prev_net_update_every)
    resampler = dde.callbacks.PDEPointResampler(period=resample_every)
    stage_callbacks.append(update_prev_net)
    stage_callbacks.append(resampler)
    stage_callbacks_2.append(update_prev_net)

if loss_strategy_enabled:
    gradnorm_cb = GradNormCallback(period=loss_update_every, lr=1e-3)
    stage_callbacks.append(gradnorm_cb)

if re_ramp_enabled:
    re_ramp = ReRampCallback(re_ramp_schedule)
    stage_callbacks.append(re_ramp)

wandb_cb = WandbLossCallback(labels, period=log_every_steps)
stage_callbacks.append(wandb_cb)
stage_callbacks_2.append(wandb_cb)

early_stopping = dde.callbacks.EarlyStopping(min_delta=1e-6, patience=5000)
stage_callbacks.append(early_stopping)
stage_callbacks_2.append(early_stopping)

if separate_enabled:
    for i in range(len(optim_schedule)):
        iter_idx = optim_schedule[i][0]
        lr_idx = optim_schedule[i][1]
        model.compile("adam", lr=lr_idx, loss_weights=loss_weights)
        losshistory, train_state = model.train(iterations=iter_idx, display_every=10, callbacks=stage_callbacks, disregard_previous_best=False, model_save_path=f"{folder_name}/ns_model_{iter_idx}_{lr_idx}")  
else:
    if lr_schedule == "step":     
        model.compile(optimizer_1, lr=learning_rate, decay=(lr_schedule, decay_steps, decay_rate), loss_weights=loss_weights)
        losshistory, train_state = model.train(iterations=itr_1,  display_every=log_every_steps, callbacks=stage_callbacks, disregard_previous_best=False, model_save_path=f"{folder_name}/model_after_{optimizer_1}")
    elif lr_schedule == "exponential":
        model.compile(optimizer_1, lr=learning_rate, decay=(lr_schedule, decay_rate), loss_weights=loss_weights)
        losshistory, train_state = model.train(iterations=itr_1,  display_every=log_every_steps, callbacks=stage_callbacks, disregard_previous_best=False, model_save_path=f"{folder_name}/model_after_{optimizer_1}")

torch.save(model.net, f"{folder_name}/net_after_{optimizer_1}.pt")
# np.save(f"{folder_name}/train_loss_after_{optimizer_1}.npy", np.array(losshistory.loss_train))
# np.save(f"{folder_name}/test_loss_after_{optimizer_1}.npy", np.array(losshistory.loss_test))
# np.save(f"{folder_name}/steps_after_{optimizer_1}.npy", np.array(losshistory.steps))
################# RAR ###############################################################################################################################################
if rar_enabled:

    #initialize initial error 
    X = geom.random_points(100000)
    f = model.predict(X, operator=navier_stokes)
    err_eq = np.abs(f[0]) + np.abs(f[1]) + np.abs(f[2])
    err_eq = err_eq.flatten()
    initial_err = np.mean(err_eq)

    #RAR
    threshold = 0.15 * initial_err 
    X = geom.random_points(100000)
    err = initial_err
    itr_ = 0
    while err > threshold and itr_ <= 5:
        f = model.predict(X, operator=navier_stokes)
        err_eq = np.absolute(f)
        err_eq = np.abs(f[0]) + np.abs(f[1]) + np.abs(f[2])
        err_eq = err_eq.flatten()
        err = np.mean(err_eq)
        print("Mean residual: %.3e" % (err))

        if err <= threshold:
            print("Target residual achieved!")
            break
            
        n_add = 50
        worst_idx = np.argsort(err_eq)[-n_add:]
        print("Adding new points:", X[worst_idx], "\n")
        data.add_anchors(X[worst_idx])
        early_stopping = dde.callbacks.EarlyStopping(min_delta=1e-4, patience=1000)
        plot_contour = PlotCallback(period=1000)
        model.compile("adam", lr=learning_rate)
        losshistory, train_state = model.train(iterations=3000, callbacks=stage_callbacks, display_every=10, disregard_previous_best=True, model_save_path=f"{folder_name}/model_after_RAR_{optimizer_1}")
        torch.save(model.net, f"{folder_name}/net_after_RAR_{optimizer_1}.pt")
        itr_ += 1
    # np.save(f"{folder_name}/train_loss_after_RAR_{optimizer_1}.npy", np.array(losshistory.loss_train))
    # np.save(f"{folder_name}/test_loss_after_RAR_{optimizer_1}.npy", np.array(losshistory.loss_test))
    # np.save(f"{folder_name}/steps_after_RAR_{optimizer_1}.npy", np.array(losshistory.steps))
else:
    itr_ = 0
    
if use_lbfgs:
    model.compile(optimizer_2)
    losshistory, train_state = model.train(iterations=itr_2, callbacks=stage_callbacks_2, display_every=10, model_save_path=f"{folder_name}/model_after_RAR_finish")

torch.save(model.net, f"{folder_name}/net_finish.pt")
# np.save(f"{folder_name}/train_loss_finish_{optimizer_1}_{optimizer_2}.npy", np.array(losshistory.loss_train))
# np.save(f"{folder_name}/test_loss_finish_{optimizer_1}_{optimizer_2}.npy", np.array(losshistory.loss_test))
# np.save(f"{folder_name}/steps_finish_{optimizer_1}_{optimizer_2}.npy", np.array(losshistory.steps))
################# RAR ###############################################################################################################################################
config.training.rar_itr_ = itr_
with open(os.path.join(folder_name, "config_used.json"), "w") as f:
    json.dump(config.to_dict(), f, indent=2)
dde.saveplot(losshistory, train_state, issave=True, isplot=True)
wandb.finish()
fh.close()
#########################################################################
