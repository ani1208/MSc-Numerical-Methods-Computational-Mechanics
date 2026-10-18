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
    def __init__(self, period=1000, momentum=0.9, n_pde_terms=1, lr=1e-4):
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
            residuals = pde(x_t, y_t)  
            R_i = residuals[i]
            L_i = torch.mean(R_i ** 2)                     
            L_i.backward(retain_graph=False)

            total_norm_sq = sum(p.grad.norm()**2 for p in net.parameters() if p.grad is not None)
            grad_norms.append(torch.sqrt(total_norm_sq).item())

        bc_list = [bc, ic]

        for bc in bc_list:
            net.zero_grad()
            try:
                bc_pts = bc.filter(geom.random_boundary_points(200))
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
def gen_testdata():
    """Import and preprocess the dataset with the exact solution."""
    # Load the data:
    data = np.load("Burgers.npz")
    # Obtain the values for t, x, and the excat solution:
    t, x, exact = data["t"], data["x"], data["usol"].T
    # Process the data and flatten it out (like labels and features):
    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    y = exact.flatten()[:, None]

    #Plot reference 
    plt.figure(figsize = (16,4))
    plt.contourf(xx, tt, exact, levels=50, cmap='Spectral_r')
    plt.colorbar(label="u")
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title("Reference Solution for 1D Burgers Equation")
    plt.savefig("Reference.png", dpi=150, bbox_inches="tight")

    return X, y, xx, tt, exact

###############################################################################

def output_transform(x, y):
    # x[:, 0:1] is x-coordinate
    # x[:, 1:2] is y-coordinate
    
    # This term is zero if x=0 or x=L
    x_boundary = x[:, 0:1] * (L - x[:, 0:1])
    
    # This term is zero if y=0 or y=W
    y_boundary = x[:, 1:2] * (W - x[:, 1:2])
    
    # Combine them. Since it's a product, if ANY edge is hit, the result is 0.
    return x_boundary * y_boundary * y

#x_input = x , t
#y = u
def pde(x_input, y):
    dy_dt = dde.grad.jacobian(y, x_input, i=0, j=1)
    dy_dx = dde.grad.jacobian(y, x_input, i=0, j=0)
    dy_xx = dde.grad.hessian(y, x_input, i=0, j=0)
    return dy_dt + y * dy_dx - 0.01 / np.pi * dy_xx

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
#Parameters for the burgers equation
L = 1
n = 1
T = 1
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

#Geometry Definition
geom = dde.geometry.Interval(-1, 1)
time_domain = dde.geometry.TimeDomain(0, T)
geomtime = dde.geometry.GeometryXTime(geom, time_domain)

#BC u(-1,t)=u(L,t)=0
# bc = dde.icbc.DirichletBC(geomtime, lambda x:0, lambda x, on boundary: on_boundary and (dde.utils.isclose(x[0], -1.0) or dde.utils.isclose(x[0], 1.0)))
bc = dde.icbc.DirichletBC(
    geomtime, 
    lambda x: 0, 
    lambda x, on_boundary: on_boundary and (dde.utils.isclose(x[0], -1.0) or dde.utils.isclose(x[0], 1.0))
)
#IC u(x,0)=sin(n.pi.x)
ic = dde.icbc.IC(
    geomtime,
    lambda x: - np.sin(n*np.pi*x[:,0:1]),
    # lambda x: np.exp(-x[:,0:1] ** 2),
    lambda _, on_initial: on_initial,
)

if anchor_incl:
    #add more points near step  
    x_d = np.linspace(-1, 1, 10)
    t_d = np.linspace(0.0, 1, 10)
    X, T = np.meshgrid(x_d, t_d)
    points = np.vstack((X.ravel(), T.ravel())).T
else:
    points = None

data = dde.data.PDE(
    geom,
    pde,
    [bc, ic],
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
# net.apply_feature_transform(input_transform)
# net.apply_output_transform(output_transform)

labels = list(loss_weights_dict.keys())
loss_weights = list(loss_weights_dict.values())

model = dde.Model(data, net)
stage_callbacks = []
stage_callbacks_2 = []
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

resampler = dde.callbacks.PDEPointResampler(period=resample_every)
stage_callbacks.append(resampler)
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
        model.compile(optimizer_1, lr=lr_idx, loss_weights=loss_weights)
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
    f = model.predict(X, operator=pde)
    err_eq = np.abs(f[0])
    err_eq = err_eq.flatten()
    initial_err = np.mean(err_eq)

    #RAR
    threshold = 0.15 * initial_err 
    X = geom.random_points(100000)
    err = initial_err
    itr_ = 0
    while err > threshold and itr_ <= 5:
        f = model.predict(X, operator=pde)
        err_eq = np.absolute(f)
        err_eq = np.abs(f[0]) 
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
        # plot_contour = PlotCallback(period=1000)
        model.compile(optimizer_1, lr=learning_rate)
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
#########################################################################
X, y_true, xx, tt, exact = gen_testdata()
y_pred = model.predict(X)
f = model.predict(X, operator=pde)
print("Mean residual:", np.mean(np.absolute(f)))
print("L2 relative error:", dde.metrics.l2_relative_error(y_true, y_pred))
np.savetxt("test.dat", np.hstack((X, y_true, y_pred)))
y_true=y_true.reshape(100, 256)
#############################################################