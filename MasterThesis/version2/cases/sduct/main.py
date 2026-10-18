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
    def __init__(self, period=1000, momentum=0.9, n_pde_terms=4, lr=1e-4):
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

        bc_list = [inlet_x, inlet_y, outlet_p, outlet_u_x, outlet_v_x, wall_x, wall_y, inlet_nu, wall_nu, outlet_nu_x]

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
        if loss_explicit_enabled:
            for i in loss_indices:
                lambda_hat[i] *= loss_explicit_weight_scale
        lambda_hat = [min(l, loss_clip) for l in lambda_hat]
        


        for i in range(len(grad_norms)):
            self.model.loss_weights[i] = self.momentum * self.model.loss_weights[i] +  (1 - self.momentum) * lambda_hat[i]

        fh.write(f"SA pde grad norm={grad_norms[3]}\n")
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
    x_min, x_max = complete_coords[:, 0].min(), complete_coords[:, 0].max()
    y_min, y_max = complete_coords[:, 1].min(), complete_coords[:, 1].max()
    x_range = np.linspace(x_min, x_max, 1200)
    y_range = np.linspace(y_min, y_max, 1200)
    xx, yy = np.meshgrid(x_range, y_range)
    points = np.vstack((xx.ravel(), yy.ravel())).T
    # Re_column = np.full((points.shape[0], 1), float(Re))
    # X_3D = np.hstack((points, Re_column))
    inside_mask = geom.inside(points)
    prediction = model.predict(points)
    u_component = prediction[:, 0]
    v_component = prediction[:, 1]
    p = prediction[:, 2]
    nu_tilde    = prediction[:,3]
    u_mag_flat = np.sqrt(u_component**2 + v_component**2)
    u_mag_flat1 = np.sqrt(u_component**2 + v_component**2)
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
    plt.title("Velocity Field", fontsize=7)
    plt.savefig(f"{folder_name}/iteration_{epoch}_comparison_Re{Re:.2e}.png", bbox_inches='tight', dpi=600)
    plt.close()

    plt.figure(figsize=(10, 5), dpi=300)
    nu_tilde[~inside_mask] = np.nan
    nu_mag = nu_tilde.reshape(xx.shape)
    numax = np.nanmax(nu_mag)
    contour = plt.contourf(xx, yy, nu_mag, levels=200, cmap='Spectral_r', antialiased=True)
    cbar = plt.colorbar(contour, orientation='horizontal', pad=0.15, shrink=0.3)
    cbar.ax.tick_params(labelsize=5)
    cbar.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    cbar.set_label('Nu tilde Magnitude', fontsize=7)
    plt.xlim(0.0, 1.9)
    plt.ylim(0.04, 0.71)
    plt.gca().set_aspect('equal')
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.xlabel("X coord.", fontsize=7)
    plt.ylabel("Y coord.", fontsize=7)
    plt.title(f"Nu Tilde Field (max {numax})", fontsize=7)
    plt.savefig(f"{folder_name}/nuContour_iteration{epoch}_Re{Re:.2e}.png", bbox_inches='tight', dpi=300)
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
    x_min, x_max = complete_coords[:, 0].min(), complete_coords[:, 0].max()
    y_min, y_max = complete_coords[:, 1].min(), complete_coords[:, 1].max()
    x_range_r = np.linspace(x_min, x_max, 200)
    y_range_r = np.linspace(y_min, y_max, 200)
    xxr, yyr = np.meshgrid(x_range_r, y_range_r)
    points_r = np.vstack((xxr.ravel(), yyr.ravel())).T
    inside_mask_r = geom.inside(points_r)
    residuals = model.predict(points_r, operator=navier_stokes_with_SA)
    labels = ["continuity", "x-mom", "y-mom", "nu transport"]
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
def wall_boundary(x, on_boundary):
    return on_boundary and not (inlet_boundary(x, on_boundary) or outlet_boundary(x, on_boundary))


def outlet_boundary(x, on_boundary):
    return on_boundary and np.isclose(x[0], 1.888)


def inlet_boundary(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0.00168133)


def inlet_velocity(x):
    xcoord = x[:, 0:1]
    ycoord = x[:, 1:2]
    #scale for making mean velocity 1
    scale = 1 / 0.9474910560577705
    return scale * np.tanh(k * (0.2000252 - np.abs(ycoord - 0.5000252))).reshape(-1, 1)


def zero_velocity(x):
    return np.zeros((x.shape[0], 1))

###############################################################################
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

def distance_to_wall(x_torch, y_torch):
    pts = torch.cat([x_torch, y_torch], dim=1).detach().cpu().numpy()
    d = griddata(sdf_points, sdf_values, pts, method='linear')
    nan_mask = np.isnan(d)
    if nan_mask.any():
        d[nan_mask] = griddata(sdf_points, sdf_values, pts[nan_mask], method='nearest')
    return torch.tensor(d, dtype=x_torch.dtype, device=x_torch.device).reshape(-1, 1)


def sdf_query(x_tensor, y_tensor):
    device = x_tensor.device
    dtype = x_tensor.dtype

    x_np = x_tensor.detach().cpu().numpy().ravel()
    y_np = y_tensor.detach().cpu().numpy().ravel()
    pts = np.column_stack((x_np, y_np))

    vals = _linear_interp(pts)

    nan_mask = np.isnan(vals)
    if nan_mask.any():
        vals[nan_mask] = _nearest_interp(pts[nan_mask])

    vals_t = torch.tensor(vals, dtype=dtype, device=device).reshape(-1, 1)
    return vals_t

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


def inlet_nu_tilde(x):
    return np.full((x.shape[0],1), 5 * (1/Re))

def init_wall_segments(top_coords, bottom_coords, device, dtype):
  
    all_x = np.concatenate([top_coords[:, 0], bottom_coords[:, 0]])
    all_y = np.concatenate([top_coords[:, 1], bottom_coords[:, 1]])
    top_n = len(top_coords)
    bottom_n = len(bottom_coords)
 
    idx1 = np.concatenate([np.arange(top_n - 1), np.arange(top_n, top_n + bottom_n - 1)])
    idx2 = idx1 + 1
 
    x1 = torch.tensor(all_x[idx1], dtype=dtype, device=device).unsqueeze(1)
    y1 = torch.tensor(all_y[idx1], dtype=dtype, device=device).unsqueeze(1)
    x2 = torch.tensor(all_x[idx2], dtype=dtype, device=device).unsqueeze(1)
    y2 = torch.tensor(all_y[idx2], dtype=dtype, device=device).unsqueeze(1)
 
    dx, dy = x2 - x1, y2 - y1
    L2 = dx * dx + dy * dy
 
    _wall_segment_cache['x1'] = x1
    _wall_segment_cache['y1'] = y1
    _wall_segment_cache['x2'] = x2
    _wall_segment_cache['y2'] = y2
    _wall_segment_cache['dx'] = dx
    _wall_segment_cache['dy'] = dy
    _wall_segment_cache['L2'] = L2
 

def combine_phi1(x, y, m=1, eps=1e-8):
    x1 = _wall_segment_cache['x1']
    y1 = _wall_segment_cache['y1']
    dx = _wall_segment_cache['dx']
    dy = _wall_segment_cache['dy']
    L2 = _wall_segment_cache['L2'] + eps

    xq = x.T
    yq = y.T

    # dx, dy = x2-x1, y2-y1
    # L2 = dx*dx + dy*dy + eps
    # t = ((xq-x1)*dx + (yq-y1)*dy) / L2
    # t = torch.clamp(t, 0.0, 1.0)
    # proj_x = x1 + t*dx
    # proj_y = y1 + t*dy
    # phi_segment = torch.sqrt((xq-proj_x)**2 + (yq-proj_y)**2 + eps)

    # phi_recip_sum = (1.0 / (phi_segment ** m)).sum(dim=0)
    # phi = phi_recip_sum ** (-1.0 / m)
    # return phi.unsqueeze(1)
    t = ((xq - x1) * dx + (yq - y1) * dy) / L2
    t = torch.clamp(t, 0.0, 1.0)
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    phi_segment = torch.sqrt((xq - proj_x) ** 2 + (yq - proj_y) ** 2 + eps)
    # print(phi_segment.shape)
    # phi_recip_sum = (1.0 / (phi_segment ** m)).sum(dim=0)
    # phi = (phi_recip_sum + eps) ** (-1.0 / m)
    # phi_binary = torch.where(phi > 0, torch.ones_like(phi), torch.zeros_like(phi))
    # return phi.unsqueeze(1) 
    phi = torch.min(phi_segment, dim=0).values
    return phi.unsqueeze(1) 


def point_to_segment_dist(px, py, x1, y1, x2, y2):
    dx, dy = x2-x1, y2-y1
    L2 = dx*dx + dy*dy
    t = ((px-x1)*dx + (py-y1)*dy) / L2
    t = torch.clamp(t, 0.0, 1.0)
    proj_x = x1 + t*dx
    proj_y = y1 + t*dy
    return torch.sqrt((px-proj_x)**2 + (py-proj_y)**2)

def navier_stokes_with_SA(x, y):

    x_coord, y_coord = x[:, 0:1], x[:, 1:2]
    u, v, p, nu_tilde = y[:, 0:1], y[:, 1:2], y[:, 2:3], y[:, 3:4]

    ########pseudo time stepping##################
    time_step = 0.1
    if pseudo_time_enabled:
        with torch.no_grad():
            prev_predicition = prev_net(x) 
        u_prev = prev_predicition[:,0:1]
        v_prev = prev_predicition[:,1:2]
        p_prev = prev_predicition[:,2:3]
        nu_tilde_prev = prev_predicition[:,3:4]
        term1 = (u - u_prev)/time_step
        term2 = (v - v_prev)/time_step
        term3 = (p - p_prev)/time_step
        term4 = (nu_tilde - nu_tilde_prev)/time_step
    else: 
        term1 = torch.zeros_like(u)
        term2 = torch.zeros_like(v)
        term3 = torch.zeros_like(p)
        term4 = torch.zeros_like(nu_tilde)
    ##############################################

    cb1, cb2 = 0.1355, 0.622
    kappa, sigma = 0.41, 2/3
    cw1 = (cb1 / kappa**2) + (1 + cb2)/sigma
    cw2, cw3 = 0.3, 2
    cv1 = 7.1
    xi = nu_tilde * Re
    fv1 = torch.pow(xi, 3) / (torch.pow(xi, 3) + cv1**3 + 1e-6)
    fv2 = 1 - xi / (1 + xi * fv1 + 1e-6)     
    u_x = dde.grad.jacobian(y, x, i=0, j=0)
    u_y = dde.grad.jacobian(y, x, i=0, j=1)
    v_x = dde.grad.jacobian(y, x, i=1, j=0)
    v_y = dde.grad.jacobian(y, x, i=1, j=1)
    p_x = dde.grad.jacobian(y, x, i=2, j=0)
    p_y = dde.grad.jacobian(y, x, i=2, j=1)
    nu_tilde_x = dde.grad.jacobian(y, x, i=3, j=0)
    nu_tilde_y = dde.grad.jacobian(y, x, i=3, j=1)
    u_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
    u_yy = dde.grad.hessian(y, x, component=0, i=1, j=1)
    v_xx = dde.grad.hessian(y, x, component=1, i=0, j=0)
    v_yy = dde.grad.hessian(y, x, component=1, i=1, j=1)
    
    # S_xx, S_yy, S_xy = u_x, v_y, 0.5*(u_y + v_x)
    # S = torch.sqrt(2*(S_xx**2 + S_yy**2 + 2*S_xy**2) + 1e-6) 

    d = sdf_query(x_coord, y_coord)
    # d = torch.clamp(d, min=0.01)

    M_const = 1e-5
    omega = torch.abs(u_y - v_x)
    S = torch.sqrt(omega**2 + M_const**2) - M_const
    
    S_N = (d**2) * S                                  
    Sbar_N = nu_tilde * fv2 / (kappa**2)              
    S_tilde_N = torch.clamp(S_N + Sbar_N, min=1e-10)  
    r = torch.clamp(nu_tilde / (S_tilde_N * kappa**2 + 1e-6), min=0.0, max=10.0)  
    g = r + cw2*(torch.pow(r, 6) - r)
    fw = g * torch.pow((1 + cw3**6) / (torch.pow(g, 6) + cw3**6), 1/6)
    SN_p = cb1 * S_tilde_N * nu_tilde                 
    SN_d = cw1 * fw * nu_tilde**2                      
    tau_x_1 = dde.grad.jacobian(nu_tilde * fv1 * u_x, x, i=0, j=0)
    tau_x_2 = dde.grad.jacobian(nu_tilde * fv1 * (u_y + v_x), x, i=0, j=1)
    tau_y_1 = dde.grad.jacobian(nu_tilde * fv1 * (u_y + v_x), x, i=0, j=0)
    tau_y_2 = dde.grad.jacobian(nu_tilde * fv1 * v_y, x, i=0, j=1)
    continuity = term3 + u_x + v_y
    x_momentum = term1 + u*u_x + v*u_y + p_x - (1.0/Re)*(u_xx+u_yy) - 2*tau_x_1 - tau_x_2
    y_momentum = term2 + u*v_x + v*v_y + p_y - (1.0/Re)*(v_xx+v_yy) - tau_y_1 - 2*tau_y_2
    nu_tilde_diffusion_x = dde.grad.jacobian((1/Re + nu_tilde) * nu_tilde_x, x, i=0, j=0)
    nu_tilde_diffusion_y = dde.grad.jacobian((1/Re + nu_tilde) * nu_tilde_y, x, i=0, j=1)
    
    _print_counter["n"] += 1
    if _print_counter["n"] % 1000 == 0:
        fh.write(f"d min={d.min().item():.4e} max={d.max().item():.4e}  "
        f"r min={r.min().item():.4e} max={r.max().item():.4e}  "
        f"g min={g.min().item():.4e} max={g.max().item():.4e}  "
        f"fw min={fw.min().item():.4e} max={fw.max().item():.4e}  "
        f"S_tilde_N mean={S_tilde_N.mean().item():.4e}  gen={SN_p.mean().item():.4e}  "
        f"destr={SN_d.mean().item():.4e}  nu_tilde mean={nu_tilde.mean().item():.4e}"
        f"S_tilde_N/nu_tilde mean = {S_tilde_N.mean().item()/nu_tilde.mean().item():.4e}\n")
        fh.flush()

    nu_convection = (d**2) * (u*nu_tilde_x + v*nu_tilde_y)
    nu_diffusion  = (d**2) * (1/sigma) * (nu_tilde_diffusion_x + nu_tilde_diffusion_y)
    nu_cross_diff = (d**2) * (cb2/sigma) * (nu_tilde_x**2 + nu_tilde_y**2)
    nu_tilde_transport = term4 + nu_convection - SN_p - nu_diffusion - nu_cross_diff + SN_d

    return [continuity, x_momentum, y_momentum, nu_tilde_transport]
    

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
Re = 2.5e6
k = 66
m = 15
_wall_segment_cache = {}
_print_counter = {"n": 0}
############################################################################################

run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
folder_name = f"runs/{run_time}"
os.makedirs(folder_name, exist_ok=True)

if fourier_enabled:
    # change the amplitude to 0.5 or 1 if required in fourier lift
    B = torch.randn(fourier_dim, 2) * fourier_amp
    torch.save(B, f"{folder_name}/fourier_B.pt")
    
fh = open(f"{folder_name}/nu_tilde_values.txt", 'a')

wandb.init(
    project=config.wandb.project,
    name=config.wandb.name,
    tags=[config.wandb.tag] if config.wandb.tag else None,
    config=config.to_dict(),
)
run_info = {"wandb_run_id": wandb.run.id, "wandb_project": config.wandb.project}
with open(os.path.join(folder_name, "wandb_run_info.json"), "w") as f:   
    json.dump(run_info, f, indent=2)
####################  Distance field generation ##############################################
#Top wall coords
top_coords = np.load('sduct_topwall.npy')
#Bottom wall coords
bottom_coords = np.flipud(np.load('sduct_bottomwall.npy'))
complete_coords = np.vstack([top_coords, bottom_coords])
x_min, x_max = complete_coords[:, 0].min(), complete_coords[:, 0].max()
y_min, y_max = complete_coords[:, 1].min(), complete_coords[:, 1].max()
x = complete_coords[:, 0:1]
y = complete_coords[:, 1:2]
bottom_coords_x = bottom_coords[:, 0:1]
bottom_coords_y1 = bottom_coords[:, 1:2] + 0.28003528
bottom_coords_y2 = bottom_coords[:, 1:2] + 0.2000252
bottom_coords_y3 = bottom_coords[:, 1:2] + 0.12001512
plt.figure(figsize=(10, 8))
plt.plot(x, y, 'o', label="Duct Coordinates", linewidth=2, markersize=4)
plt.plot(bottom_coords_x, bottom_coords_y1, 'b', label="Duct Coordinates1", linestyle="-.", linewidth=2, markersize=4)
plt.plot(bottom_coords_x, bottom_coords_y2, 'r', label="Duct Coordinates2", linestyle="--", linewidth=2, markersize=4)
plt.plot(bottom_coords_x, bottom_coords_y3, 'y', label="Duct Coordinates3", linestyle=":", linewidth=2, markersize=4)
plt.xlabel("X coordinate")
plt.ylabel("Y coordinate")
plt.savefig(f"{folder_name}/duct_shape.png", bbox_inches='tight', dpi=300)

geom = dde.geometry.Polygon(complete_coords)

sdf_data = np.load('xysdf_sduct.npy')
sdf_x = sdf_data[0].ravel()
sdf_y = sdf_data[1].ravel()
sdf_values = sdf_data[2].ravel()
sdf_points = np.column_stack((sdf_x, sdf_y))
_nearest_interp = NearestNDInterpolator(sdf_points, sdf_values)
_linear_interp = LinearNDInterpolator(sdf_points, sdf_values)
plt.figure(figsize=(10, 8))
grid_res = 100
xi = np.linspace(sdf_x.min(), sdf_x.max(), grid_res)
yi = np.linspace(sdf_y.min(), sdf_y.max(), grid_res)
X, Y = np.meshgrid(xi, yi)
Z = griddata(sdf_points, sdf_values, (X, Y), method='linear')
grid_points = np.column_stack((X.ravel(), Y.ravel()))
inside_mask = geom.inside(grid_points)
Z_flat = Z.ravel()
Z_flat[~inside_mask] = np.nan
Z = Z_flat.reshape(X.shape)
contour = plt.contourf(X, Y, Z, levels=200, cmap='Spectral_r', antialiased=True)
plt.scatter(sdf_x, sdf_y, c=sdf_values, cmap='Spectral_r',
            edgecolors='k', s=5, alpha=0.6, label="Duct Coordinates")
cbar = plt.colorbar(contour, orientation='horizontal', pad=0.15, shrink=0.3)
plt.savefig(f"{folder_name}/distance_to_wall.png", bbox_inches='tight', dpi=300)

xi = np.linspace(sdf_x.min(), sdf_x.max(), grid_res)
yi = np.linspace(sdf_y.min(), sdf_y.max(), grid_res)
X, Y = np.meshgrid(xi, yi)
grid_points = np.column_stack((X.ravel(), Y.ravel()))
sdf_grid_vals = griddata(sdf_points, sdf_values, grid_points, method='linear')
nan_mask = np.isnan(sdf_grid_vals)
sdf_grid_vals[nan_mask] = griddata(sdf_points, sdf_values, grid_points[nan_mask], method='nearest')
sdf_grid_vals = sdf_grid_vals.reshape(grid_res, grid_res)
sdf_grid_tensor = torch.tensor(sdf_grid_vals, dtype=torch.float32).unsqueeze(0).unsqueeze(0) 
######################################################################################################

inlet_x = dde.icbc.DirichletBC(geom, inlet_velocity, inlet_boundary, component=0)
inlet_y = dde.icbc.DirichletBC(geom, zero_velocity, inlet_boundary, component=1)
outlet_p = dde.icbc.DirichletBC(geom, zero_velocity, outlet_boundary, component=2)
outlet_u_x = dde.icbc.NeumannBC(geom, zero_velocity, outlet_boundary, component=0)
outlet_v_x = dde.icbc.NeumannBC(geom, zero_velocity, outlet_boundary, component=1)
wall_x = dde.icbc.DirichletBC(geom, zero_velocity, wall_boundary, component=0)
wall_y = dde.icbc.DirichletBC(geom, zero_velocity, wall_boundary, component=1)
inlet_nu = dde.icbc.DirichletBC(geom, inlet_nu_tilde, inlet_boundary, component=3)
wall_nu = dde.icbc.DirichletBC(geom, zero_velocity, wall_boundary, component=3)
outlet_nu_x = dde.icbc.NeumannBC(geom, zero_velocity, outlet_boundary, component=3)

if bc_enforcement:
    bc = [inlet_x, inlet_y, outlet_p, outlet_u_x, outlet_v_x, inlet_nu, outlet_nu_x]
else:
    bc = [inlet_x, inlet_y, outlet_p, outlet_u_x, outlet_v_x, wall_x, wall_y, inlet_nu, wall_nu, outlet_nu_x]
    
if anchor_incl:
    #add more points near step  
    offsets = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]

    near_wall_pts = []
    for off in offsets:
        top_pts = np.column_stack([top_coords[:, 0], top_coords[:, 1] - off])
        bot_pts = np.column_stack([bottom_coords[:, 0], bottom_coords[:, 1] + off])
        near_wall_pts.append(top_pts)
        near_wall_pts.append(bot_pts)

    near_wall_pts = np.vstack(near_wall_pts)
    inside_mask = geom.inside(near_wall_pts)
    points = near_wall_pts[inside_mask]
else:
    points = None

data = dde.data.PDE(
    geom,
    navier_stokes_with_SA,
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
resampler = dde.callbacks.PDEPointResampler(period=resample_every)
stage_callbacks = [plot_contour, plot_loss]
stage_callbacks_2 = [plot_contour]
stage_callbacks.append(resampler)
if pseudo_time_enabled:
    prev_net = copy.deepcopy(net)
    update_prev_net = UpdatePrevNet(period=prev_net_update_every)
    stage_callbacks.append(update_prev_net)
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
    f = model.predict(X, operator=navier_stokes_with_SA)
    err_eq = np.abs(f[0]) + np.abs(f[1]) + np.abs(f[2]) + np.abs(f[3]) 
    err_eq = err_eq.flatten()
    initial_err = np.mean(err_eq)

    #RAR
    threshold = 0.15 * initial_err 
    X = geom.random_points(100000)
    err = initial_err
    itr_ = 0
    while err > threshold and itr_ <= 5:
        f = model.predict(X, operator=navier_stokes_with_SA)
        err_eq = np.absolute(f)
        err_eq = np.abs(f[0]) + np.abs(f[1]) + np.abs(f[2]) + np.abs(f[3])
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
    losshistory, train_state = model.train(iterations=itr_2, callbacks=stage_callbacks_2, display_every=10, model_save_path=f"{folder_name}/model_after_LBFGS_finish")

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
