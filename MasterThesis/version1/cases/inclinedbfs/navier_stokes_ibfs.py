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
# dde.config.set_default_float("float64")
# torch.set_default_dtype(torch.float64)
import warnings
import shutil
warnings.filterwarnings("ignore", category=DeprecationWarning)
SEED = 1
np.random.seed(SEED)
torch.manual_seed(SEED)
dde.config.set_random_seed(SEED)

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

        bc_list = [inlet_x, inlet_y, outlet_p, wallt_x, wallt_y, wallb_x, wallb_y, wallsh_x, wallsh_y, wallsv_x, wallsv_y, inlet_nu, wallt_nu, wallb_nu, wallh_nu, wall_nu_incline, outlet_nu_x]

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
        # lambda_hat[0] *= 10
        lambda_hat = [min(l, 100.0) for l in lambda_hat]
        for i in range(len(grad_norms)):
            self.model.loss_weights[i] = self.momentum * self.model.loss_weights[i] +  (1 - self.momentum) * lambda_hat[i]

        print(f"[iter {iteration}] grad_norms={grad_norms}  new_lambdas={model.loss_weights}")

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

class UpdatePrevNet(dde.callbacks.Callback):
    def __init__(self, period=10):
        super().__init__()
        self.period = period
 
    def on_epoch_end(self):
        global prev_net
        epoch = self.model.train_state.epoch
        if epoch % self.period == 0 and epoch > 0:
            prev_net = copy.deepcopy(self.model.net)


class PlotCallback(dde.callbacks.Callback):
    def __init__(self, period=100):
        super().__init__()
        self.period = period

    def on_epoch_end(self):
        epoch = self.model.train_state.epoch
        # self.on_epoch_end runs at every iteration step
        if epoch % self.period == 0:
            contour_plotting_at_checkpoints(self.model, epoch)

class PlotCallback1(dde.callbacks.Callback):
    def __init__(self, period=100):
        super().__init__()
        self.period = period

    def on_epoch_end(self):
        epoch = self.model.train_state.epoch
        # self.on_epoch_end runs at every iteration step
        if epoch % self.period == 0:
            residual_plotting_at_checkpoints(self.model, epoch)


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



def contour_plotting_at_checkpoints(model, epoch, theta_plot=45.0):
    x_dense = np.linspace(x_[0], x_[-1], 500)
    y_dense = np.linspace(y2_[-1], y1_[0], 200)
    X, Y = np.meshgrid(x_dense, y_dense)
    points = np.vstack((X.ravel(), Y.ravel())).T
    inside_mask = geom.inside(points)
    prediction = model.predict(points)
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
    plt.savefig(f"{folder_name}/velContour_iteration{epoch}_theta{theta_plot}_Re{Re}.png", bbox_inches='tight', dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5), dpi=300)
    nu_tilde[~inside_mask] = np.nan
    nu_mag = nu_tilde.reshape(X.shape)
    numax = np.nanmax(nu_mag)
    contour = plt.contourf(X, Y, nu_mag, levels=200, cmap='Spectral_r', antialiased=True)
    cbar = plt.colorbar(contour, orientation='horizontal', pad=0.15, shrink=0.3)
    cbar.ax.tick_params(labelsize=5)
    cbar.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    cbar.set_label('Nu tilde Magnitude', fontsize=7)
    plt.xlim(x_[0] - 0.1, x_[-1] + 0.1)
    plt.ylim(y2_[-1] - 0.1, y1_[0] + 0.1)
    plt.gca().set_aspect('equal')
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.xlabel("X coord.", fontsize=7)
    plt.ylabel("Y coord.", fontsize=7)
    plt.title(f"Nu Tilde Field (theta={theta_plot} deg, max {numax})", fontsize=7)
    plt.savefig(f"{folder_name}/nuContour_iteration{epoch}_theta{theta_plot}_Re{Re}.png", bbox_inches='tight', dpi=300)
    plt.close()

def residual_plotting_at_checkpoints(model, epoch):
    x_dense = np.linspace(x_[0], x_[-1], 200)
    y_dense = np.linspace(y2_[-1], y1_[0], 200)
    X, Y = np.meshgrid(x_dense, y_dense)
    points = np.vstack((X.ravel(), Y.ravel())).T
    inside_mask = geom.inside(points)
    residuals = model.predict(points, operator=navier_stokes_with_SA)
    labels = ["continuity", "x-mom", "y-mom", "nu transport"]
    # labels = ["continuity", "x-mom", "y-mom"]
    fr.write("******************************************************************\n")
    for i, label in enumerate(labels):
        res = residuals[i].flatten()
        res_abs = np.abs(res)
        res_abs[~inside_mask] = np.nan
        res_grid = res_abs.reshape(X.shape)
        res_inside = res[inside_mask] 
        res_mean_abs = np.mean(np.abs(res_inside))
        res_l2 = np.sqrt(np.mean(res_inside**2))
        fr.write(f"residual_mean_abs/{label} : {res_mean_abs:.4e}, residual_l2/{label} : {res_l2:.4e}\n")
        plt.figure(figsize=(10, 5), dpi=300)
        contour = plt.contourf(X, Y, res_grid, levels=100, cmap='inferno', antialiased=True)
        cbar = plt.colorbar(contour, orientation='horizontal', pad=0.15, shrink=0.3)
        cbar.ax.tick_params(labelsize=5)
        cbar.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        cbar.set_label(rf"|{label} residual|", fontsize=8)
        plt.xlim(x_[0] - 0.1, x_[-1] + 0.1)
        plt.ylim(y2_[-1] - 0.1, y1_[0] + 0.1)
        plt.gca().set_aspect('equal')
        plt.title(f"{label} residual (epoch {epoch})", fontsize=9)
        plt.savefig(f"{folder_name}/residual_{label}_epoch{epoch}.png", bbox_inches='tight', dpi=300)
        plt.close()
    fr.write("******************************************************************\n")
    fr.flush()


def wall_top_boundary(x, on_boundary):
    return on_boundary and np.isclose(x[1], y1_[0]) and not np.isclose(x[0], x_[0])


def wall_bottom_boundary(x, on_boundary):
    return on_boundary and np.isclose(x[1], y2_[6]) 


def wall_mid_horizontal_boundary(x, on_boundary):
    return on_boundary and (np.isclose(x[1], y2_[0]) and x[0] < x_[3])

def wall_incline_boundary(x, on_boundary):
    if not on_boundary:
        return False
    if not (x_[3] <= x[0] <= x_[4]):
        return False
    # interpolate expected y on the incline line at this x
    y_expected = np.interp(x[0], [x_[3], x_[4]], [y2_[3], y2_[4]])
    return np.isclose(x[1], y_expected, atol=1e-6)

def outlet_boundary(x, on_boundary):
    return on_boundary and np.isclose(x[0], x_[7])


def inlet_boundary(x, on_boundary):
    return on_boundary and np.isclose(x[0], x_[0])

def inlet_velocity(x):
    ycoord = x[:, 1:2]
    y_center = (y1_[0] + y2_[0]) / 2.0
    y_halfwidth = (y1_[0] - y2_[0]) / 2.0
    amplitude = 1
    n = 3
    t = np.clip(np.abs(ycoord - y_center) / y_halfwidth, 0, 1)
    return amplitude * (1 - t**n)**(1/n)

def zero_velocity(x):
    return np.zeros((x.shape[0], 1))

def input_transform(x):
    x = x.float()
    x_scaled = (x[:,0:1]-x_min)/(x_max-x_min)
    y_scaled = (x[:,1:2]-y_min)/(y_max-y_min)
    xy = torch.cat([x_scaled, y_scaled], dim=1)
    proj = xy @ B.T
    # proj returns N,128 -> N, 2 and 2, 128
    # returns N, 256 due to N,128 and N, 128 stacked side by side , hence input is 256 for each point
    return torch.cat([torch.cos(proj), torch.sin(proj)], dim=1) 
    # return torch.cat([x_scaled, y_scaled], dim=1)  

def navier_stokes(x, y):
    u, v, p = y[:, 0:1], y[:, 1:2], y[:, 2:3]
    
    ########pseudo time stepping##################
    time_step = 0.1
    with torch.no_grad():
        prev_predicition = prev_net(x) 
    u_prev = prev_predicition[:,0:1]
    v_prev = prev_predicition[:,1:2]
    p_prev = prev_predicition[:,2:3]
    term1 = (u - u_prev)/time_step
    term2 = (v - v_prev)/time_step
    term3 = (p - p_prev)/time_step
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
#[20, -1.5]
#[42, -1.5]
def distance_to_wall(x_torch, y_torch):
    pts = torch.cat([x_torch, y_torch], dim=1).detach().cpu().numpy()
    d = griddata(sdf_points, sdf_values, pts, method='linear')
    nan_mask = np.isnan(d)
    if nan_mask.any():
        d[nan_mask] = griddata(sdf_points, sdf_values, pts[nan_mask], method='nearest')
    return torch.tensor(d, dtype=x_torch.dtype, device=x_torch.device).reshape(-1, 1)

def sdf_query(x_tensor, y_tensor):
    x_tensor = x_tensor.float()
    y_tensor = y_tensor.float()
    # device = x_tensor.device
    # grid = sdf_grid_tensor.to(device)
    x0_f = float(x_[0]) 
    Ld_f = float(Ld) 
    y2last_f = float(y2_[-1]) 
    y10_f = float(y1_[0])

    # normalize coords to [-1, 1] for grid_sample
    x_norm = 2 * (x_tensor - x0_f) / (Ld_f - x0_f) - 1
    y_norm = 2 * (y_tensor - y2last_f) / (y10_f - y2last_f) - 1

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
    # nu_tilde = NU_TILDE_SCALE * torch.sigmoid(nu_tilde_raw)
    # nu_tilde = nu_tilde_raw**2
    nu_tilde = torch.abs(nu_tilde_raw)
    # softplus has no upper bound - so try this
    # nu_tilde = torch.nn.functional.softplus(nu_tilde_raw)
    return torch.cat([u, v, p, nu_tilde], dim=1)


def inlet_nu_tilde(x):
    return np.full((x.shape[0],1), 5 * (1/Re))

def point_to_segment_dist(px, py, x1, y1, x2, y2, eps=1e-8):
    dx, dy = x2 - x1, y2 - y1
    L2 = dx * dx + dy * dy + eps
    t = ((px - x1) * dx + (py - y1) * dy) / L2
    t = torch.clamp(t, 0.0, 1.0)
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return torch.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2 + eps)
 
def combine_phi(x, y, eps = 1e-8):
 
    ones = torch.ones_like(x)
    zeros = torch.zeros_like(x)
    phi_top = point_to_segment_dist(x, y, zeros, ones * y1_[0], Ld * ones, ones * y1_[0])
    phi_bottom_up = point_to_segment_dist(x, y, x_[3] * ones, y2_[0] * ones, zeros, y2_[0] * ones)
    phi_step_incline = point_to_segment_dist(x, y, x_[4] * ones, y2_[4] * ones, x_[3] * ones, y2_[3] * ones)
    phi_bottom_down = point_to_segment_dist(x, y, Ld * ones, y2_[6] * ones, x_[4] * ones, y2_[6] * ones)
    # phi_inlet = point_to_segment_dist(x, y, zeros, y2_[0] * ones, zeros, y1_[0] * ones)
    # phi_outlet = point_to_segment_dist(x, y, Ld * ones, y1_[0] * ones, Ld * ones, y2_[6] * ones)
    # phi_recip_sum = (1.0 / (phi_inlet ** m + eps) + 1.0 / (phi_top ** m + eps) + 1.0 / (phi_outlet ** m + eps) + 1.0 / (phi_bottom_down ** m + eps) + 1.0 / (phi_step_incline ** m + eps) + 1.0 / (phi_bottom_up ** m + eps))
    # phi = (phi_recip_sum + eps) ** (-1.0 / m)
    
    # return phi
    return torch.minimum(torch.minimum(phi_top, phi_bottom_up), torch.minimum(phi_step_incline, phi_bottom_down))


def navier_stokes_with_SA(x, y):

    x_coord, y_coord = x[:, 0:1], x[:, 1:2]
    u, v, p, nu_tilde = y[:, 0:1], y[:, 1:2], y[:, 2:3], y[:, 3:4]

    ########pseudo time stepping##################
    time_step = 0.1
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
    d = torch.clamp(d, min=0.01)

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
        f"destr={SN_d.mean().item():.4e}  nu_tilde mean={nu_tilde.mean().item():.4e}\n")
        fh.flush()

    nu_convection = (d**2) * (u*nu_tilde_x + v*nu_tilde_y)
    nu_diffusion  = (d**2) * (1/sigma) * (nu_tilde_diffusion_x + nu_tilde_diffusion_y)
    nu_cross_diff = (d**2) * (cb2/sigma) * (nu_tilde_x**2 + nu_tilde_y**2)
    nu_tilde_transport = term4 + nu_convection - SN_p - nu_diffusion - nu_cross_diff + SN_d

    return [continuity, x_momentum, y_momentum, nu_tilde_transport]
    

#######################################################################################
run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
folder_name =  f"../finalresults/ibfs/plots_{run_time}_ibfs"
os.makedirs(folder_name, exist_ok=True)
os.makedirs(f"{folder_name}/ckpt", exist_ok=True)
shutil.copy(__file__, os.path.join(folder_name, "script_snapshot.py"))
k = 66
Re = 36000
itr = 50000
m = 15
_print_counter = {"n": 0}

plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
})


fh = open(f"{folder_name}/nu_tilde_values.txt", 'a')
fr = open(f"{folder_name}/residual_values.txt", 'a')
# change the amplitude to 0.5 or 1 if required in fourier lift
B = torch.randn(128, 2) * 2.0
B = B.float()
x_, y1_, y2_, Ld = inclined_backward_step(45)
x_dense = np.linspace(x_[0], Ld, 800)
y_dense = np.linspace(y2_[-1], y1_[0], 400)
X, Y = np.meshgrid(x_dense, y_dense)
pts_x = torch.tensor(X.ravel(), dtype=torch.float32).reshape(-1,1)
pts_y = torch.tensor(Y.ravel(), dtype=torch.float32).reshape(-1,1)
d_vals = combine_phi(pts_x, pts_y).detach().cpu().numpy().ravel()
np.save('xysdf_ibfs45_GEN.npy', np.array([X.ravel(), Y.ravel(), d_vals]))

sdf_data = np.load('xysdf_ibfs45_GEN.npy')
sdf_x = sdf_data[0].ravel()
sdf_y = sdf_data[1].ravel()
sdf_values = sdf_data[2].ravel()
sdf_points = np.column_stack((sdf_x, sdf_y))
_nearest_interp = NearestNDInterpolator(sdf_points, sdf_values)
_linear_interp = LinearNDInterpolator(sdf_points, sdf_values)
x_, y1_, y2_, Ld = inclined_backward_step(45)
x_min, x_max = float(x_[0]), float(x_[7])
y_min, y_max = float(y2_[6]), float(y1_[0])
x_dense = np.linspace(x_[0], Ld, 500)
y_dense = np.linspace(y2_[-1], y1_[0], 200)
X, Y = np.meshgrid(x_dense, y_dense)

plt.figure(figsize=(10, 8))
Z = griddata(sdf_points, sdf_values, (X, Y), method='linear')
y_bottom = np.interp(X.ravel(), x_, y2_).reshape(X.shape)
Z[Y < y_bottom] = np.nan
Zmax = np.nanmax(Z)
Zmin = np.nanmin(Z)
contour = plt.contourf(X, Y, Z, levels=200, cmap='Spectral_r', antialiased=True)
cbar = plt.colorbar(contour, orientation='horizontal', pad=0.15, shrink=0.3)
plt.gca().set_aspect('equal')
cbar.ax.tick_params(labelsize=4)
plt.title(f"SDF {Zmax} {Zmin}", fontsize=7)
plt.xlim(x_[0] - 0.1, Ld + 0.1)
plt.ylim(y2_[-1] - 0.1, y1_[0] + 0.1)
plt.savefig(f"{folder_name}/distance_to_wall.png", bbox_inches='tight', dpi=300)

grid_res_x, grid_res_y = 800, 400
gx = np.linspace(x_[0], Ld, grid_res_x)
gy = np.linspace(y2_[-1], y1_[0], grid_res_y)
GX, GY = np.meshgrid(gx, gy)
grid_pts = np.column_stack([GX.ravel(), GY.ravel()])

sdf_grid_vals = griddata(sdf_points, sdf_values, grid_pts, method='linear')
nan_mask = np.isnan(sdf_grid_vals)
sdf_grid_vals[nan_mask] = griddata(sdf_points, sdf_values, grid_pts[nan_mask], method='nearest')
sdf_grid_vals = sdf_grid_vals.reshape(grid_res_y, grid_res_x)
sdf_grid_tensor = torch.tensor(sdf_grid_vals, dtype=torch.float32).unsqueeze(0).unsqueeze(0) 

geom = dde.geometry.Polygon([
    [x_[0], y1_[0]], [x_[-1], y1_[-1]], [x_[-1], y2_[-1]], [x_[6], y2_[6]],
    [x_[5], y2_[5]], [x_[4], y2_[4]], [x_[3], y2_[3]],
    [x_[2], y2_[2]], [x_[1], y2_[1]], [x_[0], y2_[0]]
])

inlet_x = dde.icbc.DirichletBC(geom, inlet_velocity, inlet_boundary, component=0)
inlet_y = dde.icbc.DirichletBC(geom, zero_velocity, inlet_boundary, component=1)
outlet_p = dde.icbc.DirichletBC (geom, zero_velocity, outlet_boundary, component=2)
# outlet_u_x = dde.icbc.NeumannBC (geom, zero_velocity, outlet_boundary, component=0)
# outlet_v_x = dde.icbc.NeumannBC (geom, zero_velocity, outlet_boundary, component=1)
inlet_nu = dde.icbc.DirichletBC(geom, inlet_nu_tilde, inlet_boundary, component=3)
outlet_nu_x = dde.icbc.NeumannBC(geom, zero_velocity, outlet_boundary, component=3) 
wallt_x = dde.icbc.DirichletBC(geom, zero_velocity, wall_top_boundary, component=0)
wallt_y = dde.icbc.DirichletBC(geom, zero_velocity, wall_top_boundary, component=1)
wallb_x = dde.icbc.DirichletBC(geom, zero_velocity, wall_bottom_boundary, component=0)
wallb_y = dde.icbc.DirichletBC(geom, zero_velocity, wall_bottom_boundary, component=1)
wallsh_x = dde.icbc.DirichletBC(geom, zero_velocity, wall_mid_horizontal_boundary, component=0)
wallsh_y = dde.icbc.DirichletBC(geom, zero_velocity, wall_mid_horizontal_boundary, component=1)
wallsv_x = dde.icbc.DirichletBC(geom, zero_velocity, wall_incline_boundary, component=0)
wallsv_y = dde.icbc.DirichletBC(geom, zero_velocity, wall_incline_boundary, component=1)
wallt_nu = dde.icbc.DirichletBC(geom, zero_velocity, wall_top_boundary, component=3)
wallb_nu = dde.icbc.DirichletBC(geom, zero_velocity, wall_bottom_boundary, component=3)
wallh_nu = dde.icbc.DirichletBC(geom, zero_velocity, wall_mid_horizontal_boundary, component=3)
wall_nu_incline = dde.icbc.DirichletBC(geom, zero_velocity, wall_incline_boundary, component=3)

#add more points near step  
x_d = np.linspace(0.75, 5.0, 50)
y_d = np.linspace(-0.25, 0.2, 40)
X, Y = np.meshgrid(x_d, y_d)
points = np.vstack((X.ravel(), Y.ravel())).T
inside = geom.inside(points)
points = points[inside]  

data = dde.data.PDE(
    geom,
    navier_stokes_with_SA,
    [inlet_x, inlet_y, outlet_p, wallt_x, wallt_y, wallb_x, wallb_y, wallsh_x, wallsh_y, wallsv_x, wallsv_y, inlet_nu, wallt_nu, wallb_nu, wallh_nu, wall_nu_incline, outlet_nu_x],
    train_distribution = "LHS",
    num_domain=5000,
    num_boundary=4000,
    num_test=5000,
    anchors=points
)

labels = ["Continuity", "X Momentum", "Y  Momentum", "Nu Transport", "Inlet X", "Inlet Y", "Outlet P", \
            "Wall Top X", "Wall Top Y", "Wall Bottom X", "Wall Bottom Y", "Wall Step H X", "Wall Step H Y", "Wall Step Incline X", "Wall Step Incline Y",\
            "Inlet Nu", "Wall Nu Top", "Wall Nu Bottom", "Wall Nu horizontal", "Wall Nu Incline", "Outlet Neumann Nu"]

# layer_size = [256] + [256] * 3 + [4]
layer_size = [256] + [128] * 6 + [4]
net = dde.nn.FNN(layer_size, "swish", "Glorot uniform")
net.apply_feature_transform(input_transform)
net.apply_output_transform(output_transform)
prev_net = copy.deepcopy(net)
# loss_weights = [
#     1,     # continuity
#     10,    # x_momentum
#     10,    # y_momentum
#     5,     # nu_tilde_transport
#     50,    # inlet_x
#     20,    # inlet_y
#     5,     # outlet_p
#     50,    # wall_x
#     50,    # wall_y
#     50,    # wall_x
#     50,    # wall_y
#     50,    # wall_x
#     50,    # wall_y
#     50,    # wall_x
#     50,    # wall_y
#     20,    # inlet_nu
#     20,    # wall_nu
#     20,    # wall_nu
#     20,    # wall_nu
#     20,    # wall_nu
#     5,     # outlet_nu_x (Neumann)
# ]

loss_weights = [
    1,     # continuity
    1,    # x_momentum
    1,    # y_momentum
    1,     # nu_tilde_transport
    1,    # inlet_x
    1,    # inlet_y
    1,     # outlet_p
    # 1,
    # 1,
    1,    # wall_x
    1,    # wall_y
    1,    # wall_x
    1,    # wall_y
    1,    # wall_x
    1,    # wall_y
    1,    # wall_x
    1,    # wall_y
    1,    # inlet_nu
    1,    # wall_nu
    1,    # wall_nu
    1,    # wall_nu
    1,    # wall_nu
    1,     # outlet_nu_x (Neumann)
]

model = dde.Model(data, net)
plot_contour = PlotCallback(period=1000)
plot_residual = PlotCallback1(period=5000)
plot_loss = LossPlotCallback(period=5000, labels=labels)
update_prev_net = UpdatePrevNet(period=1)
gradnorm_cb = GradNormCallback(period=1000, lr=1e-3)
early_stopping = dde.callbacks.EarlyStopping(min_delta=1e-8, patience=5000)
checkpoint_cb = dde.callbacks.ModelCheckpoint(filepath=f"{folder_name}/ckpt/model", save_better_only=False, period=1000)
resampler = dde.callbacks.PDEPointResampler(period=1000)
re_ramp = ReRampCallback([(0, 300), (8000, 1000), (16000, 3300), (24000, 11000), (32000, 22000), (40000, 36000)])
# Re = 36000
stage_callbacks = [plot_contour, resampler, update_prev_net, gradnorm_cb, plot_loss, re_ramp, plot_residual, checkpoint_cb]
model.compile("adam", lr=1e-3, decay=("step", 2000, 0.9), loss_weights=loss_weights)
# model.restore("plots_20260822_204432_inc_isolated_soft_modified/ns_model-75000.pt")
losshistory, train_state = model.train(iterations=75000,  display_every=10, callbacks=stage_callbacks, disregard_previous_best=False, model_save_path=f"{folder_name}/ns_model")
model.compile("L-BFGS")
# model.restore("plots_20260822_011658_inc_isolated_soft_modified/ns_model_final-90699.pt")
losshistory, train_state = model.train(iterations=5000, callbacks=[update_prev_net, plot_contour, plot_residual], display_every=10, model_save_path=f"{folder_name}/ns_model_final")
np.save(f"{folder_name}/train_loss.npy", np.array(losshistory.loss_train))
np.save(f"{folder_name}/test_loss.npy", np.array(losshistory.loss_test))
np.save(f"{folder_name}/steps.npy", np.array(losshistory.steps))
################################################################################################################################################################

# log(f"Step after restore: {model.train_state.step}\n")
# #initialize initial error 
# X = geom.random_points(100000)
# f = model.predict(X, operator=navier_stokes)
# err_eq = np.abs(f[0]) + np.abs(f[1]) + np.abs(f[2])  
# err_eq = err_eq.flatten()
# initial_err = np.mean(err_eq)

# #RAR
# threshold = 0.15 * initial_err 
# X = geom.random_points(100000)
# err = initial_err
# itr_ = 0
# while err > threshold and itr_ <= 5:
#     f = model.predict(X, operator=navier_stokes)
#     err_eq = np.absolute(f)
#     err_eq = np.abs(f[0]) + np.abs(f[1]) + np.abs(f[2])  
#     err_eq = err_eq.flatten()
#     err = np.mean(err_eq)
#     print("Mean residual: %.3e" % (err))

#     if err <= threshold:
#         print("Target residual achieved!")
#         break
        
#     n_add = 50
#     worst_idx = np.argsort(err_eq)[-n_add:]
#     print("Adding new points:", X[worst_idx], "\n")
#     data.add_anchors(X[worst_idx])
#     early_stopping = dde.callbacks.EarlyStopping(min_delta=1e-4, patience=1000)
#     plot_contour = PlotCallback(period=1000)
#     model.compile("adam", lr=lr)
#     losshistory, train_state = model.train(iterations=3000, callbacks=[early_stopping, plot_contour], display_every=10, disregard_previous_best=True, model_save_path=f"{folder_name}/ns_model")
#     log(f"Step after RAR_{itr_}:{model.train_state.step}\n")
#     itr_ += 1

# model.compile("L-BFGS")
# losshistory, train_state = model.train(iterations=5000, callbacks=[plot_contour], display_every=10, model_save_path=f"{folder_name}/ns_model_final")
# log(f"Step after LBFGS: {model.train_state.step}\n")

dde.saveplot(losshistory, train_state, issave=True, isplot=True)
#########################################################################
cv1 = 7.1
thetalist = [45]
for theta_plot in thetalist:
    f = open(f"{folder_name}/details_theta{theta_plot}.txt", 'w')
    x_dense = np.linspace(x_[0], x_[-1], 500)
    y_dense = np.linspace(y2_[-1], y1_[0], 200)
    X, Y = np.meshgrid(x_dense, y_dense)
    points = np.hstack((X.ravel()[:, None], Y.ravel()[:, None]))
    inside_mask = geom.inside(points)
    prediction = model.predict(points)
    if isinstance(prediction, tuple) or isinstance(prediction, list):
        prediction = prediction[0]
    u_component = prediction[:, 0] 
    v_component = prediction[:, 1]
    p           = prediction[:,2]
    nu_tilde    = prediction[:,3]
    u_mag_flat = np.sqrt(u_component**2 + v_component**2)
    u_mag_flat[~inside_mask] = np.nan
    U_mag = u_mag_flat.reshape(X.shape)

    plt.figure(figsize=(10, 3), dpi=300)
    contour = plt.contourf(X, Y, U_mag, levels=200, cmap='Spectral_r', antialiased=True)
    cbar = plt.colorbar(contour, orientation='horizontal', pad=0.4, shrink=0.6)
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    cbar.set_label('U Magnitude', fontsize=10)
    plt.xlim(x_[0] - 0.1, x_[-1] + 0.1)
    plt.ylim(y2_[-1] - 0.1, y1_[0] + 0.1)
    # plt.gca().set_aspect('equal')
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel("X coord.", fontsize=10)
    plt.ylabel("Y coord.", fontsize=10)
    plt.title(f"Velocity Field (theta={theta_plot} deg)", fontsize=10)
    plt.savefig(f"{folder_name}/velContour_theta{theta_plot}.png", bbox_inches='tight', dpi=300)
    plt.close()

    u_component[~inside_mask] = np.nan
    v_component[~inside_mask] = np.nan
    u_component = u_component.reshape(X.shape)
    v_component = v_component.reshape(X.shape)
    plt.figure(figsize=(10, 3), dpi=300)
    contour = plt.contourf(X, Y, U_mag, levels=200, cmap='Spectral_r', antialiased=True)
    strm = plt.streamplot(
        X, Y, u_component, v_component,
        color='white',
        density=3.0,      
        linewidth=0.3,     
        arrowsize=0.5,      
        minlength=0.05,
    )
    cbar = plt.colorbar(contour, orientation='horizontal', pad=0.4, shrink=0.6)
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    cbar.set_label('U Magnitude', fontsize=10)
    plt.xlim(x_[0] - 0.1, x_[-1] + 0.1)
    plt.ylim(y2_[-1] - 0.1, y1_[0] + 0.1)
    # plt.gca().set_aspect('equal')
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel("X coord.", fontsize=10)
    plt.ylabel("Y coord.", fontsize=10)
    umax = np.nanmax(U_mag)
    plt.title(f"Velocity Field max:{umax}", fontsize=10)
    # plt.box(False)
    plt.savefig(f"{folder_name}/velStream_theta{theta_plot}.png", bbox_inches='tight', dpi=300)


    plt.figure(figsize=(10, 3), dpi=300)
    p[~inside_mask] = np.nan
    P_mag = p.reshape(X.shape)
    contour = plt.contourf(X, Y, P_mag, levels=200, cmap='Spectral_r', antialiased=True)
    cbar = plt.colorbar(contour, orientation='horizontal', pad=0.4, shrink=0.6)
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    cbar.set_label('P Magnitude', fontsize=10)
    plt.xlim(x_[0] - 0.1, x_[-1] + 0.1)
    plt.ylim(y2_[-1] - 0.1, y1_[0] + 0.1)
    # plt.gca().set_aspect('equal')
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel("X coord.", fontsize=10)
    plt.ylabel("Y coord.", fontsize=10)
    plt.title("Pressure Field", fontsize=10)
    plt.savefig(f"{folder_name}/pres_contour_theta{theta_plot}.png", bbox_inches='tight', dpi=300)

    ##### Turbulence Intensity using Bradshaw constant #####
    points_t = torch.tensor(points, dtype=torch.float32, requires_grad=True)
    prediction_t = model.net(points_t) 
    v_y = dde.grad.jacobian(prediction_t, points_t, i=1, j=1)
    u_x = dde.grad.jacobian(prediction_t, points_t, i=0, j=0)
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

    plt.figure(figsize=(10, 3), dpi=300)
    contour = plt.contourf(X, Y, nu_t_mag, levels=200, cmap='Spectral_r', antialiased=True)
    cbar = plt.colorbar(contour, orientation='horizontal', pad=0.4, shrink=0.6)
    cbar.ax.tick_params(labelsize=7)
    cbar.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1e'))
    cbar.set_label(r'$\nu_t$ Magnitude', fontsize=10)
    plt.xlim(x_[0] - 0.1, x_[-1] + 0.1)
    plt.ylim(y2_[-1] - 0.1, y1_[0] + 0.1)
    # plt.gca().set_aspect('equal')
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel("X coord.", fontsize=10)
    plt.ylabel("Y coord.", fontsize=10)
    plt.title(rf"$\nu_t$ Field (theta={theta_plot} deg, max {numax})", fontsize=10)
    plt.savefig(f"{folder_name}/nu_t.png", bbox_inches='tight', dpi=300)
    plt.close()

    plt.figure(figsize=(10, 3), dpi=300)
    contour = plt.contourf(X, Y, norm_nu_t_mag, levels=200, cmap='Spectral_r', antialiased=True)
    cbar = plt.colorbar(contour, orientation='horizontal', pad=0.4, shrink=0.6)
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1e'))
    cbar.set_label('Nu_t/Nu Magnitude', fontsize=10)
    plt.xlim(x_[0] - 0.1, x_[-1] + 0.1)
    plt.ylim(y2_[-1] - 0.1, y1_[0] + 0.1)
    # plt.gca().set_aspect('equal')
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel("X coord.", fontsize=10)
    plt.ylabel("Y coord.", fontsize=10)
    plt.title(f"Normalized Eddy viscosity field Nu_t/Nu (theta={theta_plot} deg, max {norm_numax})", fontsize=10)
    plt.savefig(f"{folder_name}/nu_t_normalized.png", bbox_inches='tight', dpi=300)
    plt.close()

    u_u = 2 * nu_t * (u_x)
    v_v = 2 * nu_t * (v_y)
    #Bradshaw constant = 0.3  
    k = 0.5 * (u_u + v_v)
    k = torch.abs(k) 
    U_ref = 1
    TI = torch.sqrt(2/3 * k) / U_ref
    TI = TI * 100

    TI_np = TI.detach().cpu().numpy().flatten()
    TI_np[~inside_mask] = np.nan
    TI_mag = TI_np.reshape(X.shape)
    TImax = np.nanmax(TI_mag)

    plt.figure(figsize=(10, 3), dpi=300)
    contour = plt.contourf(X, Y, TI_mag, levels=200, cmap='Spectral_r', antialiased=True)
    cbar = plt.colorbar(contour, orientation='horizontal', pad=0.4, shrink=0.6)
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    cbar.set_label('Turbulence Intensity %', fontsize=10)
    plt.xlim(x_[0] - 0.1, x_[-1] + 0.1)
    plt.ylim(y2_[-1] - 0.1, y1_[0] + 0.1)
    # plt.gca().set_aspect('equal')
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel("X coord.", fontsize=10)
    plt.ylabel("Y coord.", fontsize=10)
    plt.title(f"Turbulence Intensity % (theta={theta_plot} deg, max {TImax})", fontsize=10)
    plt.savefig(f"{folder_name}/Turbulence_Intensity.png", bbox_inches='tight', dpi=300)
    plt.close()
    ##########################################################

    # sample u very close to the wall, downstream of the step
    y_wall = y2_[-1] + 0.02
    x_scan = np.linspace(x_[4], Ld, 100)
    pts = np.column_stack((x_scan, np.full_like(x_scan, y_wall)))
    u_wall = model.predict(pts)[:, 0]
    # reattachment point ≈ where u_wall crosses from negative to positive
    sign_change = np.where(np.diff(np.sign(u_wall)) > 0)[0]
    # x_reattach = x_scan[sign_change[0]] if len(sign_change) else None
    # slope of a line where y=0, and find x where it crosses 0
    reattach_candidates = []
    for i in sign_change:
        x1, x2 = x_scan[i], x_scan[i+1]
        u1, u2 = u_wall[i], u_wall[i+1]
        x_reattach_i = x1 - u1*(x2 - x1)/(u2-u1)
        reattach_candidates.append((x_reattach_i, x1, x2, u1, u2))

    f.write(f"Number of sign changes (neg->pos) found: {len(reattach_candidates)}\n")
    for idx, (x_r, x1, x2, u1, u2) in enumerate(reattach_candidates):
        f.write(f"  Candidate {idx}: x_reattach={x_r:.5f}  (between x={x1:.5f} [u={u1:.5f}] and x={x2:.5f} [u={u2:.5f}])\n")

    if reattach_candidates:
        x_reattach = reattach_candidates[0][0] 
    else:
        x_reattach = None

    f.write(f"Reattachment length (first candidate) = {x_reattach}\n")

    if Re == 36000:
        #plot velocity profiles
        plt.figure(figsize=(14, 7))
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

            if loc == 0:
                y_fine = np.linspace(y2_[3], y1_[0], 100)
            else:
                y_fine = np.linspace(y2_[6], y1_[0], 200)   
            x_fine = np.full(y_fine.shape[0], step_x + step_h * loc)
            grid_fine = np.column_stack((x_fine, y_fine))
            u_pred_fine = model.predict(grid_fine)[:, 0]     
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
            y_plot_scaled = (y_data * step_h) + y2_[6]
            # plt.axvline(x=loc, color=colors[i], linestyle='--', alpha=0.5)
            label = "Makiola et al. Experiment" if i == 0 else ""
            plt.scatter(x_plot_data, y_plot_scaled, label=label, color='gray', marker='o')

            x_coord_new = np.full(y_data.shape[0], step_x + step_h * loc)
            onedgrid = np.column_stack((x_coord_new, y_plot_scaled))
            u_pred_for_exp = model.predict(onedgrid)[:, 0]

            diff = u_pred_for_exp - u_over_U0
            l2_num_exp = np.sqrt(np.sum(diff**2))
            l2_den_exp = np.sqrt(np.sum(u_over_U0**2))
            l2_rel_exp = (l2_num_exp / l2_den_exp) * 100
            total_num_exp += np.sum(diff**2)
            total_den_exp += np.sum(u_over_U0**2)
            mae_exp = np.mean(np.abs(diff)) 
            max_ae_exp = np.max(np.abs(diff)) 
            u_mean_exp = np.mean(np.abs(u_over_U0))
            # mae_exp_pct = (mae_exp / u_mean_exp) * 100 
            # max_ae_exp_pct = (max_ae_exp / u_mean_exp) * 100
            #change to ref value as 1 
            mae_exp_pct = (mae_exp) * 100 
            max_ae_exp_pct = (max_ae_exp) * 100
            #######################################

            # fem########################
            file_path = f'makiola1/fem/loc{loc}.csv'
            my_data = np.genfromtxt(file_path, delimiter=',')
            x_global = my_data[:, 0]
            y_data = my_data[:, 1]

            u_over_U0 = (x_global - loc) / k_
            x_plot_data = loc + u_over_U0
            y_plot_scaled = (y_data * step_h) + y2_[6]
            # plt.axvline(x=loc, color=colors[i], linestyle='--', alpha=0.5)
            label = "FVM Louda et al." if i == 0 else ""
            plt.plot(x_plot_data, y_plot_scaled, label=label, color=colors[i], linestyle='--')

            x_coord_new = np.full(y_data.shape[0], step_x + step_h * loc)
            onedgrid = np.column_stack((x_coord_new, y_plot_scaled))
            u_pred_for_fem = model.predict(onedgrid)[:, 0]

            diff1 = u_pred_for_fem - u_over_U0
            l2_num_fem = np.sqrt(np.sum(diff1**2))
            l2_den_fem = np.sqrt(np.sum(u_over_U0**2))
            l2_rel_fem = (l2_num_fem / l2_den_fem) * 100
            total_num_fem += np.sum(diff1**2)
            total_den_fem += np.sum(u_over_U0**2)
            mae_fem = np.mean(np.abs(diff1)) 
            max_ae_fem = np.max(np.abs(diff1)) 
            u_mean_fem = np.mean(np.abs(u_over_U0)) 
            # mae_fem_pct = (mae_fem / u_mean_fem) * 100 
            # max_ae_fem_pct = (max_ae_fem / u_mean_fem) * 100
            #change to ref value as 1 
            mae_fem_pct = (mae_fem) * 100 
            max_ae_fem_pct = (max_ae_fem) * 100
            #######################################

            f.write(f"Station x/S={loc}: EXP MAE ={mae_exp:.5f}, MAE % ={mae_exp_pct:.5f} % | FEM MAE={mae_fem:.5f}, MAE % ={mae_fem_pct:.5f} %\n")
            f.write(f"Station x/S={loc}: EXP MaxAE ={max_ae_exp:.5f}, MaxAE % ={max_ae_exp_pct:.5f} % | FEM MaxAE={max_ae_fem:.5f}, MaxAE % ={max_ae_fem_pct:.5f} %\n")
            f.write(f"Station x/S={loc}: EXP abs L2={l2_num_exp:.5f}, rel L2={l2_rel_exp:.5f} % | FEM abs L2={l2_num_fem:.5f}, rel L2={l2_rel_fem:.5f} %\n")
        overall_rel_l2_exp = ( np.sqrt(total_num_exp) / np.sqrt(total_den_exp) ) * 100
        overall_rel_l2_fem = ( np.sqrt(total_num_fem) / np.sqrt(total_den_fem) ) * 100
        f.write(f"Overall relative L2 (experiment): {overall_rel_l2_exp:.5f} %\n")
        f.write(f"Overall relative L2 (FEM): {overall_rel_l2_fem:.5f} %\n")
        f.close()

        plt.xlabel('x/H')
        plt.ylabel('y')
        plt.grid(True, linestyle=':')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title("Velocity Profiles")
        plt.savefig(f"{folder_name}/velocity_profile_Re{Re}.png", bbox_inches='tight')
#############################################################
#3#Loss Curve
plt.figure(figsize=(10, 6))
# labels = ["Continuity", "X Momentum", "Y  Momentum", "Nu Transport", "Inlet X", "Inlet Y", "Outlet Neumann U", "Outlet Neumann V", "Outlet P", \
#             "Wall Top X", "Wall Top Y", "Wall Bottom X", "Wall Bottom Y", "Wall Step H X", "Wall Step H Y", "Wall Step Incline X", "Wall Step Incline Y", \
#             "Inlet Nu", "Wall Nu Top", "Wall Nu Bottom", "Wall Nu horizontal", "Wall Nu Incline", "Outlet Neumann Nu"]
# train_loss = np.array(losshistory.loss_train)
# steps = np.array(losshistory.steps)
train_loss = np.array(losshistory.loss_train)
steps = np.array(losshistory.steps)
plt.gca().set_prop_cycle(color=plt.cm.tab20(np.linspace(0, 1, 14)))
for i in range(train_loss.shape[1]):
    temp_loss = train_loss[:,i]
    ref_loss = temp_loss[0]
    normalized_loss = temp_loss/ref_loss
    plt.semilogy(steps, normalized_loss+ 1e-10, label=labels[i])

# Labels and Style
plt.xlabel("Iterations")
plt.ylabel("Normalized Loss ($L / L_{initial}$)")
plt.title("Training Loss")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
plt.grid(True, which="both", ls="-", alpha=0.3)
plt.tight_layout()

plt.savefig(f"{folder_name}/training_loss.png", dpi=150, bbox_inches="tight")

#Test loss
plt.figure(figsize=(10, 6))
test_loss = np.array(losshistory.loss_test)
steps = np.array(losshistory.steps)

for i in range(test_loss.shape[1]):
    temp_loss = test_loss[:,i]
    ref_loss = temp_loss[0]
    normalized_loss = temp_loss/ref_loss
    plt.semilogy(steps, normalized_loss+ 1e-10, label=labels[i])

# Labels and Style
plt.xlabel("Iterations")
plt.ylabel("Normalized Loss ($L / L_{initial}$)")
plt.title("Test Loss")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
plt.grid(True, which="both", ls="-", alpha=0.3)
plt.tight_layout()

plt.savefig(f"{folder_name}/test_loss.png", dpi=150, bbox_inches="tight")
#############################################################
#############################################################

fh.close()
fr.close()