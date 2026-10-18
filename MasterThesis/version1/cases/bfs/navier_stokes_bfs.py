import os
import json
import glob
import deepxde as dde
from deepxde.backend import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
from datetime import datetime
from numpy import genfromtxt
import copy
import matplotlib.ticker as ticker
import shutil
# dde.config.set_default_float("float64")
# torch.set_default_dtype(torch.float64)

SEED = 1
np.random.seed(SEED)
torch.manual_seed(SEED)
dde.config.set_random_seed(SEED)

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
            residuals = navier_stokes(x_t, y_t)  
            R_i = residuals[i]
            L_i = torch.mean(R_i ** 2)                     
            L_i.backward(retain_graph=False)

            total_norm_sq = sum(p.grad.norm()**2 for p in net.parameters() if p.grad is not None)
            grad_norms.append(torch.sqrt(total_norm_sq).item())

        bc_list = [inlet_x, inlet_y, outlet_p, wallt_x, wallt_y, wallb_x, wallb_y,
                   wallsh_x, wallsh_y, wallsv_x, wallsv_y]

        for bc in bc_list:
            net.zero_grad()
            bc_pts = bc.filter(geom.random_boundary_points(2000))  
            # print(f"  {bc} -> {bc_pts.shape[0]} points")
            if bc_pts.shape[0] == 0:
                grad_norms.append(1e-8)  
                continue
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
        for i in range(len(grad_norms)):
            # self.model.loss_weights[i] = (1 - self.momentum) * self.model.loss_weights[i] + self.momentum * lambda_hat[i]
            self.model.loss_weights[i] = self.momentum * self.model.loss_weights[i] +  (1 - self.momentum) * lambda_hat[i]

        print(f"[iter {iteration}] grad_norms={grad_norms}  new_lambdas={model.loss_weights}")

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

class UpdatePrevNet(dde.callbacks.Callback):
    def __init__(self, period=10):
        super().__init__()
        self.period = period
 
    def on_epoch_end(self):
        global prev_net
        epoch = self.model.train_state.epoch
        if epoch % self.period == 0 and epoch > 0:
            prev_net = copy.deepcopy(self.model.net)

class PlotCallback1(dde.callbacks.Callback):
    def __init__(self, period=100):
        super().__init__()
        self.period = period

    def on_epoch_end(self):
        epoch = self.model.train_state.epoch
        # self.on_epoch_end runs at every iteration step
        if epoch % self.period == 0:
            residual_plotting_at_checkpoints(self.model, epoch)

class PlotCallback(dde.callbacks.Callback):
    def __init__(self, period=100):
        super().__init__()
        self.period = period

    def on_epoch_end(self):
        epoch = self.model.train_state.epoch
        # self.on_epoch_end runs at every iteration step
        if epoch % self.period == 0:
            contour_plotting_at_checkpoints(self.model, epoch)

def contour_plotting_at_checkpoints(model, epoch):
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
    plt.savefig(f"{folder_name}/iteration_{epoch}_comparison_Re{Re}.png", bbox_inches='tight', dpi=300)
    plt.close()

def residual_plotting_at_checkpoints(model, epoch):
    Re = 183.3
    x_dense = np.linspace(0.0, 30.0, 500)
    y_dense = np.linspace(0.0, 2.061, 200)
    X, Y = np.meshgrid(x_dense, y_dense)
    points = np.vstack((X.ravel(), Y.ravel())).T
    inside_mask = geom.inside(points)
    residuals = model.predict(points, operator=navier_stokes)
    labels = ["continuity", "x-mom", "y-mom"]
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
        plt.xlim(-0.1, 30.1)
        plt.ylim(-0.1, 2.1)
        plt.gca().set_aspect('equal')
        plt.title(f"{label} residual (epoch {epoch})", fontsize=9)
        plt.savefig(f"{folder_name}/residual_{label}_epoch{epoch}.png", bbox_inches='tight', dpi=300)
        plt.close()
    fr.write("******************************************************************\n")
    fr.flush()

def wall_top_boundary(x, on_boundary):
    return on_boundary and np.isclose(x[1], 2.061)


def wall_bottom_boundary(x, on_boundary):
    return on_boundary and np.isclose(x[1], 0.0)


def wall_mid_horizontal_boundary(x, on_boundary):
    return on_boundary and (np.isclose(x[1], 1) and x[0] < 2.0)


def wall_mid_vertical_boundary(x, on_boundary):
    return on_boundary and (x[1] < 1 and np.isclose(x[0], 2.0))


def outlet_boundary(x, on_boundary):
    return on_boundary and np.isclose(x[0], 30.0)


def inlet_boundary(x, on_boundary):
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

def input_transform(x):
    x_scaled = (x[:,0:1]-x_min)/(x_max-x_min)
    y_scaled = (x[:,1:2]-y_min)/(y_max-y_min)
    xy = torch.cat([x_scaled, y_scaled], dim=1)
    proj = xy @ B.T
    # proj returns N,128 -> N, 2 and 2, 128
    # returns N, 256 due to N,128 and N, 128 stacked side by side , hence input is 256 for each point
    return torch.cat([torch.cos(proj), torch.sin(proj)], dim=1) 
    # return torch.cat([x_scaled, y_scaled], dim=1) 

def output_transformer(inputs, outputs, k4=8.0, k5=8.0, k3=4.0):

    x = inputs[:, 0]
    y = inputs[:, 1]
    u_raw, v_raw, p_raw = outputs[:, 0], outputs[:, 1], outputs[:, 2]

    ones = torch.ones_like(x)

    dist_top = 2.061 - y
    dist_bottom = torch.where(x >= 2.0, y, ones)
    dist_step_hor = torch.where(x <= 2.0, y - 1.0, ones)
    dist_step_ver = torch.where(y <= 1.0, x - 2.0, ones)

    wall_multiplier = dist_top * dist_bottom * dist_step_hor * dist_step_ver

    u_final = wall_multiplier * u_raw
    v_final = wall_multiplier * v_raw
    p_final = p_raw

    return torch.stack((u_final, v_final, p_final), dim=1)

def navier_stokes(x, y):
    
    u, v, p = y[:, 0:1], y[:, 1:2], y[:, 2:3]
    
    ########pseudo time stepping##################
    time_step = 10
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

n=3

def log(msg):
    with open(f"{folder_name}/details.txt", "a") as f:
        f.write(msg + "\n")
x_min, x_max = 0, 30
y_min, y_max = 0, 2.061

run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
folder_name = f"../finalresults/bfs/plots_{run_time}_step_isolated_soft"
os.makedirs(folder_name, exist_ok=True)
os.makedirs(f"{folder_name}/ckpt", exist_ok=True)
fr = open(f"{folder_name}/residual_values.txt", 'a')
shutil.copy(__file__, os.path.join(folder_name, "script_snapshot.py"))

plt.rcParams.update({
    'font.size': 16,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
})

Re = 183.3
itr = 75000

B = torch.randn(128, 2) * 2.0

geom = dde.geometry.Polygon([
    [0.0, 2.061], [30.0, 2.061], [30.0, 0.0], [2.0, 0.0], [2.0, 1.0],
    [0.0, 1.0]
])
# geom_param = dde.geometry.TimeDomain(re_min, re_max)
# geom_total = dde.geometry.GeometryXTime(geom, geom_param)

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

#add more points near step 

x_d = np.linspace(2.0, 30.0, 200)
y_d = np.linspace(0.0, 1.0, 100)
X, Y = np.meshgrid(x_d, y_d)
points = np.vstack((X.ravel(), Y.ravel())).T

# loss_weights = [
# 1,     # continuity
# 10,    # x_momentum
# 10,    # y_momentum
# 50,    # inlet_x
# 20,    # inlet_y
# 5,     # outlet_p
# # 1,     # outlet_u_x (Neumann)
# # 1,     # outlet_v_x (Neumann)
# 50,    # wall_x
# 50,    # wall_y
# 50,    # wall_x
# 50,    # wall_y
# 50,    # wall_x
# 50,    # wall_y
# 50,    # wall_x
# 50,    # wall_y
# ]


loss_weights = [
1,     # continuity
1,    # x_momentum
1,    # y_momentum
1,    # inlet_x
1,    # inlet_y
1,     # outlet_p
# 1,     # outlet_u_x (Neumann)
# 1,     # outlet_v_x (Neumann)
1,    # wall_x
1,    # wall_y
1,    # wall_x
1,    # wall_y
1,    # wall_x
1,    # wall_y
1,    # wall_x
1,    # wall_y
]

data = dde.data.PDE(
    geom, 
    navier_stokes,
    [inlet_x, inlet_y, outlet_p, wallt_x, wallt_y, wallb_x, wallb_y, wallsh_x, wallsh_y, wallsv_x, wallsv_y], 
    train_distribution="LHS",
    num_domain=5000,
    num_boundary=4000, 
    num_test=5000,
    anchors = points
)


layer_size = [256] + [64] * 6 + [3]
net = dde.nn.FNN(layer_size, "swish", "Glorot uniform")
# net.apply_output_transform(output_transformer)
net.apply_feature_transform(input_transform)
prev_net = copy.deepcopy(net)
model = dde.Model(data, net)
update_prev_net = UpdatePrevNet(period=1)
plot_residual = PlotCallback1(period=5000)
plot_contour = PlotCallback(period=1000)
gradnorm_cb = GradNormCallback(period=1000, lr=1e-3)
checkpoint_cb = dde.callbacks.ModelCheckpoint(filepath=f"{folder_name}/ckpt/model", save_better_only=False, period=1000)
resampler = dde.callbacks.PDEPointResampler(period=1000)
model.compile("adam", lr=1e-3, decay=("step", 2000, 0.9), loss_weights=loss_weights)
losshistory, train_state = model.train(iterations=50000,  display_every=10, callbacks=[gradnorm_cb, update_prev_net, plot_contour, resampler, plot_residual], disregard_previous_best=False, model_save_path=f"{folder_name}/ns_model")
model.compile("L-BFGS")
losshistory, train_state = model.train(iterations=2000, callbacks=[plot_contour, update_prev_net, plot_residual], display_every=10, model_save_path=f"{folder_name}/ns_model_final")
np.save(f"{folder_name}/train_loss.npy", np.array(losshistory.loss_train))
np.save(f"{folder_name}/test_loss.npy", np.array(losshistory.loss_test))
np.save(f"{folder_name}/steps.npy", np.array(losshistory.steps))

dde.saveplot(losshistory, train_state, issave=True, isplot=True)
#########################################################################
Relist = [183.3]
x_dense = np.linspace(0.0, 30.0, 500)
y_dense = np.linspace(0.0, 2.061, 200)
X, Y = np.meshgrid(x_dense, y_dense)
points = np.vstack((X.ravel(), Y.ravel())).T
for Re in Relist:
    f = open(f"{folder_name}/details.txt", 'w')
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
    plt.figure(figsize=(10, 3), dpi=300)
    contour = plt.contourf(X, Y, U_mag, levels=200, cmap='Spectral_r', antialiased=True)
    cbar = plt.colorbar(contour, orientation='horizontal', pad=0.4, shrink=0.6)
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    cbar.set_label('U Magnitude', fontsize=10)
    plt.xlim(-0.1, 30.1)
    plt.ylim(-0.1, 2.1)
    # plt.gca().set_aspect('equal')
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel("X coord.", fontsize=10)
    plt.ylabel("Y coord.", fontsize=10)
    plt.title("Velocity Field", fontsize=10)
    # plt.box(False)
    plt.savefig(f"{folder_name}/vel_contour_Re{Re}.png", bbox_inches='tight', dpi=300)


    u_component = u_component.reshape(X.shape)
    v_component = v_component.reshape(X.shape)
    u_component[(X < 2.0) & (Y < 1.0)] = np.nan
    v_component[(X < 2.0) & (Y < 1.0)] = np.nan
    plt.figure(figsize=(10, 3), dpi=300)
    contour = plt.contourf(X, Y, U_mag, levels=200, cmap='Spectral_r', antialiased=True)
    plt.streamplot(
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
    plt.xlim(-0.1, 30.1)
    plt.ylim(-0.1, 2.1)
    # plt.gca().set_aspect('equal') 
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel("X coord.", fontsize=10)
    plt.ylabel("Y coord.", fontsize=10)
    plt.title("Velocity Field", fontsize=10)
    plt.savefig(f"{folder_name}/vel_contour_stream_Re{Re}.png", bbox_inches='tight', dpi=300)

    plt.figure(figsize=(10, 3), dpi=300)
    P_mag = p.reshape(X.shape)
    P_mag[(X < 2.0) & (Y < 1.0)] = np.nan
    contour = plt.contourf(X, Y, P_mag, levels=200, cmap='Spectral_r', antialiased=True)
    cbar = plt.colorbar(contour, orientation='horizontal', pad=0.4, shrink=0.6)
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    cbar.set_label('P Magnitude', fontsize=10)
    plt.xlim(-0.1, 30.1)
    plt.ylim(-0.1, 2.1)
    # plt.gca().set_aspect('equal')
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel("X coord.", fontsize=10)
    plt.ylabel("Y coord.", fontsize=10)
    plt.title("Pressure Field", fontsize=10)
    plt.savefig(f"{folder_name}/pres_contour_Re{Re}.png", bbox_inches='tight', dpi=300)

    if Re == 183.3:
        #plot velocity profiles
        plt.figure(figsize=(10, 2))
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
            u_pred = model.predict(onedgrid) 
            u_pred = u_pred[:,0]
            y_plot_scaled = y_coord_new * 4.9
            # x_plot_data = loc + (u_pred / U0)
            x_plot_data = loc + u_pred
            plt.plot(x_plot_data, y_plot_scaled, color=colors[i], label=f"PINN z'/S = {loc}")
            wodata = np.genfromtxt(f"../plots_20260916_191025_step_isolated_soft_wo_features/plots_20260919_001903_step_isolated_soft/pinn_pred_loc{loc}.csv", delimiter=",", skip_header=1)
            x_other, y_other = wodata[:, 0], wodata[:, 1]
            plt.plot(x_other, y_other, color=colors[i], linestyle='--', label=f"PINN (w/o features) z'/S = {loc}")
            file_path = f'armaly1/loc{loc}.csv'
            my_data = np.genfromtxt(file_path, delimiter=',')
            armaly_x = my_data[:,0]
            armaly_y = my_data[:,1]
            armaly_x_normalized = armaly_x / (87.954 * (2/3))
            armaly_x_final = loc + armaly_x_normalized
            armaly_y_scaled = armaly_y / 4.9
            # plt.axvline(x=loc, color=colors[i], linestyle='--', alpha=0.5)
            armaly_label = "Armaly et.al (1983)" if i == 0 else ""
            plt.scatter(armaly_x_final, armaly_y, label=armaly_label, color='gray', marker='o')

            x_coord_new = np.full(armaly_y_scaled.shape[0], step_x + step_h * loc)
            onedgrid = np.column_stack((x_coord_new, armaly_y_scaled))
            u_pred = model.predict(onedgrid)[:, 0]
         
            diff = u_pred - armaly_x_normalized
            l2_num = np.sqrt(np.sum(diff**2))
            l2_den = np.sqrt(np.sum(armaly_x_normalized**2))
            l2_rel = (l2_num / l2_den) * 100
         
            total_num += np.sum(diff**2)
            total_den += np.sum(armaly_x_normalized**2)
         
            f.write(f"Station x/S={loc}: abs L2 = {l2_num:.5f}, relative L2 = {l2_rel:.5f} %\n")
            mae_exp = np.mean(np.abs(diff)) 
            max_ae_exp = np.max(np.abs(diff)) 
            u_range_exp = np.max(armaly_x_normalized) - np.min(armaly_x_normalized) 
            # replace ref value as 1 as their velocity and mine is non dim
            # mae_exp_pct = (mae_exp / u_range_exp) * 100 
            # max_ae_exp_pct = (max_ae_exp / u_range_exp) * 100
            mae_exp_pct = (mae_exp) * 100 
            max_ae_exp_pct = (max_ae_exp) * 100

            f.write(f"Station x/S={loc}: EXP MAE={mae_exp:.5f} ({mae_exp_pct:.2f}%), Max AE={max_ae_exp:.5f} ({max_ae_exp_pct:.2f}%)\n")
        overall_rel_l2 = ( np.sqrt(total_num) / np.sqrt(total_den) ) * 100
        f.write(f"Overall relative L2 error: {overall_rel_l2:.5f} %\n")
     

        plt.xlabel("Adimensionalized x/S coordinate")
        plt.ylabel("y coordinate (mm)")
        plt.xlim(0, 15)
        plt.ylim(0, 10.2)  
        plt.grid(True, linestyle=':')
        legend = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title("Velocity Profiles")
        plt.savefig(f"{folder_name}/velocity_profile_Re{Re}.png", dpi=300, bbox_inches="tight", bbox_extra_artists=(legend,))

# plt.figure(figsize=(10,3), dpi=150)
# tpc = plt.tricontourf(clean_points[:, 0], clean_points[:, 1], clean_u_mag, levels=50, cmap='Spectral_r')
# step_rect = plt.Rectangle((0, 0), 2, 1, color='white', zorder=10)
# plt.gca().add_patch(step_rect)
# plt.colorbar(tpc, label="Velocity Magnitude")
# plt.axis('equal')
# plt.title("U Magnitude")
# plt.savefig(f"{folder_name}/pinn_velocity_contour.png", dpi=150)

# Create 10 test points exactly on the inlet line (x = 0)
# test_y = np.linspace(1, 2.061, 40)
# test_x = np.zeros_like(test_y)
# Re_col = np.ones_like(test_y) * 389
# inlet_points = np.vstack((test_x, test_y, Re_col)).T

# # # Force the model to predict values for these exact coordinates
# predictions = model.predict(inlet_points)
# u_velocities = predictions[:, 0]
# v_velocities = predictions[:, 1]
# u_mag_flat = np.sqrt(u_velocities**2 + v_velocities**2)

# print("\n--- ACTUAL RAW VALUES AT X = 0 ---")
# for y_val, u_val in zip(test_y, u_velocities):
#     print(f"At y = {y_val:.2f} --> True u = {u_val:.4f}")
# for y_val, v_val in zip(test_y, v_velocities):
#     print(f"At y = {y_val:.2f} --> True v = {v_val:.4f}")
# for y_val, m_val in zip(test_y, u_mag_flat):
#     print(f"At y = {y_val:.2f} -->  u mag = {m_val:.4f}")

#############################################################
# 3#Loss Curve
plt.figure(figsize=(10, 6))
labels = ["Continuity", "X Momentum", "Y  Momentum", "Inlet X", "Inlet Y", "Outlet P", "Wall Top X", "Wall Top Y", "Wall Bottom X", "Wall Bottom Y", "Wall Step H X", "Wall Step H Y", "Wall Step V X", "Wall Step V Y"]
# labels = ["Continuity", "X Momentum", "Y  Momentum", "Outlet U Neumann"]
# train_loss = np.array(losshistory.loss_train)
# steps = np.array(losshistory.steps)
train_loss = np.load("train_loss.npy")
steps = np.load("steps.npy")
for i in range(train_loss.shape[1]):
    temp_loss = train_loss[:,i]
    ref_loss = temp_loss[0]
    normalized_loss = temp_loss/ref_loss
    plt.semilogy(steps, normalized_loss+ 1e-10, label=labels[i])

# Labels and Style
plt.xlabel("Iterations")
plt.ylabel("Normalized Loss ($L / L_{initial}$)")
plt.title("Training Loss")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid(True, which="both", ls="-", alpha=0.3)
plt.tight_layout()

plt.savefig(f"{folder_name}/training_loss.png", dpi=300, bbox_inches="tight")

#Test loss
plt.figure(figsize=(10, 6))
test_loss = np.load("test_loss.npy")
steps = np.load("steps.npy")

for i in range(test_loss.shape[1]):
    temp_loss = test_loss[:,i]
    ref_loss = temp_loss[0]
    normalized_loss = temp_loss/ref_loss
    plt.semilogy(steps, normalized_loss+ 1e-10, label=labels[i])

# Labels and Style
plt.xlabel("Iterations")
plt.ylabel("Normalized Loss ($L / L_{initial}$)")
plt.title("Test Loss")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid(True, which="both", ls="-", alpha=0.3)
plt.tight_layout()

plt.savefig(f"{folder_name}/test_loss.png", dpi=300, bbox_inches="tight")
#############################################################
# sample u very close to the wall, downstream of the step
y_wall = 0.02
x_scan = np.linspace(3.0, 30.0, 500)
pts = np.column_stack((x_scan, np.full_like(x_scan, y_wall)))
u_wall = model.predict(pts)[:, 0]
# reattachment point ≈ where u_wall crosses from negative to positive
sign_change = np.where(np.diff(np.sign(u_wall)) > 0)[0]
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
#############################################################

f.close()
fr.close()