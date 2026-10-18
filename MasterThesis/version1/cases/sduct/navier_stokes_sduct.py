import deepxde as dde
from deepxde.backend import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from numpy import genfromtxt
import matplotlib.ticker as ticker
from matplotlib.ticker import FormatStrFormatter
from scipy.interpolate import griddata
from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator
import sys
import shutil
import copy

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
        # lambda_hat[3] *= 10
        lambda_hat = [min(l, 1000.0) for l in lambda_hat]
        for i in range(len(grad_norms)):
            self.model.loss_weights[i] = self.momentum * self.model.loss_weights[i] +  (1 - self.momentum) * lambda_hat[i]

        fh.write(f"SA pde grad norm={grad_norms[3]}\n")
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

class UpdatePrevNet(dde.callbacks.Callback):
    def __init__(self, period=10):
        super().__init__()
        self.period = period
 
    def on_epoch_end(self):
        global prev_net
        epoch = self.model.train_state.epoch
        if epoch % self.period == 0 and epoch > 0:
            prev_net = copy.deepcopy(self.model.net)


def plot_loss_curves(losshistory, epoch, labels):
    train_loss = np.array(losshistory.loss_train)
    test_loss = np.array(losshistory.loss_test)
    steps = np.array(losshistory.steps)

    # Train
    plt.figure(figsize=(10, 6))
    # plt.gca().set_prop_cycle(plt.cm.tab20(np.linspace(0, 1, 14)))
    plt.gca().set_prop_cycle(color=plt.cm.tab20(np.linspace(0, 1, 14)))
    for i in range(train_loss.shape[1]):
        ref_loss = train_loss[0, i]
        normalized_loss = train_loss[:, i] / ref_loss
        plt.semilogy(steps, normalized_loss + 1e-10, label=labels[i])
    plt.xlabel("Iterations")
    plt.ylabel("Normalized Loss ($L / L_{initial}$)")
    plt.title(f"Training Loss (epoch {epoch})")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{folder_name}/training_loss_epoch{epoch}.png", dpi=150, bbox_inches="tight")
    plt.close()


def contour_plotting_at_checkpoints(model, epoch):
    # Re = 1e6
    # Re = 2.5e6
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

    nu_tilde_field = np.clip(prediction[:, 3], 0.0, 0.01)
    with open(f"{folder_name}/details_iteration{epoch}.txt", "w") as f:
        f.write(f"nu_tilde min/max/mean, {nu_tilde_field.min()}, {nu_tilde_field.max()}, {nu_tilde_field.mean()} \n")
        f.write(f"u min/max/mean, {u_mag_flat1.min()}, {u_mag_flat1.max()}, {u_mag_flat1.mean()}")
        f.close()


def residual_plotting_at_checkpoints(model, epoch):
    # Re = 1e6
    # Re = 2.5e6
    x_min, x_max = complete_coords[:, 0].min(), complete_coords[:, 0].max()
    y_min, y_max = complete_coords[:, 1].min(), complete_coords[:, 1].max()
    x_range_r = np.linspace(x_min, x_max, 200)
    y_range_r = np.linspace(y_min, y_max, 200)
    xxr, yyr = np.meshgrid(x_range_r, y_range_r)
    points_r = np.vstack((xxr.ravel(), yyr.ravel())).T
    inside_mask_r = geom.inside(points_r)
    residuals = model.predict(points_r, operator=navier_stokes_with_SA)
    labels = ["continuity", "x-mom", "y-mom", "nu transport"]
    fr.write("******************************************************************\n")
    for i, label in enumerate(labels):
        res = residuals[i].flatten()
        res_abs = np.abs(res)
        res_abs[~inside_mask_r] = np.nan
        res_grid = res_abs.reshape(xxr.shape)
        res_inside = res[inside_mask_r] 
        res_mean_abs = np.mean(np.abs(res_inside))
        res_l2 = np.sqrt(np.mean(res_inside**2))
        fr.write(f"residual_mean_abs/{label} : {res_mean_abs:.4e}, residual_l2/{label} : {res_l2:.4e}\n")
        plt.figure(figsize=(10, 5), dpi=300)
        contour = plt.contourf(xxr, yyr, res_grid, levels=100, cmap='inferno', antialiased=True)
        cbar = plt.colorbar(contour, orientation='horizontal', pad=0.15, shrink=0.3)
        cbar.ax.tick_params(labelsize=5)
        cbar.set_label(rf"|{label} residual|", fontsize=8)
        plt.xlim(0.0, 1.9)
        plt.ylim(0.04, 0.71)
        plt.gca().set_aspect('equal')
        plt.title(f"{label} residual (epoch {epoch})", fontsize=9)
        plt.savefig(f"{folder_name}/residual_{label}_epoch{epoch}.png", bbox_inches='tight', dpi=300)
        plt.close()
    fr.write("******************************************************************\n")
    fr.flush()


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

def input_transform(x):
    #change this to [-1,1]
    #can change to [0,1] if the above doesnt work
    x_scaled = 2 * (x[:, 0:1] - x_min) / (x_max - x_min) - 1
    y_scaled = 2 * (x[:, 1:2] - y_min) / (y_max - y_min) - 1
    xy = torch.cat([x_scaled, y_scaled], dim=1)
    proj = xy @ B.T
    # return torch.cat([x_scaled, y_scaled], dim=1)
    return torch.cat([torch.cos(proj), torch.sin(proj)], dim=1) 

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
    return np.full((x.shape[0], 1), 5 * (1 / Re))


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


def navier_stokes_with_SA(x, y):

    x_coord, y_coord = x[:, 0:1], x[:, 1:2]
    u, v, p, nu_tilde = y[:, 0:1], y[:, 1:2], y[:, 2:3], y[:, 3:4]

    #######pseudo time stepping##################
    # time_step = 0.1
    # with torch.no_grad():
    #     prev_predicition = prev_net(x) 
    # u_prev = prev_predicition[:,0:1]
    # v_prev = prev_predicition[:,1:2]
    # p_prev = prev_predicition[:,2:3]
    # nu_tilde_prev = prev_predicition[:,3:4]
    # term1 = (u - u_prev)/time_step
    # term2 = (v - v_prev)/time_step
    # term3 = (p - p_prev)/time_step
    # term4 = (nu_tilde - nu_tilde_prev)/time_step
    # term1 = torch.zeros_like(u)
    # term2 = torch.zeros_like(v)
    # term3 = torch.zeros_like(p)
    # term4 = torch.zeros_like(nu_tilde)
    #############################################

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
    continuity = u_x + v_y
    x_momentum = u*u_x + v*u_y + p_x - (1.0/Re)*(u_xx+u_yy) - 2*tau_x_1 - tau_x_2
    y_momentum = u*v_x + v*v_y + p_y - (1.0/Re)*(v_xx+v_yy) - tau_y_1 - 2*tau_y_2
    nu_tilde_diffusion_x = dde.grad.jacobian((1/Re + nu_tilde) * nu_tilde_x, x, i=0, j=0)
    nu_tilde_diffusion_y = dde.grad.jacobian((1/Re + nu_tilde) * nu_tilde_y, x, i=0, j=1)
    
    _print_counter["n"] += 1
    if _print_counter["n"] % 1000 == 0:
        fh.write(f"d min={d.min().item():.4e} max={d.max().item():.4e}  "
        f"r min={r.min().item():.4e} max={r.max().item():.4e}  "
        f"g min={g.min().item():.4e} max={g.max().item():.4e}  "
        f"fw min={fw.min().item():.4e} max={fw.max().item():.4e}  "
        f"S_tilde_N mean={S_tilde_N.mean().item():.4e}  gen={SN_p.mean().item():.4e}  "
        f"destr={SN_d.mean().item():.4e}  nu_tilde mean={nu_tilde.mean().item():.4e} "
        f"S_tilde_N/nu_tilde mean = {S_tilde_N.mean().item()/nu_tilde.mean().item():.4e}\n")
        fh.flush()

    nu_convection = (d**2) * (u*nu_tilde_x + v*nu_tilde_y)
    nu_diffusion  = (d**2) * (1/sigma) * (nu_tilde_diffusion_x + nu_tilde_diffusion_y)
    nu_cross_diff = (d**2) * (cb2/sigma) * (nu_tilde_x**2 + nu_tilde_y**2)
    nu_tilde_transport = nu_convection - SN_p - nu_diffusion - nu_cross_diff + SN_d

    return [continuity, x_momentum, y_momentum, nu_tilde_transport]

run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
folder_name =  f"../finalresults/sduct/plots_{run_time}_sduct"
os.makedirs(folder_name, exist_ok=True)
os.makedirs(f"{folder_name}/ckpt", exist_ok=True)
shutil.copy(__file__, os.path.join(folder_name, "script_snapshot.py"))

B = torch.randn(128, 2) * 2.0
# plt.rcParams.update({
#     "text.usetex": False,
#     "font.family": "serif",
#     "font.serif": ["cmr10"],      
#     "mathtext.fontset": "cm",     
#     "axes.formatter.use_mathtext": True,
#     "font.size": 11,
#     "axes.labelsize": 11,
#     "legend.fontsize": 9,
#     "axes.linewidth": 0.8,
#     "lines.linewidth": 1.2,
# })

plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 14,
    'axes.titlesize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
})

Re = 2.5e6
###calc using bruns data####
k = 66
###########################
_print_counter = {"n": 0}
fh = open(f"{folder_name}/nu_tilde_values.txt", 'a')
fr = open(f"{folder_name}/residual_values.txt", 'a')
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

offsets = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]

near_wall_pts = []
for off in offsets:
    top_pts = np.column_stack([top_coords[:, 0], top_coords[:, 1] - off])
    bot_pts = np.column_stack([bottom_coords[:, 0], bottom_coords[:, 1] + off])
    near_wall_pts.append(top_pts)
    near_wall_pts.append(bot_pts)

near_wall_pts = np.vstack(near_wall_pts)
inside_mask = geom.inside(near_wall_pts)
near_wall_pts = near_wall_pts[inside_mask]

data = dde.data.PDE(
    geom,
    navier_stokes_with_SA,
    [inlet_x, inlet_y, outlet_p, outlet_u_x, outlet_v_x, wall_x, wall_y, inlet_nu, wall_nu, outlet_nu_x],
    train_distribution="LHS",
    num_domain=5000,
    num_boundary=4000,
    num_test=5000,
    anchors=near_wall_pts
)

labels = ["Continuity", "X Momentum", "Y  Momentum", "Nu Transport", "Inlet X", "Inlet Y",
          "Outlet_P", "Outlet Neumann U", "Outlet Neumann V", "Wall X", "Wall Y",
          "Inlet Nu", "Wall Nu", "Outlet Neumann Nu"]

# layer_size = [256] + [64] * 6 + [4]
layer_size = [2] + [64] * 6 + [4]
net = dde.nn.FNN(layer_size, "tanh", "Glorot uniform")
net.apply_feature_transform(input_transform)
net.apply_output_transform(output_transform)
# prev_net = copy.deepcopy(net)
loss_weights = [
    10,     # continuity
    10,    # x_momentum
    10,    # y_momentum
    20,     # nu_tilde_transport
    50,    # inlet_x
    20,    # inlet_y
    5,     # outlet_p
    5,     # outlet_u_x (Neumann)
    5,     # outlet_v_x (Neumann)
    50,    # wall_y
    50,    # wall_x
    20,    # inlet_nu
    20,    # wall_nu
    50,     # outlet_nu_x (Neumann)
]

# loss_weights = [
#     1,     # continuity
#     1,    # x_momentum
#     1,    # y_momentum
#     1,     # nu_tilde_transport
#     1,    # inlet_x
#     1,    # inlet_y
#     1,     # outlet_p
#     1,     # outlet_u_x (Neumann)
#     1,     # outlet_v_x (Neumann)
#     1,    # wall_y
#     1,    # wall_x
#     1,    # inlet_nu
#     1,    # wall_nu
#     1,     # outlet_nu_x (Neumann)
# ]

model = dde.Model(data, net)
plot_contour = PlotCallback(period=1000)
plot_residual = PlotCallback1(period=5000)
# update_prev_net = UpdatePrevNet(period=1)
plot_loss = LossPlotCallback(period=5000, labels=labels)
# gradnorm_cb = GradNormCallback(period=1000, lr=1e-3)
early_stopping = dde.callbacks.EarlyStopping(min_delta=1e-8, patience=5000)
checkpoint_cb = dde.callbacks.ModelCheckpoint(filepath=f"{folder_name}/ckpt/model", save_better_only=False, period=1000)
resampler = dde.callbacks.PDEPointResampler(period=1000)
re_ramp = ReRampCallback([(0, 1e3), (10000, 1e4), (20000, 1e5), (30000, 2.5e6)])
stage_callbacks = [plot_contour, plot_loss, plot_residual, resampler, re_ramp]

model.compile("adam", lr=1e-3, decay=("step", 2000, 0.9), loss_weights=loss_weights)
# model.restore("plots_20260826_002847_s_duct_isolated_turbulent_soft_modified/ns_model-75000.pt")
losshistory, train_state = model.train(iterations=55000,  display_every=10, callbacks=stage_callbacks, disregard_previous_best=False, model_save_path=f"{folder_name}/ns_model")
model.compile("L-BFGS")
# model.restore("plots_20260831_234204_s_duct_isolated_turbulent_soft_modified_og/ns_model_final-70000.pt")
losshistory, train_state = model.train(iterations=5000, callbacks=[plot_contour, update_prev_net, plot_loss, plot_residual], display_every=10, model_save_path=f"{folder_name}/ns_model_final")
# dde.saveplot(losshistory, train_state, issave=True, isplot=True)
np.save(f"{folder_name}/train_loss.npy", np.array(losshistory.loss_train))
np.save(f"{folder_name}/test_loss.npy", np.array(losshistory.loss_test))
np.save(f"{folder_name}/steps.npy", np.array(losshistory.steps))

#initialize initial error 
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
#     f = model.predict(X, operator=navier_stokes_with_SA)
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

#########################################################################
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
p_pred = model.predict(point)
p_ref = p_pred[:, 0]
print(f"p_ref : {p_ref}")

outlet_y = np.linspace(0.05999999999999983, 0.46, 20)
outlet_x = np.full_like(outlet_y, 1.888)
outlet_pts = np.column_stack((outlet_x, outlet_y))
p_outlet = model.predict(outlet_pts)[:, 2]
print(p_outlet[0])

for Re in Relist:
    f = open(f"{folder_name}/details_Re{Re}.txt", 'w')
    # Re_column = np.full((points.shape[0], 1), float(Re))
    # X_3D = np.hstack((points, Re_column))
    inside_mask = geom.inside(points)
    prediction = model.predict(points)
    u_component = prediction[:, 0]
    v_component = prediction[:, 1]
    p = prediction[:, 2]
    u_mag_flat = np.sqrt(u_component**2 + v_component**2)
    u_mag_flat[~inside_mask] = np.nan
    U_mag = u_mag_flat.reshape(xx.shape)
    plt.figure(figsize=(10, 5), dpi=300)
    contour = plt.contourf(xx, yy, U_mag, levels=200, cmap='Spectral_r', antialiased=True)
    cbar = plt.colorbar(contour, orientation='horizontal', pad=0.15, shrink = 0.6)
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    cbar.set_label('U Magnitude', fontsize=10)
    plt.xlim(0.0, 1.9)
    plt.ylim(0.04, 0.71)
    plt.gca().set_aspect('equal')
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel("X coord.", fontsize=10)
    plt.ylabel("Y coord.", fontsize=10)
    plt.title(rf"Dimensionless Velocity Magnitude |u|/U0", fontsize=10)
    plt.savefig(f"{folder_name}/vel_contour_Re{Re:.2e}.png", bbox_inches='tight', dpi=300)

    #C_p
    Cp = 2 * (p - p_outlet[0])
    plt.figure(figsize=(10, 5), dpi=300)
    Cp[~inside_mask] = np.nan
    Cp_mag = Cp.reshape(xx.shape)
    contour = plt.contourf(xx, yy, Cp_mag, levels=25, cmap='Spectral_r', antialiased=True)
    cbar = plt.colorbar(contour, orientation='horizontal', pad=0.15, shrink = 0.6)
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    cbar.set_label(r"$C_{p}$", fontsize=10)
    plt.xlim(0.0, 1.9)
    plt.ylim(0.04, 0.71)
    plt.gca().set_aspect('equal')
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel("X coord.", fontsize=10)
    plt.ylabel("Y coord.", fontsize=10)
    plt.title(rf"Pressure Coefficient $C_{{p}}$", fontsize=10)
    plt.savefig(f"{folder_name}/pres_contour_Re{Re:.2e}.png", bbox_inches='tight', dpi=300)

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
    nu_t_mag = nu_t_np.reshape(xx.shape)
    norm_nu_t_mag = norm_nu_t_np.reshape(xx.shape)
    numax = np.nanmax(nu_t_mag)
    norm_numax = np.nanmax(norm_nu_t_mag)

    plt.figure(figsize=(10, 5), dpi=300)
    contour = plt.contourf(xx, yy, nu_t_mag, levels=200, cmap='Spectral_r', antialiased=True)
    cbar = plt.colorbar(contour, orientation='horizontal', pad=0.15, shrink = 0.8)
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1e'))
    cbar.set_label(r"$\nu_{t}$ Magnitude", fontsize=10)
    plt.xlim(0.0, 1.9)
    plt.ylim(0.04, 0.71)
    plt.gca().set_aspect('equal')
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel("X coord.", fontsize=10)
    plt.ylabel("Y coord.", fontsize=10)
    plt.title(rf"$\nu_{{t}}$ Field (deg, max {numax})", fontsize=10)
    plt.savefig(f"{folder_name}/nu_t.png", bbox_inches='tight', dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5), dpi=300)
    contour = plt.contourf(xx, yy, norm_nu_t_mag, levels=200, cmap='Spectral_r', antialiased=True)
    cbar = plt.colorbar(contour, orientation='horizontal', pad=0.15, shrink = 0.6)
    cbar.ax.tick_params(labelsize=8)
    cbar.locator = ticker.MaxNLocator(nbins=8)
    cbar.update_ticks()
    cbar.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1e'))
    cbar.set_label(r"$\nu_{t}/ \nu$ Magnitude", fontsize=10)
    plt.xlim(0.0, 1.9)
    plt.ylim(0.04, 0.71)
    plt.gca().set_aspect('equal')
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel("X coord.", fontsize=10)
    plt.ylabel("Y coord.", fontsize=10)
    plt.title(rf"Normalized Eddy viscosity field $\nu_{{t}} / \nu $ (deg, max {norm_numax})", fontsize=10)
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
    TI_mag = TI_np.reshape(xx.shape)
    TImax = np.nanmax(TI_mag)

    plt.figure(figsize=(10, 5), dpi=300)
    contour = plt.contourf(xx, yy, TI_mag, levels=200, cmap='Spectral_r', antialiased=True)
    cbar = plt.colorbar(contour, orientation='horizontal', pad=0.15, shrink = 0.6)
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    cbar.set_label('Turbulence Intensity %', fontsize=10)
    plt.xlim(0.0, 1.9)
    plt.ylim(0.04, 0.71)
    plt.gca().set_aspect('equal')
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel("X coord.", fontsize=10)
    plt.ylabel("Y coord.", fontsize=10)
    plt.title(f"Turbulence Intensity % (deg, max {TImax})", fontsize=10)
    plt.savefig(f"{folder_name}/Turbulence_Intensity.png", bbox_inches='tight', dpi=300)
    plt.close()
    ##########################################################

# #plot Cp and u profiles
locs = [0.28003528, 0.2000252, 0.12001512]
colors = ['b', 'g', 'r']
colors1 = ['b', 'r']
label = ["U", "M", "D"]
res = ["bruns", "garbaruk", "cnn"]
labelt = ["Bruns et al.", "Garbaruk et al.", "Bokil et al. (PICNN)"]
res1 = ["bruns", "cnn"]
labelt1 = ["Bruns et al.", "Bokil et al. (PICNN)"]
marktype = ["o", "", ""]
total_num = {label: 0.0 for label in labelt}     
total_den = {label: 0.0 for label in labelt}
total_num_v = {label: 0.0 for label in labelt1}    
total_den_v = {label: 0.0 for label in labelt1}
for i, loc in enumerate(locs):
    bottom_coords_x = bottom_coords[:, 0:1]
    bottom_coords_y = bottom_coords[:, 1:2] + loc
    mid_curve = np.column_stack([bottom_coords_x, bottom_coords_y])
    prediction = model.predict(mid_curve)
    u_pred = prediction[:, 0]
    v_pred = prediction[:, 1]
    p_pred = prediction[:, 2]
    Cp = 2 * (p_pred - p_outlet[0])
    i0 = np.argmin(bottom_coords_x[:, 0])
    Cp = Cp - Cp[i0] 
    # new_bottom_coords_x = (bottom_coords_x - x_inlet_pinn)/(x_outlet_pinn - x_inlet_pinn)
    new_bottom_coords_x = bottom_coords_x / 0.4
    sort_idx = np.argsort(new_bottom_coords_x.flatten())
    x_sorted = new_bottom_coords_x.flatten()[sort_idx]
    Cp_sorted = Cp[sort_idx]
    plt.figure(figsize=(10, 5))
    plt.plot(new_bottom_coords_x, Cp, color='k', label=f"PINN")

    for j, type in enumerate(res):
        file_path = f'data/cp_{label[i]}_{type}.csv'
        my_data = np.genfromtxt(file_path, delimiter=',')
        x = my_data[:, 0]
        norm_x = x
        cp = my_data[:, 1]
        cp = cp - cp[0]
        if j == 0:
            plt.scatter(norm_x, cp, label=f"{labelt[j]}", color=colors[j], marker='o')
        elif j == 1:
            plt.plot(norm_x, cp, color=colors[j], linestyle="-.", label=f"{labelt[j]}")
        else:
            plt.plot(norm_x, cp, color=colors[j], linestyle="--", label=f"{labelt[j]}")
        plt.xlabel("Normalized x coordinate")
        plt.ylabel(r"$C_p - C_p|_{x=0}$")
        # plt.grid(True, linestyle=':')
        legend = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title(f"Pressure Coefficient $C_{{p}}$ at section {label[i]}")
        plt.savefig(f"{folder_name}/cp_profile_{label[i]}.png", dpi=300, bbox_inches="tight", bbox_extra_artists=(legend,))

        x_query = norm_x
        Cp_ = np.interp(x_query, x_sorted.flatten(), Cp_sorted)
        # y_at_query = np.interp(x_query, bottom_coords_x.flatten(), bottom_coords_y.flatten()) 
        # onedgrid = np.column_stack((x_query, y_at_query))
        # p_pred_ = model.predict(onedgrid)[:, 2]
        # Cp_ = 2 * (p_pred_ - p_outlet[0])

        diff = Cp_ - cp
        l2_num = np.sqrt(np.sum(diff**2))
        l2_den = np.sqrt(np.sum(cp**2))
        l2_rel = (l2_num / l2_den) * 100
        total_num[labelt[j]] += np.sum(diff**2)
        total_den[labelt[j]] += np.sum(cp**2)

        abs_error = np.abs(Cp_ - cp)        
        mean_abs_error = np.mean(abs_error)  
        max_abs_error = np.max(abs_error)    
        cp_mean = np.mean(np.abs(cp))
        mean_abs_error_pct = (mean_abs_error / cp_mean) * 100
        max_abs_error_pct = (max_abs_error / cp_mean) * 100
        
        f.write(f"Station loc={label[i]} against {labelt[j]} Cp: MAE={mean_abs_error:.5f} ({mean_abs_error_pct:.2f}%), Max AE={max_abs_error:.5f} ({max_abs_error_pct:.2f}%)\n")

        f.write(f"Station loc={label[i]} against {labelt[j]} Cp: abs L2={l2_num:.5f}, rel L2={l2_rel:.5f} %\n")

    plt.close()

    #u profile
    plt.figure(figsize=(10, 5))
    u_mag_flat = np.sqrt(u_pred**2 + v_pred**2)
    u_mag_flat = u_mag_flat - u_mag_flat[i0]
    u_mag_sorted = u_mag_flat[sort_idx]
    plt.plot(new_bottom_coords_x, u_mag_flat, color='k', label=f"PINN")
    for k, type in enumerate(res1):
        file_path = f'data/u_{label[i]}_{type}.csv'
        my_data = np.genfromtxt(file_path, delimiter=',')
        x = my_data[:, 0]
        # norm_x = (x - x_inlet_exp) / (x_outlet_exp - x_inlet_exp)
        norm_x = x
        umag = my_data[:, 1]
        umag = umag - umag[0]
        if k == 0:
            plt.scatter(norm_x, umag, label=f"{labelt1[k]}", color=colors1[k], marker='o')
        elif k == 1:
            plt.plot(norm_x, umag, label=f"{labelt1[k]}", color=colors1[k], linestyle="--")
        plt.xlabel("Normalized x coordinate")
        plt.ylabel(r"$\vert u \vert / U_0$")
        # plt.grid(True, linestyle=':')
        legend = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title(rf"Dimensionless Velocity Magnitude $\vert u \vert / U_0$ at section {label[i]}")
        plt.savefig(f"{folder_name}/u_profile_{label[i]}.png", dpi=300, bbox_inches="tight", bbox_extra_artists=(legend,))

        x_query = norm_x
        u_mag_flat_ = np.interp(x_query, x_sorted, u_mag_sorted)
        # y_at_query = np.interp(x_query, bottom_coords_x.flatten(), bottom_coords_y.flatten()) 
        # onedgrid = np.column_stack((x_query, y_at_query))
        # prediction = model.predict(onedgrid)
        # u_pred_ = prediction[:,0]
        # v_pred_ = prediction[:,1]
        # u_mag_flat_ = np.sqrt(u_pred_**2 + v_pred_**2)

        diff = u_mag_flat_ - umag
        l2_num = np.sqrt(np.sum(diff**2))
        l2_den = np.sqrt(np.sum(umag**2))
        l2_rel = (l2_num / l2_den) * 100
        total_num_v[labelt1[k]] += np.sum(diff**2)
        total_den_v[labelt1[k]] += np.sum(umag**2)

        abs_error = np.abs(u_mag_flat_ - umag)        
        mean_abs_error = np.mean(abs_error)  
        max_abs_error = np.max(abs_error)    
        u_mean = np.mean(np.abs(umag))
        #change to ref value as 1
        # mean_abs_error_pct = (mean_abs_error / u_mean) * 100
        # max_abs_error_pct = (max_abs_error / u_mean) * 100
        mean_abs_error_pct = (mean_abs_error) * 100
        max_abs_error_pct = (max_abs_error) * 100

        f.write(f"Station loc={label[i]} against {labelt1[k]} U: MAE={mean_abs_error:.5f} ({mean_abs_error_pct:.2f}%), Max AE={max_abs_error:.5f} ({max_abs_error_pct:.2f}%)\n")

        f.write(f"Station loc={label[i]} against {labelt1[k]} U: abs L2={l2_num:.5f}, rel L2={l2_rel:.5f}\n")

    plt.close()

overall_rel_l2_bruns = (np.sqrt(total_num["Bruns et al."]) / np.sqrt(total_den["Bruns et al."])) * 100 
overall_rel_l2_garbaruk = (np.sqrt(total_num["Garbaruk et al."]) / np.sqrt(total_den["Garbaruk et al."])) * 100
overall_rel_l2_cnn = (np.sqrt(total_num["Bokil et al. (PICNN)"]) / np.sqrt(total_den["Bokil et al. (PICNN)"])) * 100
f.write(f"Overall relative L2 Cp (Bruns et al.): {overall_rel_l2_bruns:.5f} %\n")
f.write(f"Overall relative L2 Cp (Garbaruk et al.): {overall_rel_l2_garbaruk:.5f} %\n")
f.write(f"Overall relative L2 Cp (Bokil et al.): {overall_rel_l2_cnn:.5f} %\n")

overall_rel_l2_bruns = (np.sqrt(total_num_v["Bruns et al."]) / np.sqrt(total_den_v["Bruns et al."])) * 100    
overall_rel_l2_cnn = (np.sqrt(total_num_v["Bokil et al. (PICNN)"]) / np.sqrt(total_den_v["Bokil et al. (PICNN)"])) * 100
f.write(f"Overall relative L2 |u|/U0 (Bruns et al.): {overall_rel_l2_bruns:.5f}\n")
f.write(f"Overall relative L2 |u|/U0 (Bokil et al.): {overall_rel_l2_cnn:.5f}\n")

f.close()

# #############################################################
# # 3#Loss Curve
plt.figure(figsize=(10, 6))
# labels = ["Continuity", "X Momentum", "Y  Momentum", "Nu Transport", "Inlet X", "Inlet Y", "Outlet_P", "Outlet Neumann U", "Outlet Neumann V", "Wall X", "Wall Y", "Inlet Nu", "Wall Nu", "Outlet Neumann Nu"]
# labels = ["Continuity", "X Momentum", "Y  Momentum", "Outlet U Neumann"]
train_loss = np.array(losshistory.loss_train)
steps = np.array(losshistory.steps)
# plt.gca().set_prop_cycle(plt.cm.tab20(np.linspace(0, 1, 14)))
plt.gca().set_prop_cycle(color=plt.cm.tab20(np.linspace(0, 1, 14)))
for i in range(train_loss.shape[1]):
    temp_loss = train_loss[:, i]
    ref_loss = temp_loss[0]
    normalized_loss = temp_loss / ref_loss
    plt.semilogy(steps, normalized_loss + 1e-10, label=labels[i])

# Labels and Style
plt.xlabel("Iterations")
plt.ylabel("Normalized Loss ($L / L_{initial}$)")
plt.title("Training Loss")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
plt.grid(True, which="both", ls="-", alpha=0.3)
plt.tight_layout()

plt.savefig(f"{folder_name}/training_loss.png", dpi=300, bbox_inches="tight")

#Test loss
plt.figure(figsize=(10, 6))
test_loss = np.array(losshistory.loss_test)
steps = np.array(losshistory.steps)

for i in range(test_loss.shape[1]):
    temp_loss = test_loss[:, i]
    ref_loss = temp_loss[0]
    normalized_loss = temp_loss / ref_loss
    plt.semilogy(steps, normalized_loss + 1e-10, label=labels[i])

# Labels and Style
plt.xlabel("Iterations")
plt.ylabel("Normalized Loss ($L / L_{initial}$)")
plt.title("Test Loss")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
plt.grid(True, which="both", ls="-", alpha=0.3)
plt.tight_layout()

plt.savefig(f"{folder_name}/test_loss.png", dpi=300, bbox_inches="tight")
# #############################################################
# Visualise loss landscape 
# write code to visualise loss landscape
fh.close()
fr.close()