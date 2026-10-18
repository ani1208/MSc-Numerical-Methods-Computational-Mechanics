import os
import deepxde as dde
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from datetime import datetime
from deepxde.backend import torch
from numpy import genfromtxt
import pandas as pd
import seaborn as sns 
import matplotlib.ticker as ticker
from matplotlib.ticker import FormatStrFormatter

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
    x1 = np.linspace(0, L, 21)
    y1 = np.linspace(0, W, 41)
    xx, yy = np.meshgrid(x1, y1, indexing='ij')
    X_space = np.vstack((xx.ravel(), yy.ravel())).T
    Re=75
    Pr=5
    Re_column = np.full((X_space.shape[0], 1), float(Re))
    Pr_column = np.full((X_space.shape[0], 1), float(Pr))
    X_4D = np.hstack((X_space, Re_column, Pr_column))
    y_pred = model.predict(X_4D)
    y_pred_reshaped = y_pred.reshape(len(x1), len(y1))

    plt.figure(figsize=(8, 10))
    # step = 0.047
    # all_levels = np.arange(0.0, 0.940 + (2 * step), step)
    plt.contourf(xx, yy, y_pred_reshaped, levels=50, cmap='Spectral_r')
    plt.colorbar(label="u")
    # explicit_ticks = np.arange(0.094, 0.940 + step, 0.094)
    # cbar.set_ticks(explicit_ticks)
    # cbar.set_ticklabels([f"{t:.3f}" for t in explicit_ticks])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("PINNs Solution for 2D Convection-Diffusion Equation")
    plt.savefig(f"{folder_name}/iteration_{epoch}_comparison_Re{Re}_Pr{Pr}.png", dpi=300, bbox_inches="tight")  
    plt.close()

#x_input = x , y
#u = u
def pde(x_input, u):
    Re = x_input[:, 2:3]
    Pr = x_input[:, 3:4]
    v1 = -torch.sin(np.pi * x_input[:, 0:1]) * torch.cos(np.pi * x_input[:, 1:2])
    v2 = torch.cos(np.pi * x_input[:, 0:1]) * torch.sin(np.pi * x_input[:, 1:2])
    dT_dx = dde.grad.jacobian(u, x_input, i=0, j=0)
    dT_dy = dde.grad.jacobian(u, x_input, i=0, j=1)
    dT_xx = dde.grad.hessian(u, x_input, i=0, j=0)
    dT_yy = dde.grad.hessian(u, x_input, i=1, j=1)
    diffusivity = 1.0 / (Re * Pr)
    diffusion = diffusivity * (dT_xx + dT_yy)
    advection = v1 * dT_dx + v2 * dT_dy
    return diffusion - advection


def boundary_left(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], 0.0)

def boundary_right(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], L)

def boundary_top(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[1], W)

def boundary_bottom(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[1], 0.0)

def neumann_y_condition(x, y, _):
    return dde.grad.jacobian(y, x, i=0, j=1)

def apply_hard_bcs(x, y):
    x_coord = x[:, 0:1]
    y_coord = x[:, 1:2]

    #dirichlet
    dirichlet_part = (x_coord / L)
    transition_dist = x_coord * (L - x_coord)

    #final 
    u_final = dirichlet_part + transition_dist * y
    
    return u_final

# this func exploits symmetry, it treats the field as centered around y=1 
def modify_inputs(x):
    x_coord = x[:, 0:1] # 0.2
    y_coord = x[:, 1:2] # 0.5 and 1.5
    y_symmetric = torch.square(y_coord - 1.0) # Both become 0.25!
    Re = x[:, 2:3] 
    Pr = x[:, 3:4]
    return torch.concat([x_coord, y_symmetric, Re, Pr], axis=1)

def apply_hard_dirichlet(x, y):
    x_coord = x[:, 0:1]
    raw_NN_output = y
    
    # Smooth distance function:
    return x_coord + x_coord * (1.0 - x_coord) * raw_NN_output

def sample_boundary_y0(n):
    x = np.random.uniform(0, L, n)
    y = np.zeros(n)
    return np.vstack((x, y)).T

def sample_domain_near_y0(n):
    x = np.random.uniform(0, L, n)
    y = np.random.uniform(0.1, 0.85, n)
    return np.vstack((x, y)).T

def sample_domain_near_y1(n):
    x = np.random.uniform(0, L, n)
    y = np.random.uniform(0.8, 1.2, n)
    return np.vstack((x, y)).T

plt.rcParams.update({
    'font.size': 16,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
})

eps = 1e-4
L = 1
W = 2
max_refinement_steps = 4 
step = 0
max_residual = 1.0
previous_max_res = float('inf')
# Prlist = [1, 3, 5]
# Relist = [25, 75, 100]
# Prlist = [1]
# Relist = [100]
low = [0, 0, 5.0, 0.5]
# High bounds: [x_max, y_max, Re_max, Pr_max]
high = [L, W, 100.0, 5.0]
# This creates a 4D sampling domain
geom = dde.geometry.Hypercube(low, high)

run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
folder_name = f"../../incomp/finalresults/convdiff/range/plots_{run_time}_main"
os.makedirs(folder_name, exist_ok=True)

#Geometry Definition
# geom = dde.geometry.Rectangle([0, 0], [L, W])

#BC u(0,y)=0;u(L,y)=1
#NBC u'(y=0)=u'(y=2)=0
bc_list = [
    # dde.icbc.DirichletBC(geom, lambda x: 0, boundary_left),
    # dde.icbc.DirichletBC(geom, lambda x: 1, boundary_right),
    dde.icbc.OperatorBC(geom, neumann_y_condition, boundary_bottom),
    dde.icbc.OperatorBC(geom, neumann_y_condition, boundary_top),
]

# pde_resampler = dde.callbacks.PDEPointResampler(period=100, pde_points=True, bc_points=False)

#PDE
data = dde.data.PDE(
    geom,
    pde,
    bc_list,
    # train_distribution = "LHS",
    num_domain = 8000,
    num_boundary = 2000,
    num_test = 8000,
)

layer_size = [4] + [128] * 6 + [1]
activation = "tanh"
initializer = "Glorot normal"
net = dde.nn.FNN(layer_size, activation, initializer)
net.apply_feature_transform(modify_inputs)
# Force BCs
net.apply_output_transform(apply_hard_dirichlet)
model = dde.Model(data, net)
model.compile("adam", lr=0.0001)
plot_contour = PlotCallback(period=1000)
losshistory, train_state = model.train(iterations=30000, callbacks=[plot_contour], display_every=10, disregard_previous_best=False, model_save_path=f"{folder_name}/cd_model")
model.compile("L-BFGS")
# model.restore("plots_20260628_232425_main/cd_model_final-25000.pt")
losshistory, train_state = model.train(iterations=5000, callbacks=[plot_contour], display_every=10, model_save_path=f"{folder_name}/cd_model_final")


dde.saveplot(losshistory, train_state, issave=True, isplot=True)
#############################################################
# #Save Plots for Analytical vs PINNs Solutions
# 1 #PINNs solution
Relist = [25, 75, 100]
Prlist = [1, 3, 5]
x1 = np.linspace(0, L, 21)
y1 = np.linspace(0, W, 41)
xx, yy = np.meshgrid(x1, y1, indexing='ij')
X_space = np.vstack((xx.ravel(), yy.ravel())).T

for Re in Relist:
    for Pr in Prlist:
        Re_column = np.full((X_space.shape[0], 1), float(Re))
        Pr_column = np.full((X_space.shape[0], 1), float(Pr))
        X_4D = np.hstack((X_space, Re_column, Pr_column))
        y_pred = model.predict(X_4D)
        y_pred_reshaped = y_pred.reshape(len(x1), len(y1))
        plt.figure(figsize=(25, 8))
        plt.subplot(1, 3, 1)
        ctr = plt.contourf(xx, yy, y_pred_reshaped, levels=50, cmap='Spectral_r')
        cbar = plt.colorbar(ctr, label="u PINN", ax=plt.gca(), format=ticker.ScalarFormatter(useMathText=True))
        cbar.formatter.set_powerlimits((0, 0))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f"PINNs Solution (Re={Re}, Pr={Pr})")
        # plt.savefig(f"{folder_name}/pinns.png", dpi=300, bbox_inches="tight")

        my_data = genfromtxt(f'Re{Re}Pr{Pr}.csv', delimiter=',')
        plt.subplot(1, 3, 2)
        x1 = np.linspace(0, 1, 21)
        y1 = np.linspace(0, 2, 41)
        # print(my_data)
        X = my_data[:, 0:2]      
        y = my_data[:, 2:3]
        xx, yy = np.meshgrid(x1, y1, indexing='ij')
        exact = y.reshape(len(x1), len(y1))
        ctr = plt.contourf(xx, yy, exact, levels=50, cmap='Spectral_r')
        # plt.colorbar()
        cbar = plt.colorbar(ctr, label="u FEM", ax=plt.gca(), format=ticker.ScalarFormatter(useMathText=True))
        cbar.formatter.set_powerlimits((0, 0))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title("Reference Solution")
        # plt.savefig("ReferenceRe.png", dpi=300, bbox_inches="tight")

        plt.subplot(1, 3, 3)
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
        plt.savefig(f"{folder_name}/comparison_Re{Re}_Pr{Pr}.png", dpi=300, bbox_inches="tight")
        ###########L2 Norm in a .txt file ###########################
        l2 = (np.linalg.norm(y_true_reshaped - y_pred_reshaped) / np.linalg.norm(y_true_reshaped)) * 100
        # Use append mode ("a") or dynamic filenames to preserve all combinations
        with open(f"{folder_name}/L2error_Re{Re}_Pr{Pr}.txt", "w") as f:
            f.write(f"L2 Error for Re={Re}, Pr={Pr}: {l2:.4f}%\n")
        #############################################################    

#############################################################
# 3#Loss Curve
# After training
train_loss = np.array(losshistory.loss_train)
steps = np.array(losshistory.steps)
pde_losses = train_loss[:,0]/10
# lbc_losses = train_loss[:,1]
# rbc_losses = train_loss[:,2]
btbc_losses = train_loss[:,1]
tbc_losses = train_loss[:,2]
ref_pde = pde_losses[0]
# ref_lbc = lbc_losses[0]
# ref_rbc = rbc_losses[0]
ref_btbc = btbc_losses[0]
ref_tbc = tbc_losses[0]
#normalize
normalized_pde = pde_losses/ref_pde
# normalized_lbc = lbc_losses/ref_lbc
# normalized_rbc = rbc_losses/ref_rbc
normalized_btbc = btbc_losses/ref_btbc
normalized_tbc = tbc_losses/ref_tbc
plt.figure(figsize=(10, 6))
plt.semilogy(steps, normalized_pde, label="PDE")
# plt.semilogy(steps, normalized_lbc, label="Left Dirichlet BC")
# plt.semilogy(steps, normalized_rbc, label="Right Dirichlet BC")
plt.semilogy(steps, normalized_btbc, label="Bottom Neumann BC")
plt.semilogy(steps, normalized_tbc, label="Top Neumann BC")

# Labels and Style
plt.xlabel("Iterations")
plt.ylabel("Normalized Loss ($L / L_{initial}$)")
plt.title("Training Loss")
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.3)
plt.tight_layout()

plt.savefig(f"{folder_name}/training_loss.png", dpi=300, bbox_inches="tight")

#Test loss
test_loss = np.array(losshistory.loss_test)
steps = np.array(losshistory.steps)
pde_losses = test_loss[:,0]/10
# lbc_losses = test_loss[:,1]
# rbc_losses = test_loss[:,2]
btbc_losses = test_loss[:,1]
tbc_losses = test_loss[:,2]
ref_pde = pde_losses[0]
# ref_lbc = lbc_losses[0]
# ref_rbc = rbc_losses[0]
ref_btbc = btbc_losses[0]
ref_tbc = tbc_losses[0]
#normalize
normalized_pde = pde_losses/ref_pde
# normalized_lbc = lbc_losses/ref_lbc
# normalized_rbc = rbc_losses/ref_rbc
normalized_btbc = btbc_losses/ref_btbc
normalized_tbc = tbc_losses/ref_tbc
plt.figure(figsize=(10, 6))
plt.semilogy(steps, normalized_pde, label="PDE")
# plt.semilogy(steps, normalized_lbc, label="Left Dirichlet BC")
# plt.semilogy(steps, normalized_rbc, label="Right Dirichlet BC")
plt.semilogy(steps, normalized_btbc, label="Bottom Neumann BC")
plt.semilogy(steps, normalized_tbc, label="Top Neumann BC")


# Labels and Style
plt.xlabel("Iterations")
plt.ylabel("Normalized Loss ($L / L_{initial}$)")
plt.title("Test Loss")
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.3)
plt.tight_layout()

plt.savefig(f"{folder_name}/test_loss.png", dpi=300, bbox_inches="tight")
#############################################################
