import os
import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
from deepxde.backend import torch
from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as ticker
from numpy import genfromtxt

SEED = 1
dde.config.set_random_seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

#x_input = x , y, t
#u = u
def pde(x_input, u):
    dy_dt = dde.grad.jacobian(u, x_input, i=0, j=2)
    dy_xx = dde.grad.hessian(u, x_input, i=0, j=0)
    dy_yy = dde.grad.hessian(u, x_input, i=1, j=1)
    
    return dy_dt - a * (dy_xx + dy_yy)

def boundary_left(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], 0.0)

def boundary_right(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], L)

def boundary_top(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[1], W)

def boundary_bottom(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[1], 0.0)

def apply_hard_bcs(x, y):
    # x[:, 0:1] is x-coordinate
    # x[:, 1:2] is y-coordinate
    
    # This term is zero if x=0 or x=L
    x_boundary = x[:, 0:1] * (L - x[:, 0:1])
    
    # This term is zero if y=0 or y=W
    y_boundary = x[:, 1:2] * (W - x[:, 1:2])
    
    # Combine them. Since it's a product, if ANY edge is hit, the result is 0.
    return x_boundary * y_boundary * y

#Parameters for the heat equation
a = 0.4
L = 1
W = 1
n = m = 1
T = 1

plt.rcParams.update({
    'font.size': 16,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
})

#Geometry Definition
geom = dde.geometry.Rectangle([0, 0], [L, W])
# points = np.array([[0,0], [L,0], [L,W], [0,W]])
# boundary_points = points
# geom = dde.geometry.PointCloud(points, boundary_points)
time_domain = dde.geometry.TimeDomain(0, T)
geomtime = dde.geometry.GeometryXTime(geom, time_domain)

#BC u(0,y)=u(L,y)=0
#BC u(x,0)=u(x,W)=0
bc_list = [
    dde.icbc.DirichletBC(geomtime, lambda x: 0, boundary_left),
    dde.icbc.DirichletBC(geomtime, lambda x: 0, boundary_right),
    dde.icbc.DirichletBC(geomtime, lambda x: 0, boundary_bottom),
    dde.icbc.DirichletBC(geomtime, lambda x: 0, boundary_top),
]

#IC u(x,y,0)=sin(n.pi.x/L)
ic = dde.icbc.IC(
    geomtime,
    lambda x: np.sin(n * np.pi * x[:, 0:1] / L) * np.sin(m * np.pi * x[:, 1:2] / W),
    lambda _, on_initial: on_initial,
)

pde_resampler = dde.callbacks.PDEPointResampler(period=10)

#PDE
data = dde.data.TimePDE(
    geomtime,
    pde,
    [ic],
    num_domain = 4000,
    num_boundary = 500,
    num_initial = 500,
    num_test = 4000,
    )


#NN
# layer_size = [3] + [80] * 4 + [1]
layer_size = [3] + [50] * 4 + [1]
activation = "tanh"
initializer = "Glorot normal"
net = dde.nn.FNN(layer_size, activation, initializer)
# Force BCs
net.apply_output_transform(apply_hard_bcs)
model = dde.Model(data, net)
# lbfgs_tracker = LBFGSLogger()
# early_stopping = dde.callbacks.EarlyStopping(min_delta=1e-4, patience=2000)
model.compile("adam", lr=0.0001)
losshistory, train_state = model.train(iterations=15000, callbacks=[pde_resampler], display_every=10)
# model.compile("adam", lr=0.00001)
# losshistory, train_state = model.train(iterations=2000, callbacks=[pde_resampler], display_every=10)
model.compile("L-BFGS")
losshistory, train_state = model.train(display_every=10)

# Plot/print the results
dde.saveplot(losshistory, train_state, issave=True, isplot=True)

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

# Pre-calculate global min/max bounds for consistent Z-axis scaling across plots
u_global_predictions = []
u_initial = []
for t_val in time_slices:
    X_eval = np.vstack((xx_space.ravel(), yy_space.ravel(), np.full(xx_space.size, t_val))).T
    u_global_predictions.append(model.predict(X_eval).reshape(xx_space.shape))

u_min_global = min(u.min() for u in u_global_predictions)
u_max_global = max(u.max() for u in u_global_predictions)

for i, t_val in enumerate(time_slices):
    u_slice = u_global_predictions[i]
    # u_min_local = u_slice.min()
    u_min_local = 0.0
    u_max_local = u_slice.max()

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
    plt.savefig(f"step_pinn_{t_val:.3f}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    my_data = genfromtxt(f'dirichlet/U_vals_{t_val:.3f}.csv', delimiter=',', skip_header=1)
    x = my_data[:, 0:1]      
    y = my_data[:, 1:2]
    u = my_data[:, 2:3]
    u_true_reshaped = u.reshape(len(y_steps), len(x_steps)).T
    if i == 0:
        u_initial = u_true_reshaped
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
    plt.savefig(f"step_absolute_error_{t_val:.3f}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    with open(f"U_vals_{t_val:.3f}.csv", "w") as f:
        f.write("x,y,u\n")
        x_flat = xx_space.ravel(order='F')
        y_flat = yy_space.ravel(order='F')
        u_flat = u_slice.ravel(order='F')

        for x_v, y_v, u_v in zip(x_flat, y_flat, u_flat):
            f.write(f"{x_v:.6f},{y_v:.6f},{u_v:.6f}\n")

    ###########L2 Norm in a .txt file ###########################
    l2 = (np.linalg.norm(u_true_reshaped - u_slice) / np.linalg.norm(u_initial)) * 100
    with open(f"L2error_{t_val:.3f}.txt", "w") as f:
        f.write(f"L2 Error: {l2:.4f}%\n")
    #############################################################  

#############################################################
# prediction for x,y=0.5 
point = np.array([[0.50196078, 0.50196078, 0.0]])
predicted_val = model.predict(point)
x_0 = 0.50196078
y_0 = 0.50196078
initial_val = lambda x, y: np.sin(n * np.pi * x_0 / L) * np.sin(m * np.pi * y_0 / W)
peak_value = initial_val(x_0, y_0)
print("Predicted val at center (t=0)", predicted_val)
print("Exact val at center (t=0)", peak_value)
##############################################################
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
    
    pde_residual_slice = model.predict(X_slice, operator=pde)
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
        y_pred_t0 = model.predict(X_slice).flatten()
        y_true_t0 = (np.sin(n * np.pi * X_slice[:, 0] / L) * np.sin(m * np.pi * X_slice[:, 1] / W))
        ic_mse_slice = np.mean(np.square(y_pred_t0 - y_true_t0))
        weighted_pde_loss += (w_ic * ic_mse_slice)

    weighted_time_summary.append(weighted_pde_loss)

    plt.tight_layout(pad=2.0)
    plt.savefig(f"Time_Slices_Comparison_{t_val:.3f}.png", dpi=300, bbox_inches="tight")

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

plt.savefig("Time_Summary_Weighted_Loss.png", dpi=300, bbox_inches="tight")
print("Saved: Time_Summary_Weighted_Loss.png")
##################################################################
# 3#Loss Curve
plt.figure(figsize=(10, 6))
# labels = ["PDE", "BC Left", "BC Right", "BC Bottom", "BC Top", "IC"]
train_loss = np.array(losshistory.loss_train)
steps = losshistory.steps
pde_losses = train_loss[:,0]
ic_losses = train_loss[:,1]
ref_pde = pde_losses[0]
ref_ic = ic_losses[0]
#normalize
normalized_pde = pde_losses/ref_pde
normalized_ic = ic_losses/ref_ic
plt.semilogy(steps, normalized_pde, label="PDE")
plt.semilogy(steps, normalized_ic, label="IC")
# Labels and Style
plt.xlabel("Iterations")
plt.ylabel("Normalized Loss ($L / L_{initial}$)")
plt.title("Training Loss")
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.3)
plt.tight_layout()
plt.savefig("training_loss.png", dpi=300, bbox_inches="tight")

#Test loss
test_loss = np.array(losshistory.loss_test)
steps = np.array(losshistory.steps)
pde_losses = test_loss[:,0]
ic_losses = test_loss[:,1]
ref_pde = pde_losses[0]
ref_ic = ic_losses[0]
#normalize
normalized_pde = pde_losses/ref_pde
normalized_ic = ic_losses/ref_ic
plt.figure(figsize=(10, 6))
plt.semilogy(steps, normalized_pde, label="PDE")
plt.semilogy(steps, normalized_ic, label="IC")

# Labels and Style
plt.xlabel("Iterations")
plt.ylabel("Normalized Loss ($L / L_{initial}$)")
plt.title("Test Loss")
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.3)
plt.tight_layout()

plt.savefig("test_loss.png", dpi=300, bbox_inches="tight")
#############################################################
