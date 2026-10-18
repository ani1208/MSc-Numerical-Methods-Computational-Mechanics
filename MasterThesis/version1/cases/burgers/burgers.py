import os
import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
from deepxde.backend import torch

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

#x_input = x , t
#y = u
def pde(x_input, y):
    dy_dt = dde.grad.jacobian(y, x_input, i=0, j=1)
    dy_dx = dde.grad.jacobian(y, x_input, i=0, j=0)
    dy_xx = dde.grad.hessian(y, x_input, i=0, j=0)
    return dy_dt + y * dy_dx - 0.01 / np.pi * dy_xx

#Parameters for the burgers equation
L = 1
n = 1
T = 1

plt.rcParams.update({
    'font.size': 16,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
})


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

#PDE
data = dde.data.TimePDE(
    geomtime,
    pde,
    [bc, ic],
    num_domain = 2500,
    num_boundary = 100,
    num_initial = 500,
    num_test = 2500,
    )

#NN
layer_size = [2] + [50] * 3 + [1]
activation = "tanh"
initializer = "Glorot normal"
net = dde.nn.FNN(layer_size, activation, initializer)
model = dde.Model(data, net)
early_stopping = dde.callbacks.EarlyStopping(min_delta=1e-8, patience=5000)
model.compile("adam", lr=0.0001, loss_weights=[100, 1, 1], decay=("cosine", 15000, 0.0))
losshistory, train_state = model.train(iterations=15000, display_every=10, model_save_path=f"b_model")
model.compile("L-BFGS")
losshistory, train_state = model.train(iterations=2000, display_every=10, model_save_path=f"b_model_final")
# model.restore("b_model_final-30000.pt")
# Plot/print the results
dde.saveplot(losshistory, train_state, issave=True, isplot=True)
#############################################################

X, y_true, xx, tt, exact = gen_testdata()
y_pred = model.predict(X)
f = model.predict(X, operator=pde)
print("Mean residual:", np.mean(np.absolute(f)))
print("L2 relative error:", dde.metrics.l2_relative_error(y_true, y_pred))
np.savetxt("test.dat", np.hstack((X, y_true, y_pred)))
y_true=y_true.reshape(100, 256)
#############################################################
#Save Plots for Analytical vs PINNs Solutions
# 1#Analytical vs PINNs solution
t = np.linspace(0, T, 100)
x = np.linspace(-1, 1, 256)
xx, tt = np.meshgrid(x,t)
X = np.vstack((xx.ravel(), tt.ravel())).T

y_pred = model.predict(X)
y_pred_reshaped = y_pred.reshape(len(t), len(x))

plt.figure(figsize=(30, 10))
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

plt.subplot(1, 3, 3)
plt.contourf(xx, tt, np.abs(y_true - y_pred_reshaped), levels=50, cmap='inferno')
plt.colorbar(label="absolute error u", format='%.1e')
plt.xlabel('x')
plt.ylabel('t')
plt.title(r"$∣ analytical - prediction ∣$")
plt.savefig("pinns.png", dpi=300, bbox_inches="tight")

###########L2 Norm in a .txt file ###########################
l2 = (np.linalg.norm(y_true - y_pred_reshaped) / np.linalg.norm(y_true)) * 100
with open(f"L2error.txt", "w") as f:
    f.write(f"L2 Error: {l2:.4f}%\n")
#############################################################

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

    X_slice = np.vstack((x, np.full_like(x, t_val))).T
    pde_residual_slice = model.predict(X_slice, operator=pde)
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
        y_pred_t0 = model.predict(X_slice)
        y_true_t0 = np.sin(n * np.pi * x / L)
        ic_mse_slice = np.mean(np.square(y_pred_t0.flatten() - y_true_t0.flatten()))
        weighted_pde_loss += (w_ic * ic_mse_slice)

    weighted_time_summary.append(weighted_pde_loss)

plt.tight_layout(pad=2.0)
plt.savefig("Time_Slices_Comparison.png", dpi=300, bbox_inches="tight")
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

plt.savefig("Time_Summary_Weighted_Loss.png", dpi=300, bbox_inches="tight")

#############################################################
# 3#Loss Curve
# plt.figure(figsize=(10, 6))
# labels = ["PDE", "BC", "IC"]
# plt.figure(figsize=(10, 6))
train_loss = np.array(losshistory.loss_train)
steps = losshistory.steps
pde_losses = train_loss[:,0]
bc_losses = train_loss[:,1]
ic_losses = train_loss[:,2]
ref_pde = pde_losses[0]
ref_bc = bc_losses[0]
ref_ic = ic_losses[0]
#normalize
normalized_pde = pde_losses/ref_pde
normalized_bc = bc_losses/ref_bc
normalized_ic = ic_losses/ref_ic
plt.figure(figsize=(10, 6))
plt.semilogy(steps, normalized_pde, label="PDE")
plt.semilogy(steps, normalized_bc, label="BC")
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
bc_losses = train_loss[:,1]
ic_losses = test_loss[:,2]
ref_pde = pde_losses[0]
ref_bc = bc_losses[0]
ref_ic = ic_losses[0]
#normalize
normalized_pde = pde_losses/ref_pde
normalized_bc = bc_losses/ref_bc
normalized_ic = ic_losses/ref_ic
plt.figure(figsize=(10, 6))
plt.semilogy(steps, normalized_pde, label="PDE")
plt.semilogy(steps, normalized_bc, label="BC")
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