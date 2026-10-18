import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pyvista
import ufl
from petsc4py import PETSc
from mpi4py import MPI
from datetime import datetime
from dolfinx import fem, mesh, io, plot
from scipy.interpolate import griddata
from dolfinx.fem.petsc import (
    assemble_vector,
    assemble_matrix,
    create_vector,
    apply_lifting,
    set_bc,
)
import matplotlib.ticker as ticker

mpl.use("Agg") 

run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
folder_name = f"plots_{run_time}_heat2d_dirichlet"
os.makedirs(folder_name, exist_ok=True)

#Time discretization
t = 0.0  # Start time
T = 1.1  # Final time
num_steps = 1100
dt = T/num_steps

#Spatial discretization
L, W = 1, 1
n, m = 1, 1
nx, ny = 100, 100
x_min, x_max = 0, 1
y_min, y_max = 0, 1

x_coords = np.linspace(x_min, x_max, nx)
y_coords = np.linspace(y_min, y_max, ny)
xx_space, yy_space = np.meshgrid(x_coords, y_coords)


domain = mesh.create_rectangle(
    MPI.COMM_WORLD,
    [np.array([0, 0]), np.array([L, W])],
    [nx, ny],
    mesh.CellType.quadrilateral,
)
V = fem.functionspace(domain, ("Lagrange", 1))
fenics_points = V.tabulate_dof_coordinates()[:, :2] 

#initial condition
def initial_condition(x):
    return np.sin(n * np.pi * x[0] / L) * np.sin(m * np.pi * x[1] / W)


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

u_n = fem.Function(V)
u_n.name = "u_n"
u_n.interpolate(initial_condition)

# Create boundary condition
fdim = domain.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(
    domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool)
)
bc = fem.dirichletbc(
    PETSc.ScalarType(0), fem.locate_dofs_topological(V, fdim, boundary_facets), V
)

#output file
xdmf = io.XDMFFile(domain.comm, "heat2d.xdmf", "w")
xdmf.write_mesh(domain)

#solution variable 
uh = fem.Function(V)
uh.name = "uh"
uh.interpolate(initial_condition)

num_local_dofs = V.dofmap.index_map.size_local
local_points = V.tabulate_dof_coordinates()[:num_local_dofs, :2]
u_slice0 = griddata(local_points, uh.x.array[:num_local_dofs], (xx_space, yy_space), method="linear")

xdmf.write_function(uh, t)

u_min_local0 = np.nanmin(u_slice0)
u_max_local0 = np.nanmax(u_slice0)
if u_max_local0 == u_min_local0:
    u_max_local0 += 1e-5

fig, ax = plt.subplots(figsize=(8, 6))
surf = ax.contourf(xx_space, yy_space, u_slice0, cmap="Spectral_r", levels=25,
                    vmin=u_min_local0, vmax=u_max_local0)
ax.set_aspect("equal")
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_title(f"Time: {t:.3f} s - FEM Solution")
cbar = fig.colorbar(surf, ax=ax, shrink=0.7)
cbar.formatter = ticker.ScalarFormatter(useMathText=True)
cbar.formatter.set_powerlimits((0, 0))
cbar.update_ticks()
plt.savefig(f"{folder_name}/step_{t:.3f}.png", dpi=150, bbox_inches="tight")
plt.close(fig)

with open(f"{folder_name}/U_vals_0.000.csv", "w") as f:
        f.write("x,y,u\n")
        x_flat = xx_space.ravel()
        y_flat = yy_space.ravel()
        u_flat = u_slice0.ravel()  
        
        for x_v, y_v, u_v in zip(x_flat, y_flat, u_flat):
            if not np.isnan(u_v):
                f.write(f"{x_v:.6f},{y_v:.6f},{u_v:.6f}\n")


#define the problem in variational UFL 
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
alpha = 0.4
f = fem.Constant(domain, PETSc.ScalarType(0))
a = u * v * ufl.dx + dt * alpha * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = u_n * v * ufl.dx

#assembly
bilinear_form = fem.form(a)
linear_form = fem.form(L)
A = assemble_matrix(bilinear_form, bcs=[bc])
A.assemble()
b = create_vector(fem.extract_function_spaces(linear_form))

#create solver 
solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)

#plotting
structured_quad_grid = pyvista.ImageData()
structured_quad_grid.dimensions = (nx, ny, 1)
structured_quad_grid.spacing = ((x_max - x_min)/(nx-1), (y_max - y_min)/(ny-1), 0.0)
structured_quad_grid.origin = (x_min, y_min, 0.0)
grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(V))
grid.point_data["uh"] = uh.x.array
sampled_grid = structured_quad_grid.sample(grid)

plotter = pyvista.Plotter()
plotter.open_gif("u_time.gif", fps=10)

warped = sampled_grid.warp_by_scalar("uh", factor=1)

viridis = mpl.colormaps.get_cmap("Spectral_r").resampled(25)
sargs = dict(
    title_font_size=25,
    label_font_size=20,
    fmt="%.2e",
    color="black",
    position_x=0.1,
    position_y=0.8,
    width=0.8,
    height=0.1,
)
structured_quad_grid.point_data["uh"] = np.zeros(structured_quad_grid.n_points)
renderer = plotter.add_mesh(
    warped,
    show_edges=True,
    lighting=False,
    cmap=viridis,
    scalar_bar_args=sargs,
    clim=[0, max(uh.x.array)],
)

time_slices = [0, 0.25, 0.5, 0.75, 1.0]
for i in range(num_steps):
    
    print(t)
    # Update the right hand side reusing the initial vector
    with b.localForm() as loc_b:
        loc_b.set(0)
    assemble_vector(b, linear_form)

    # Apply Dirichlet boundary condition to the vector
    apply_lifting(b, [bilinear_form], [[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, [bc])

    # Solve linear problem
    solver.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()

    # Update solution at previous time step (u_n)
    u_n.x.array[:] = uh.x.array

    t += dt

    # Write solution to file
    xdmf.write_function(uh, t)
    
    
    num_local_dofs = V.dofmap.index_map.size_local
    local_points = V.tabulate_dof_coordinates()[:num_local_dofs, :2]
    u_values = uh.x.array[:num_local_dofs]
    
    # Map raw FEniCS data directly onto your xx_space, yy_space meshgrid arrays
    u_slice = griddata(local_points, u_values, (xx_space, yy_space), method="linear")
    
    u_min_local = np.nanmin(u_slice)
    u_max_local = np.nanmax(u_slice)
    if u_max_local == u_min_local:  # Prevent divide-by-zero layout errors on flat profiles
        u_max_local += 1e-5

    # if any(np.isclose(t, target) for target in time_slices):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    surf = ax.contourf(xx_space, yy_space, u_slice, cmap="Spectral_r", levels=25,vmin=u_min_local, vmax=u_max_local)

    ax.set_aspect("equal")
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_title(f"Time: {t:.3f} s - FEM Solution")
    cbar = fig.colorbar(surf, ax=ax, shrink=0.7)
    cbar.formatter = ticker.ScalarFormatter(useMathText=True)
    cbar.formatter.set_powerlimits((0, 0)) 
    cbar.update_ticks()

    plt.savefig(f"{folder_name}/step_{t:.3f}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


    with open(f"{folder_name}/U_vals_{t:.3f}.csv", "w") as f:
        f.write("x,y,u\n")
        x_flat = xx_space.ravel()
        y_flat = yy_space.ravel()
        u_flat = u_slice.ravel()  
        
        for x_v, y_v, u_v in zip(x_flat, y_flat, u_flat):
            if not np.isnan(u_v):
                f.write(f"{x_v:.6f},{y_v:.6f},{u_v:.6f}\n")

    

plotter.close()
xdmf.close()


A.destroy()
b.destroy()
solver.destroy()
