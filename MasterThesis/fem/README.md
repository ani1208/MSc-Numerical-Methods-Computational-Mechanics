# Heat Equation 2D

FEniCS solvers for the 2D heat equation on the unit square, used as a reference benchmark for PINNs.

$$
\frac{\partial u}{\partial t} = \alpha \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right), \qquad (x, y) \in [0, L] \times [0, W]
$$

<p align="center">
  <img width="300" alt="Square domain" src="https://github.com/user-attachments/assets/eaebf2e8-99f1-434b-b042-9a35844c2d17" />
  <br><em>Computational domain</em>
</p>

**Initial condition** (both cases)

$$
u(x, y, 0) = \sin\left(\frac{n\pi x}{L}\right) \sin\left(\frac{n\pi y}{W}\right), \quad 0 < x < L,\ 0 < y < W
$$

## Case 1: Dirichlet

| Boundary | Condition |
|---|---|
| All edges | $u(0,y,t) = u(1,y,t) = u(x,0,t) = u(x,1,t) = 0$ |

## Case 2: Mixed Dirichlet / Neumann

| Boundary | Condition |
|---|---|
| Left, right, bottom | $u(0,y,t) = u(1,y,t) = u(x,0,t) = 0$ |
| Top ($y = 1$) | $\left.\dfrac{\partial u}{\partial y}\right\|_{y=1} = 0$ |
