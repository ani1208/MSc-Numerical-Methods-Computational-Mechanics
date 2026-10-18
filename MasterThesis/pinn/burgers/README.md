# Burgers' Equation 1D

PINN approximation for the viscous Burgers' equation in 1D

$$
\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2}, \qquad \nu = \frac{0.01}{\pi}
$$

**Domain:** $x \in [-1, 1], \quad t \in [0, 1]$

**Initial condition**

$$
u(x, 0) = -\sin(n \pi x)
$$

## Boundary Conditions

| Boundary | Condition |
|---|---|
| Left and right ($x = \pm 1$) | $u(-1, t) = u(1, t) = 0$ |

# Results

<p align="center">
  <img width="1000" alt="pinns" src="https://github.com/user-attachments/assets/a77be289-b790-458b-a4c5-d106fbf65e80" />
</p>
