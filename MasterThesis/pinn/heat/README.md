# Heat Equation 2D

PINN approximation for the heat equation in 2D 

$$
\frac{\partial u}{\partial t} = \alpha \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right), \qquad (x, y) \in [0, L] \times [0, W]
$$

<p align="center">
   <img width="1769" height="1734" alt="geom_square" src="https://github.com/user-attachments/assets/f0e69fba-fa7f-49be-8d84-2e53b78ef571" />
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

# Results

<table>
  <tr>
    <th>Dirichlet case</th>
    <th>Mixed case</th>
  </tr>
  <tr>
    <td><img width="2062" height="1706" alt="step_pinn_1 000" src="https://github.com/user-attachments/assets/160f4282-63ed-4d64-b7b7-9732c7f1dd76" />
    <td><img width="1998" height="1728" alt="step_pinn_1 000 (1)" src="https://github.com/user-attachments/assets/23058003-e2ca-4dcf-93d3-81bc5c7d0b0a" />   
  </tr>
</table>
