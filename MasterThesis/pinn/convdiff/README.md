# Convection–Diffusion 2D

PINN approximation for the steady convection–diffusion equation in 2D

$$
u \frac{\partial T}{\partial x} + v \frac{\partial T}{\partial y} = \frac{1}{Re \, Pr} \left( \frac{\partial^2 T}{\partial x^2} + \frac{\partial^2 T}{\partial y^2} \right), \qquad (x, y) \in [0, 1] \times [0, 2]
$$

<p align="center">
  <img width="300" alt="geom_rectangle" src="https://github.com/user-attachments/assets/0da52f4f-1d5b-476f-b87a-15d90dda2848" />
  <br><em>Computational domain</em>
</p>

**Velocity field**


$$
u = -\sin(\pi x)\cos(\pi y), \qquad v = \cos(\pi x)\sin(\pi y)
$$

## Boundary Conditions

| Boundary | Condition |
|---|---|
| Left and right (Dirichlet) | $T(0, y) = 0, \quad T(1, y) = 1, \quad 0 \le y \le 2$ |
| Bottom and top (Neumann) | $\left.\dfrac{\partial T}{\partial y}\right\|_{y=0} = 0, \quad \left.\dfrac{\partial T}{\partial y}\right\|_{y=2} = 0, \quad 0 \le x \le 1$ |

## Training Cases

- **Isolated:** trained on a particular $Re$, $Pr$ and predicted on the same values.
- **Parameterized:** trained on a range of $Re$, $Pr$ and predicted on a specific value within that range.

# Results

<div align="center">
<table>
  <tr>
    <th>Isolated (Re = 100, Pr = 5)</th>
  </tr>
  <tr>
    <td><img width="800" alt="Isolated result" src="https://github.com/user-attachments/assets/911745a8-f677-4648-86f5-9b28d711ad3a" /></td>
  </tr>
  <tr>
    <th>Parameterized (Re = 100, Pr = 5)</th>
  </tr>
  <tr>
    <td><img width="800" alt="Parameterized result" src="https://github.com/user-attachments/assets/36eb7b20-4b30-473b-8846-ead20a8dfebb" /></td>
  </tr>
</table>
</div>
