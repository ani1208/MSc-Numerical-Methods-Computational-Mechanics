# Navier–Stokes: Backward-Facing Step

PINN approximation for the steady, incompressible 2D Navier–Stokes equations over a backward-facing step at Re = 389

$$
\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0
$$

$$
u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} = -\frac{\partial p}{\partial x} + \frac{1}{Re} \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right)
$$

$$
u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y} = -\frac{\partial p}{\partial y} + \frac{1}{Re} \left( \frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2} \right)
$$

<p align="center">
 <img width="700" alt="geom_bfs" src="https://github.com/user-attachments/assets/4520c1f5-3f2c-47a4-a32c-c0389fe30495" />
  <br><em>Computational domain</em>
</p>

## Boundary Conditions

| Boundary | Condition |
|---|---|
| Inlet | $u = 6(y - 1)(2.061 - y), \quad v = 0$ |
| Outlet | $p = 0$ |
| Top wall | $u = 0, \quad v = 0$ |
| Bottom wall | $u = 0, \quad v = 0$ |
| Mid-horizontal wall | $u = 0, \quad v = 0$ |
| Mid-vertical wall | $u = 0, \quad v = 0$ |

**Benchmark:** [ref]

# Results

<div align="center">
<table>
  <tr>
    <th>Velocity field</th>
  </tr>
  <tr>
    <td><img width="700" alt="vel_contour_stream_w_features" src="https://github.com/user-attachments/assets/8516921a-2d0d-4857-ae5b-ba821ed4c655" /></td>
  </tr>
  <tr>
    <th>Pressure field</th>
  </tr>
  <tr>
    <td><img width="700" alt="pres_contour_w_features" src="https://github.com/user-attachments/assets/dedac745-c1f0-4b1a-ae87-ad44dcd3ec10" /></td>
  </tr>
</table>
</div>

