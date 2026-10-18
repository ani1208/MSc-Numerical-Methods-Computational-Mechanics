# RANS: S-Shaped Duct

PINN approximation for the steady, incompressible 2D RANS equations with the Spalart–Allmaras turbulence model in an S-shaped duct at Re = 10⁶

**Continuity and momentum**

$$
\nabla \cdot \mathbf{u} = 0
$$

$$
(\mathbf{u} \cdot \nabla)\mathbf{u} = -\nabla p + \nabla \cdot \left[ \left( \frac{1}{Re} + \nu_t \right) \left( \nabla \mathbf{u} + \nabla \mathbf{u}^T \right) \right]
$$

**Spalart–Allmaras transport equation**

$$
(\mathbf{u} \cdot \nabla)\tilde{\nu} = c_{b1} \tilde{S} \tilde{\nu} - c_{w1} f_w \left( \frac{\tilde{\nu}}{d} \right)^2 + \frac{1}{\sigma} \left[ \nabla \cdot \left( \left( \frac{1}{Re} + \tilde{\nu} \right) \nabla \tilde{\nu} \right) + c_{b2} \left| \nabla \tilde{\nu} \right|^2 \right], \qquad \nu_t = \tilde{\nu} f_{v1}
$$

<p align="center">
  <img width="700" alt="S-shaped duct domain" src="YOUR_DOMAIN_IMAGE_LINK" />
  <br><em>Computational domain (U, M, D: upper, middle and lower evaluation lines)</em>
</p>

## Boundary Conditions

| Boundary | Condition |
|---|---|
| Inlet | $u = k \tanh\left(0.2000252 - \lvert y - 0.5000252 \rvert\right), \quad v = 0, \quad \tilde{\nu} = 5 / Re$ |
| Outlet | $\dfrac{\partial u}{\partial x} = 0, \quad \dfrac{\partial v}{\partial x} = 0, \quad \dfrac{\partial \tilde{\nu}}{\partial x} = 0, \quad p = 0$ |
| Top wall | $u = 0, \quad v = 0, \quad \tilde{\nu} = 0$ |
| Bottom wall | $u = 0, \quad v = 0, \quad \tilde{\nu} = 0$ |

where $k$ is the steepness of the inlet profile.

**Benchmark:** [ref]

# Results

<div align="center">
<table>
  <tr>
    <th>Velocity field</th>
  </tr>
  <tr>
    <td><img width="800" alt="Velocity field" src="YOUR_VELOCITY_IMAGE_LINK" /></td>
  </tr>
  <tr>
    <th>Pressure field</th>
  </tr>
  <tr>
    <td><img width="800" alt="Pressure field" src="YOUR_PRESSURE_IMAGE_LINK" /></td>
  </tr>
</table>
</div>
