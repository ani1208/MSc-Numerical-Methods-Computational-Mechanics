# RANS: S-Shaped Duct

PINN approximation for the steady, incompressible 2D RANS equations with the Spalart–Allmaras turbulence model in an S-shaped duct at Re = 10⁶

**Continuity**

$$
\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0
$$

**X-momentum**

$$
u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} - 2 \frac{\partial}{\partial x} \Big[ \nu_t \frac{\partial u}{\partial x} \Big] - \frac{\partial}{\partial y} \Big[ \nu_t \Big( \frac{\partial u}{\partial y} + \frac{\partial v}{\partial x} \Big) \Big] - \frac{\partial p}{\partial x} - \frac{1}{\mathrm{Re}} \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right) = 0
$$

**Y-momentum**

$$
u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y} - \frac{\partial}{\partial x} \Big[ \nu_t \Big( \frac{\partial u}{\partial y} + \frac{\partial v}{\partial x} \Big) \Big] - 2 \frac{\partial}{\partial y} \Big[ \nu_t \frac{\partial v}{\partial y} \Big] - \frac{\partial p}{\partial y} - \frac{1}{\mathrm{Re}} \left( \frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2} \right) = 0
$$

where $\nu_t = \tilde{\nu} f_{v1}$.

**Spalart–Allmaras transport equation**

```math
\begin{aligned}
\underbrace{\left(u\frac{\partial \tilde{\nu}}{\partial x}+v\frac{\partial \tilde{\nu}}{\partial y}\right)}_{\text{Convection}}
&-\underbrace{c_{b1}\tilde{S}\tilde{\nu}}_{\text{Production}}
-\underbrace{\frac{1}{\sigma}\left[\frac{\partial}{\partial x}\left((\nu+\tilde{\nu})\frac{\partial\tilde{\nu}}{\partial x}\right)+\frac{\partial}{\partial y}\left((\nu+\tilde{\nu})\frac{\partial\tilde{\nu}}{\partial y}\right)\right]}_{\text{Diffusion}} \\
&-\underbrace{\frac{c_{b2}}{\sigma}\left[\left(\frac{\partial\tilde{\nu}}{\partial x}\right)^{2}+\left(\frac{\partial\tilde{\nu}}{\partial y}\right)^{2}\right]}_{\text{Cross-diffusion}}
+\underbrace{c_{w1}f_w\left(\frac{\tilde{\nu}}{d}\right)^{2}}_{\text{Destruction}} = 0
\end{aligned}
```


<p align="center">
  <img width="700" alt="geom_sduct" src="https://github.com/user-attachments/assets/5cda3506-91e9-4873-85b8-1e92ae40d179" />
  <br><em>Computational domain (U, M, D: upper, middle and lower section lines)</em>
</p>


## Boundary Conditions

| Boundary | Condition |
|---|---|
| Inlet | $u = k \tanh\left(0.2000252 - \lvert y - 0.5000252 \rvert\right), \quad v = 0, \quad \tilde{\nu} = 5 / Re$ |
| Outlet | $\dfrac{\partial u}{\partial x} = 0, \quad \dfrac{\partial v}{\partial x} = 0, \quad \dfrac{\partial \tilde{\nu}}{\partial x} = 0, \quad p = 0$ |
| Top wall | $u = 0, \quad v = 0, \quad \tilde{\nu} = 0$ |
| Bottom wall | $u = 0, \quad v = 0, \quad \tilde{\nu} = 0$ |

where $k$ is the steepness of the inlet profile.

# Results

<div align="center">
<table>
  <tr>
    <th>Velocity field</th>
  </tr>
  <tr>
    <td><img width="700" alt="vel_contour_Re2 50e+06" src="https://github.com/user-attachments/assets/8defb9e1-c3f7-45db-ab2d-7e49d092ffaa" /></td>
  </tr>
  <tr>
    <th>Pressure field</th>
  </tr>
  <tr>
    <td><img width="700" alt="sduct_SA_overlay" src="https://github.com/user-attachments/assets/2b83ef90-d02c-46e7-b515-2566d0696479" /></td>
  </tr>
</table>
</div>

