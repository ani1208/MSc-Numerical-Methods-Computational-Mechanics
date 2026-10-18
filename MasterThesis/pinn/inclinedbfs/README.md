# RANS: Inclined Backward-Facing Step

PINN approximation for the steady, incompressible 2D RANS equations with the Spalart–Allmaras turbulence model over an inclined (45°) backward-facing step at Re = 36,000

**Continuity**

```math
\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0
```

**X-momentum**

```math
u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} - 2 \frac{\partial}{\partial x} \Big[ \nu_t \frac{\partial u}{\partial x} \Big] - \frac{\partial}{\partial y} \Big[ \nu_t \Big( \frac{\partial u}{\partial y} + \frac{\partial v}{\partial x} \Big) \Big] + \frac{\partial p}{\partial x} - \frac{1}{\mathrm{Re}} \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right) = 0
```

**Y-momentum**

```math
u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y} - \frac{\partial}{\partial x} \Big[ \nu_t \Big( \frac{\partial u}{\partial y} + \frac{\partial v}{\partial x} \Big) \Big] - 2 \frac{\partial}{\partial y} \Big[ \nu_t \frac{\partial v}{\partial y} \Big] + \frac{\partial p}{\partial y} - \frac{1}{\mathrm{Re}} \left( \frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2} \right) = 0
```

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
  <img width="700" alt="geom_ibfs45" src="https://github.com/user-attachments/assets/086dc223-5870-481e-af85-290d742fc68b" />
  <br><em>Computational domain (45° inclined step)</em>
</p>

## Boundary Conditions

| Boundary | Condition |
|---|---|
| Inlet | $u = A\,(1 - t^n)^{1/n}, \quad t = \lvert y - y_{mid} \rvert / y_{halfwidth}, \quad n = 2, \quad v = 0, \quad \tilde{\nu} = 5 / Re$ |
| Outlet | $\dfrac{\partial \tilde{\nu}}{\partial x} = 0, \quad p = 0$ |
| Top wall | $u = 0, \quad v = 0, \quad \tilde{\nu} = 0$ |
| Bottom wall | $u = 0, \quad v = 0, \quad \tilde{\nu} = 0$ |
| Horizontal wall | $u = 0, \quad v = 0, \quad \tilde{\nu} = 0$ |
| Vertical wall | $u = 0, \quad v = 0, \quad \tilde{\nu} = 0$ |


# Results



<div align="center">
<table>
  <tr>
    <th>Velocity field</th>
  </tr>
  <tr>
    <td><img width="700" alt="velStream_theta45" src="https://github.com/user-attachments/assets/2fc8b6cb-beaf-42b6-8590-a8cb54032608" /></td>
  </tr>
  <tr>
    <th>Pressure field</th>
  </tr>
  <tr>
    <td><img width="700" alt="pres_contour_theta45" src="https://github.com/user-attachments/assets/2c437794-f611-48c7-856b-0a3ac0458ff4" /></td>
  </tr>
</table>
</div>
