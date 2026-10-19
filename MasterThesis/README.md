# Physics-Informed Neural Networks (PINNs) for Solving RANS Equations in Duct Flows

> **Master's thesis** · 📄 [Thesis report](https://drive.google.com/file/d/1JV-8ZsBMY9zo8paNGa1WWrTm2Iohakd_/view?usp=drive_link)

This repository contains the code for approximating the Reynolds-averaged Navier–Stokes (RANS) equations in ducted geometries, as an attempt to model the turbulent flows occurring in aerospace duct geometries and turbomachinery.

The thesis is presented as a compendium of case studies to understand the capabilities of PINNs to approximate PDEs, starting from fundamental ones such as the heat, convection–diffusion and Burgers' equations (which involve some form of transport phenomena) and moving to complex ones such as Navier–Stokes and RANS.

---

## Motivation

1. To be able to model flow problems (in the context of aerospace turbomachinery and engine intakes) quickly and draw insights, for which the deep learning method, PINNs can be used.
2. Can PINNs act as alternatives to traditional PDE solvers? That is, without using data, can PINNs predict flows accurately?

## Research Questions

1. Leverage recent advancements in PINNs from a mathematical perspective to tackle their drawbacks in solving the RANS equations, predict mean flow fields, and benchmark them against ground truth.
2. Can these techniques help capture physically meaningful results?
3. Can the tedious part of any ML workflow, i.e. hyperparameter tuning of network parameters and others such as loss weights, be avoided, thereby making this method a systematic and consistent success?
4. How do changes in network parameters and different features affect the results?

## Key Takeaways

1. PINNs performed fairly well on simpler PDEs, with excellent approximation accuracy.
2. Problems arise when complex PDEs like Navier–Stokes and RANS are encountered, where performance suffers. This indicates the inability of vanilla PINNs to approximate flow fields without additional sophistication.
3. Hence, additional methods and techniques from the literature were used to capture flow separation, boundary layers and recirculation bubbles, and to get the physics right. However, the accuracy still needs improvement, which could be achieved with more sophisticated techniques such as random weight factorization, the SOAP optimizer and residual adaptive networks.

---

## Case Studies

| # | Case study | Benchmark solution |
|---|---|---|
| 1 | Burgers' equation (1D) ([code](./pinn/burgers)) | DeepXDE reference dataset ([data](https://github.com/lululxvi/deepxde/blob/master/examples/dataset/Burgers.npz)) |
| 2 | Heat equation (2D) ([code](./pinn/heat)) | FEM solution using FEniCS ([code](./fem)) |
| 3 | Convection–diffusion (2D) ([code](./pinn/convdiff)) | Wolfram Mathematica ([data](https://demonstrations.wolfram.com/SteadyStateTwoDimensionalConvectionDiffusionEquation/))|
| 4 | Navier–Stokes on a backward-facing step (Re = 389) ([code](./pinn/bfs)) | [Paper](https://www.researchgate.net/publication/232005183_Experimental_and_theoretical_investigation_of_backward-facing_step_flow) |
| 5 | RANS on an S-shaped duct (Re = 10⁶) ([code](./pinn/sduct)) | [Paper](https://www.researchgate.net/publication/226501405_Analysis_of_Three-Dimensional_Turbulent_Flow_in_an_S-Shaped_Rectangular_Channel) |
| 6 | RANS on an inclined backward-facing step (Re = 36,000) ([code](./pinn/inclinedbfs)) | [Paper](https://www.sciencedirect.com/science/article/pii/S0142727X13001409) |

> **Note:** Only the heat equation benchmark was generated in this work, using the finite element method (FEM) in FEniCS. The Burgers' benchmark is taken from DeepXDE itself, the convection–diffusion benchmark was computed in Wolfram Mathematica, and the benchmarks for the remaining three cases are taken from the literature.

---

## Implementation

**Library:** [DeepXDE](https://deepxde.readthedocs.io/en/latest/) (PyTorch backend)

**Techniques used:**

| # | Technique | Reference |
|---|---|---|
| 1 | Fourier feature embedding of inputs | [Paper](https://www.sciencedirect.com/science/article/pii/S0045782521002759) |
| 2 | Adaptive loss-term weighting (GradNorm scheme) | [Paper](https://arxiv.org/abs/2308.08468) |
| 3 | Pseudo-time stepping to avoid spurious/trivial solutions | [Paper](https://arxiv.org/abs/2604.23528) |
| 4 | Curriculum training | [Paper](https://arxiv.org/abs/2109.01050) |

> 📄 Refer to the [thesis report](https://drive.google.com/file/d/1JV-8ZsBMY9zo8paNGa1WWrTm2Iohakd_/view?usp=drive_link) for a detailed understanding of the methods.
