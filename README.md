## Coupling schemes for Fluid-Structure Interaction


Understanding of various coupling schemes used for **Fluid-Structure Interaction (FSI)** is quite challening, considering especially that FSI problems often require high-fidelity simulations. However, efforts have been made to simplify the problem and gain some insights into the behaviour of different FSI coupling schemes [1-4]. One such a simplified model is the well-known spring-mass-damper system which is recast as an FSI problem. Despite its simplicity, this problems serves as a useful model to understand the characteristics of FSI coupling schemes.

![Original](./figures/sdof-model-original.png)

![Original](./figures/sdof-model-modified.png)

This repository provides a set of Jupyter notebooks with embedded Python scripts to understand various properties and solution behaviour of different coupling schemes used for FSI. Python scripts provide functions to calculate
* spectral radius of the amplification matrix which tells us a great deal about stability and numerical damping characteristics, and
* numerical solutions of the FSI problem with different coupling schemes.

Currently, monolithic scheme and staggered schemes based on
* Dirichlet-Neumann coupling,
* force and displacement predictors, and
* Backward Euler time integration scheme 

are covered. But the concepts can be extended to other types of schemes.

#

### References
[1] C A Felippa, K C Park and C Farhat, *Partitioned analysis of coupled mechanical systems*, Computer Methods in Applied Mechanics and Engineering, 190:3247-3270, 2001.
[DOI](https://doi.org/10.1016/S0045-7825(00)00391-1)

[2] W G Dettmer and D Perić, *A new staggered scheme for fluid-structure interaction*, International Journal for Numerical Methdos in Engineering, 93:1-22, 2013.
[DOI](https://doi.org/10.1002/nme.4370)

[3] C Kadapa, *A second-order accurate non-intrusive staggered scheme for the interaction of ultra-lightweight rigid bodies with fluid flow*, Ocean Engineering, 217:107940, 2020. [DOI](https://doi.org/10.1016/j.oceaneng.2020.107940)

[4] W G Dettmer, A Lovrić, C Kadapa, D Perić, *New iterative and staggered solution schemes for incompressible fluid-structure interaction based on Dirichlet-Neumann coupling*, International Journal for Numerical Methdos in Engineering, 122:5204-5235, 2021. [DOI](https://doi.org/10.1002/nme.6494)

