# Quantum simulation with IAO
This repository contains the code used in " Quantum simulation of molecular systems with intrinsic atomic orbitals" paper.

Reference: https://arxiv.org/abs/2011.08137




## IAO construction


Here you can find the routines to extract a little active subspace from a big basis calculation 
using the IAO procedure and use this basis to initiate a calculation on quantum computers using 
the PySCFDriver integrated in Qiskit

 - the **src** directory contains the source code. It is sufficient to copy the entire directory and from that import the methods using
 
   `from generate_hamiltonian import *`
 
 - the **molecule_geometries** contains some example of geometries that can be used
 
 - the file *example.py* contains a usage example of the methods: we contruct a molecule using the IAO procedure and use it to initialize a VQE calculation 

To be executed, the Qiskit version required is


| qiskit              | 0.20.1 |
|---------------------|--------|
| qiskit-aer          | 0.6.1  |
| qiskit-aqua         | 0.7.5  |
| qiskit-ibm-provider | 0.8.0  |
| qiskit-ignis        | 0.4.0  |
| qiskit-terra        | 0.15.2 |
