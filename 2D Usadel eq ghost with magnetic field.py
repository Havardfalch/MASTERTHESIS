# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 13:55:36 2023

@author: Havard
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate as spint
from linearized_Usadel_solver_2D import get_Usadel_solution
import time

t1 = time.time()
def g_bc_SSNSS(epsilon, theta, Nx, Ny):
    """Boundary condition for a normal metal with a superconductor on all 4 sides with the same phase
    """
    g=np.zeros((Ny+2, Nx+2), dtype = complex)
    g[1:Ny+1,0] = -np.sinh(np.arctanh(1/epsilon))
    g[1:Ny+1,-1] = -np.sinh(np.arctanh(1/epsilon))
    g[0,1:Nx+1] = -np.sinh(np.arctanh(1/epsilon))
    g[-1,1:Nx+1] = -np.sinh(np.arctanh(1/epsilon))
    g = np.reshape(g,(Nx+2)*(Ny+2))
    return g

Nx = 101
Ny = 101
x,dx = np.linspace(0,10,Nx,retstep = True)
y, dy = np.linspace(0,10,Ny,retstep = True)

A = np.zeros((Ny, Nx,2))
for i in range(Ny):
    for j in range(Nx):
        A[i,j,0] = (i-Ny/2)
        A[i,j,1] = -(j-Nx/2)
A*=5e-3

epsilons1 = np.linspace(0,2,300,dtype = complex)
epsilons2 = np.linspace(2.1,30, 280, dtype = complex)
epsilons = np.append(epsilons1,epsilons2)
epsilons[0]+= 1e-5

epsilons_minus = -1*epsilons
f_sols = np.zeros((epsilons.shape[0],Ny,Nx),dtype = complex)
f_sols_minus = np.zeros((epsilons.shape[0],Ny,Nx),dtype = complex)
f_grads = np.zeros((epsilons.shape[0],Ny,Nx,2),dtype = complex)
f_grads_minus = np.zeros((epsilons.shape[0],Ny,Nx,2),dtype = complex)

theta = 0


#Solves with BC for N with S on all 4 sides
f_sols, f_grads = get_Usadel_solution(epsilons, f_sols, f_grads, g_bc_SSNSS, x, y, dx, dy, A, theta = theta)

f_sols_minus,f_grads_minus = get_Usadel_solution(epsilons_minus, f_sols_minus, f_grads_minus, g_bc_SSNSS, x, y, dx, dy, A, theta = theta)

pair_corr = spint.trapz(f_sols-f_sols_minus, x = epsilons, axis = 0)
current_x = np.trapz(f_sols*np.conjugate(f_grads_minus[:,:,:,0])- f_sols_minus*np.conjugate(f_grads[:,:,:,0]), x = epsilons, axis = 0)
current_y = np.trapz(f_sols*np.conjugate(f_grads_minus[:,:,:,1])- f_sols_minus*np.conjugate(f_grads[:,:,:,1]), x = epsilons, axis = 0)
print(np.average(current_x), np.average(current_y))
plt.pcolormesh(x,y,current_x.real,cmap='seismic', vmin = -np.max(np.abs(current_x.real)), vmax = np.max(np.abs(current_x.real)))
plt.colorbar()
plt.title("Current in x-direction")
plt.show()

plt.pcolormesh(x,y,current_y.real,cmap='seismic', vmin = -np.max(np.abs(current_y.real)), vmax = np.max(np.abs(current_y.real)))
plt.colorbar()
plt.title("Current in y-direction")
plt.show()
plt.quiver(x[::2],y[::2],current_x[::2,::2], current_y[::2,::2])
plt.title("Supercurrent")
plt.show()
plt.quiver(x[::2],y[::2],A [::2,::2,0], A[::2,::2,1])
plt.title("Magnetic vector potential")
plt.show()
"""
plt.pcolormesh(x,y,pair_corr.real,cmap='seismic')
plt.title("Real part of pair correlation")
plt.colorbar()
plt.show()
plt.pcolormesh(x,y,pair_corr.imag,cmap='seismic')
plt.title("Imaginary part of pair correlation")
plt.colorbar()
plt.show()
"""

t2 = time.time()
print("Time taken", t2-t1)
