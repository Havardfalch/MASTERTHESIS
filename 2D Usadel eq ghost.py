# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 13:36:40 2023

@author: Havard
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 10:57:21 2023

@author: Havard
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate as spint
from linearized_Usadel_solver_2D import get_Usadel_solution
from Boundary_conditions_for_linearized_Usadel import g_bc_SN, g_bc_SNS
import time


t1 = time.time()


Nx = 50
Ny = 10
x,dx = np.linspace(0,2,Nx,retstep = True)
y, dy = np.linspace(0,2,Ny,retstep = True)

A = np.zeros((Ny, Nx,2))


epsilons1 = np.linspace(0,2,300,dtype = complex)
epsilons2 = np.linspace(2.1,30, 280, dtype = complex)
epsilons = np.append(epsilons1,epsilons2)
epsilons[0]+= 1e-5
#epsilons+= 1j*1e-1
epsilons_minus = -1*epsilons
f_sols = np.zeros((epsilons.shape[0],Ny,Nx),dtype = complex)
f_sols_minus = np.zeros((epsilons.shape[0],Ny,Nx),dtype = complex)
f_grads = np.zeros((epsilons.shape[0],Ny,Nx,2),dtype = complex)
f_grads_minus = np.zeros((epsilons.shape[0],Ny,Nx,2),dtype = complex)

thetas = np.linspace(0,4*np.pi,32)
theta = 0
currents = np.zeros(thetas.shape[0], dtype = complex)
pair_corrs = np.zeros((thetas.shape[0], Ny, Nx), dtype = complex)
"""
#Solves with BC for SN system
f_sols, f_grads = get_Usadel_solution(epsilons, f_sols, f_grads, g_bc_SN, x, y, dx, dy, A, theta = theta)

f_sols_minus,f_grads_minus = get_Usadel_solution(epsilons_minus, f_sols_minus, f_grads_minus, g_bc_SN, x, y, dx, dy, A, theta = theta)
pair_corr = spint.trapz(f_sols-f_sols_minus, x = epsilons, axis = 0)
current_x = np.trapz(f_sols*np.conjugate(f_grads_minus[:,:,:,0])- f_sols_minus*np.conjugate(f_grads[:,:,:,0]), x = epsilons, axis = 0)
current_y = np.trapz(np.real(f_sols*np.conjugate(f_grads_minus[:,:,:,1])- f_sols_minus*np.conjugate(f_grads[:,:,:,1])), x = epsilons, axis = 0)
"""

#Solves with BC for SNS system
for i in range(thetas.shape[0]):
    if (i)%(thetas.shape[0]//4)==0 and i>0:
        print(i)
    theta = thetas[i]
    f_sols, f_grads = get_Usadel_solution(epsilons, f_sols, f_grads, g_bc_SNS, x, y, dx, dy, A, theta = theta)
    

    f_sols_minus,f_grads_minus = get_Usadel_solution(epsilons_minus, f_sols_minus, f_grads_minus, g_bc_SNS, x, y, dx, dy, A, theta = theta)



    current_x = np.trapz(f_sols*np.conjugate(f_grads_minus[:,:,:,0])- f_sols_minus*np.conjugate(f_grads[:,:,:,0]), x = epsilons, axis = 0)
    current_y = np.trapz(np.real(f_sols*np.conjugate(f_grads_minus[:,:,:,1])- f_sols_minus*np.conjugate(f_grads[:,:,:,1])), x = epsilons, axis = 0)
    pair_corrs[i] = spint.trapz(f_sols-f_sols_minus, x = epsilons, axis = 0)
    currents[i] = np.average(current_x)
   

#Plotting for the SNS case
plt.plot(thetas, currents.real)
plt.title("Real current")
plt.xlabel(r'Angle $\theta$')
plt.ylabel(r'Supercurrent $J$')
plt.show()
plt.plot(thetas, currents.imag)
plt.title("Imaginary part of current")
plt.show()

#Plotting for the SN case
"""
plt.pcolormesh(x,y,current_x.real,cmap='seismic')
plt.title("Real part of current in x direction")
plt.colorbar()
plt.show()
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
