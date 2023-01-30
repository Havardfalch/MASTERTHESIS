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
import time

t1 = time.time()
def g_bc_SNS(epsilon, theta, Nx, Ny):
    g=np.zeros((Ny+2, Nx+2), dtype = complex)
    g[1:Ny+1,0] = -np.sinh(np.arctanh(1/epsilon))* np.exp(1j*theta/2)
    g[1:Ny+1,-1] = -np.sinh(np.arctanh(1/epsilon)) * np.exp(-1j*theta/2)

    g = np.reshape(g,(Nx+2)*(Ny+2))
    return g

def g_bc_SN(epsilon, theta, Nx, Ny):
    g=np.zeros((Ny+2, Nx+2), dtype = complex)
    g[1:Ny+1,0] = -np.sinh(np.arctanh(1/epsilon))
    g = np.reshape(g,(Nx+2)*(Ny+2))
    return g

Nx = 100
Ny = 10
x,dx = np.linspace(0,2,Nx,retstep = True)
y, dy = np.linspace(0,2,Ny,retstep = True)
f = x

A = np.zeros((Ny, Nx,2))
"""for i in range(Ny):
    for j in range(Nx):
        A[i,j,0] = j*0.001
        A[i,j,1] = i*0.001
"""

epsilons1 = np.linspace(0,2,300,dtype = complex)
epsilons2 = np.linspace(2.1,30, 280, dtype = complex)
epsilons = np.append(epsilons1,epsilons2)
#epsilons[0]+= 1e-5
epsilons+= 1j*1e-6
epsilons_minus = -1*epsilons
f_sols = np.zeros((epsilons.shape[0],Ny,Nx),dtype = complex)
f_sols_minus = np.zeros((epsilons.shape[0],Ny,Nx),dtype = complex)
f_grads = np.zeros((epsilons.shape[0],Ny,Nx,2),dtype = complex)
f_grads_minus = np.zeros((epsilons.shape[0],Ny,Nx,2),dtype = complex)

thetas = np.linspace(0,2*np.pi,200)
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


plt.plot(thetas, currents.real)
plt.title("Real current")
plt.xlabel(r'Angle $\theta$')
plt.ylabel(r'Supercurrent $J$')
plt.show()
plt.plot(thetas, currents.imag)
plt.title("Imaginary part of current")
plt.show()
f_av = np.sum(f_sols, axis = 1)/f_sols.shape[1]
pair_corr = spint.trapz(f_sols-f_sols_minus, x = epsilons, axis = 0)
av_f = spint.trapz(f_sols, dx = dy, axis = 1)

plt.pcolormesh(x,y,pair_corr.real,cmap='seismic')
plt.title("Real part of pair correlation")
plt.colorbar()
plt.show()
plt.pcolormesh(x,y,pair_corr.imag,cmap='seismic')
plt.title("Imaginary part of pair correlation")
plt.colorbar()
plt.show()


t2 = time.time()
print("Time taken", t2-t1)
