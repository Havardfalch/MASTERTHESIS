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

import cProfile as profile
import pstats

a = np.linspace(0,2*np.pi,10)





t1 = time.time()
def g_bc_SSNSS(epsilon, theta, Nx, Ny, use_kl = False):
    """Boundary condition for a normal metal with a superconductor on all 4 sides with the same phase
    """
    g=np.zeros((Ny+2, Nx+2), dtype = complex)
    phase = np.exp(-1j*(np.linspace(0,2*np.pi,(2*Ny)+(2*Nx))))
    if use_kl:
        g[0,1:Nx+1] = np.sinh(np.arctanh(1/epsilon)) *phase[0:Nx]
        g[1:Ny+1,-1] = -np.sinh(np.arctanh(1/epsilon)) *phase[Nx:Ny+Nx]
        g[-1,1:Nx+1] = -np.sinh(np.arctanh(1/epsilon))*np.flip(phase[Ny+Nx:Ny+(2*Nx)])
        g[1:Ny+1,0] = np.sinh(np.arctanh(1/epsilon))*np.flip(phase[Ny+(2*Nx):(2*Ny)+(2*Nx)])
    else:
        g[0,1:Nx+1] = -np.sinh(np.arctanh(1/epsilon)) *phase[0:Nx]
        g[1:Ny+1,-1] = -np.sinh(np.arctanh(1/epsilon)) *phase[Nx:Ny+Nx]
        g[-1,1:Nx+1] = -np.sinh(np.arctanh(1/epsilon))*np.flip(phase[Ny+Nx:Ny+(2*Nx)])
        g[1:Ny+1,0] = -np.sinh(np.arctanh(1/epsilon))*np.flip(phase[Ny+(2*Nx):(2*Ny)+(2*Nx)])
    g = np.reshape(g,(Nx+2)*(Ny+2))
    #g/=3
    
    return g

Nx = 100
Ny = 100
e = -1
Lx = 2
Ly = 2
use_kl = True
x,dx = np.linspace(0,Lx,Nx,retstep = True)
y, dy = np.linspace(0,Ly,Ny,retstep = True)

A = np.zeros((Ny, Nx,2))
for i in range(Ny):
    for j in range(Nx):
        A[i,j,0] = (i-Ny/2)*(Ly/Ny)
        A[i,j,1] = -(j-Nx/2)*(Lx/Nx)
multiplier = 1*1e-1
A*=multiplier
print("A multiplier", multiplier)
if use_kl:
    print("Using Kuprianov-Lukichev bcs")
else:
    print("Not using Kuprianov-Lukichev bcs")
epsilons1 = np.linspace(0,2,300,dtype = complex)
epsilons2 = np.linspace(2.1,30, 280, dtype = complex)
epsilons = np.append(epsilons1,epsilons2)
epsilons[0]+= 1e-5
#epsilons+=1j*1e-4
epsilons_minus = -1*epsilons
f_sols = np.zeros((epsilons.shape[0],Ny,Nx),dtype = complex)
f_sols_minus = np.zeros((epsilons.shape[0],Ny,Nx),dtype = complex)
f_grads = np.zeros((epsilons.shape[0],Ny,Nx,2),dtype = complex)
f_grads_minus = np.zeros((epsilons.shape[0],Ny,Nx,2),dtype = complex)

theta = 0


#Solves with BC for N with S on all 4 sides
prof = profile.Profile()
prof.enable()
f_sols, f_grads = get_Usadel_solution(epsilons, f_sols, f_grads, g_bc_SSNSS, x, y, dx, dy, A, theta = theta, use_kl = use_kl)

f_sols_minus,f_grads_minus = get_Usadel_solution(epsilons_minus, f_sols_minus, f_grads_minus, g_bc_SSNSS, x, y, dx, dy, A, theta = theta,use_kl = use_kl)
prof.disable()
pair_corr = spint.trapz(f_sols-f_sols_minus, x = epsilons, axis = 0)
current_x = np.trapz((f_sols*np.conjugate(f_grads_minus[:,:,:,0])- f_sols_minus*np.conjugate(f_grads[:,:,:,0]) + 2*e*A[:,:,0]*1j*(f_sols*np.conjugate(f_sols_minus)-f_sols_minus*np.conjugate(f_sols))).real, x = epsilons, axis = 0)
current_y = np.trapz((f_sols*np.conjugate(f_grads_minus[:,:,:,1])- f_sols_minus*np.conjugate(f_grads[:,:,:,1])+ 2*e*A[:,:,0]*1j*(f_sols*np.conjugate(f_sols_minus)-f_sols_minus*np.conjugate(f_sols))).real, x = epsilons, axis = 0)
abs_current = np.sqrt((current_x.real)**2+(current_y.real)**2)


plt.pcolormesh(x,y,current_x.real,cmap='seismic', vmin = -np.max(np.abs(current_x.real)), vmax = np.max(np.abs(current_x.real)))
plt.colorbar()
plt.title("Current in x-direction")
plt.show()
#☺
div_current_x = current_x[:,:-2]- current_x[:,2:]
div_current_y = current_y[:-2]- current_y[2:]


plt.pcolormesh(x,y,current_y.real,cmap='seismic', vmin = -np.max(np.abs(current_y.real)), vmax = np.max(np.abs(current_y.real)))
plt.colorbar()
plt.title("Current in y-direction")
plt.show()

slc = 8
plt.quiver(x[::slc],y[::slc],current_x[::slc,::slc].real, current_y[::slc,::slc].real)
plt.title("Supercurrent")
plt.show()
"""
plt.quiver(x[::slc],y[::slc],A [::slc,::slc,0], A[::slc,::slc,1])
plt.title("Magnetic vector potential")
plt.show()

plt.pcolormesh(x[1:-1],y, div_current_x.real,cmap='seismic', vmin = -np.max(np.abs(div_current_x.real)), vmax = np.max(np.abs(div_current_x.real)))
plt.colorbar()
plt.title("Divergence of current in x-direction")
plt.show()
plt.pcolormesh(x,y[1:-1], div_current_y.real,cmap='seismic', vmin = -np.max(np.abs(div_current_y.real)), vmax = np.max(np.abs(div_current_y.real)))
plt.colorbar()
plt.title("Divergence of current in y-direction")
plt.show()
"""
plt.pcolormesh(x,y,abs_current, cmap = 'seismic')
plt.colorbar()
plt.title("Absolute value of current")
plt.show()

# print profiling output
#stats = pstats.Stats(prof).strip_dirs().sort_stats("cumtime")
#stats.print_stats(10)


plt.pcolormesh(x,y,pair_corr.real,cmap='seismic')
plt.title("Real part of pair correlation")
plt.colorbar()
plt.show()
plt.pcolormesh(x,y,pair_corr.imag,cmap='seismic',vmin = -np.max(np.abs(pair_corr.imag)), vmax = np.max(np.abs(pair_corr.imag)))
plt.title("Imaginary part of pair correlation")
plt.colorbar()
plt.show()

t2 = time.time()
print("Time taken", t2-t1)
