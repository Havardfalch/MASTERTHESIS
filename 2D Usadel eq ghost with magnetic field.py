# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 13:55:36 2023

@author: Havard
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate as spint
from linearized_Usadel_solver_2D import get_Usadel_solution, update_A_field
from Boundary_conditions_for_linearized_Usadel import g_bc_SSNSS
import time
import matplotlib.colors as colors

import cProfile as profile
import pstats



t1 = time.time()




Nx = 200
Ny = 200
e = -1
Lx = 12
Ly = 12
D = 1
n = 3
theta = n*2*np.pi
epsilon_min = 0
tol = 1e-3
gamma = 3
use_kl = False
x,dx = np.linspace(-Lx/2,Lx/2,Nx,retstep = True)
y, dy = np.linspace(-Ly/2,Ly/2,Ny,retstep = True)
max_its = 1


A = np.zeros((Ny, Nx,2))
for i in range(Ny):
    for j in range(Nx):
        A[i,j,0] = (i-(Ny-1)/2)*n*np.pi/(2*np.max(y)*Ny)
        A[i,j,1] = -(j-(Nx-1)/2)*n*np.pi/(2*np.max(x)*Nx)
multiplier = 0
A*=multiplier


print("Max of A", np.max(A))
print("Total phase in SC", theta)
if use_kl:
    print("Using Kuprianov-Lukichev bcs")
else:
    print("Using transparent bcs")
print("Applied voltage", epsilon_min)
print("Size in x:", Lx, "Points in x:", Nx, "Size in y:", Ly, "Points in y:", Ny)


epsilons1 = np.linspace(epsilon_min,2,300,dtype = complex)
epsilons2 = np.linspace(2,30, 280, dtype = complex)
epsilons = np.append(epsilons1,epsilons2)
epsilons[0]+= 1e-5

epsilons_minus = -epsilons

epsilons+= -1j*(1e-4)
epsilons_minus+= -1j*(1e-4)

A_delta = np.zeros_like(A)
old_A_delta = np.ones_like(A)
num_its = 0
#Solves with BC for N with S on all 4 sides
#prof = profile.Profile()
#prof.enable()
#while np.linalg.norm(A_delta-old_A_delta)>tol and num_its<max_its:
f_sols = np.zeros((epsilons.shape[0],Ny,Nx),dtype = complex)
f_sols_minus = np.zeros((epsilons.shape[0],Ny,Nx),dtype = complex)
f_grads = np.zeros((epsilons.shape[0],Ny,Nx,2),dtype = complex)
f_grads_minus = np.zeros((epsilons.shape[0],Ny,Nx,2),dtype = complex)


f_sols, f_grads = get_Usadel_solution(epsilons, f_sols, f_grads, g_bc_SSNSS, x, y, dx, dy, A, theta = theta, use_kl = use_kl,D=D)

f_sols_minus,f_grads_minus = get_Usadel_solution(epsilons_minus, f_sols_minus, f_grads_minus, g_bc_SSNSS, x, y, dx, dy, A, theta = theta,use_kl = use_kl,D=D)

pair_corr = spint.trapz(f_sols-f_sols_minus, x = epsilons, axis = 0)
abs_corr = np.abs(pair_corr)
current_x = spint.trapz((f_sols*np.conjugate(f_grads_minus[:,:,:,0])- f_sols_minus*np.conjugate(f_grads[:,:,:,0]) + 2*e*(A)[:,:,0]*1j*(f_sols*np.conjugate(f_sols_minus)-f_sols_minus*np.conjugate(f_sols))).real, x = epsilons, axis = 0).real
current_y = spint.trapz((f_sols*np.conjugate(f_grads_minus[:,:,:,1])- f_sols_minus*np.conjugate(f_grads[:,:,:,1]) + 2*e*(A)[:,:,1]*1j*(f_sols*np.conjugate(f_sols_minus)-f_sols_minus*np.conjugate(f_sols))).real, x = epsilons, axis = 0).real
abs_current = np.sqrt((current_x.real)**2+(current_y.real)**2)
xv,yv = np.meshgrid(x,y)
#old_A_delta = np.copy(A_delta)
#A_delta = update_A_field(xv, yv, current_x, current_y, dx, dy)
num_its+=1
'''print(np.max(A_delta), np.min(A_delta))
    print("Sum", np.sum(A_delta), "Average", np.average(A_delta), "Average abs", np.average(np.abs(A_delta)))
    print("Norm of difference from last iteration:", np.linalg.norm(A_delta-old_A_delta))'''

#prof.disable()
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

slc = Nx//20
plt.quiver(x[::slc],y[::slc],current_x[::slc,::slc].real, current_y[::slc,::slc].real)
plt.title("Supercurrent")
plt.show()

plt.quiver(x[::slc],y[::slc],(np.abs(current_x[::slc,::slc].real)**(1/4))*np.sign(current_x[::slc,::slc].real), (np.abs(current_y[::slc,::slc].real)**(1/4))*np.sign(current_y[::slc,::slc].real), angles = (np.arctan2(current_y[::slc,::slc].real, current_x[::slc,::slc].real)*180.0/np.pi))
plt.title("Fourth root of supercurrent")
plt.show()

plt.quiver(x[::slc],y[::slc],(A+A_delta) [::slc,::slc,0], (A)[::slc,::slc,1])
plt.title("Magnetic vector potential")
plt.show()
"""
plt.pcolormesh(x[1:-1],y, div_current_x.real,cmap='seismic', vmin = -np.max(np.abs(div_current_x.real)), vmax = np.max(np.abs(div_current_x.real)))
plt.colorbar()
plt.title("Divergence of current in x-direction")
plt.show()
plt.pcolormesh(x,y[1:-1], div_current_y.real,cmap='seismic', vmin = -np.max(np.abs(div_current_y.real)), vmax = np.max(np.abs(div_current_y.real)))
plt.colorbar()
plt.title("Divergence of current in y-direction")
plt.show()

plt.pcolormesh(x[1:-1],y[1:-1], div_current_x.real[1:-1] + div_current_y.real[:,1:-1],cmap='seismic')
plt.colorbar()
plt.title("Divergence of current")
plt.show()
"""
plt.pcolormesh(x,y,abs_current, cmap = 'seismic')
plt.colorbar()
plt.title("Absolute value of current")
plt.show()

plt.pcolormesh(x,y,abs_current,norm=colors.LogNorm(vmin=np.min(abs_current), vmax=np.max(abs_current)), cmap = 'seismic')
plt.colorbar()
plt.title("Absolute value of current")
plt.show()




"""
plt.pcolormesh(x,y,pair_corr.real,cmap='seismic')
plt.title("Real part of pair correlation")
plt.colorbar()
plt.show()
plt.pcolormesh(x,y,pair_corr.imag,cmap='seismic',vmin = -np.max(np.abs(pair_corr.imag)), vmax = np.max(np.abs(pair_corr.imag)))
plt.title("Imaginary part of pair correlation")
plt.colorbar()
plt.show()
"""
plt.pcolormesh(x,y,abs_corr,cmap='seismic', norm=colors.LogNorm(vmin = np.min(abs_corr), vmax = np.max(np.abs(abs_corr))))
plt.title("Absolute value of pair correlation")
plt.colorbar()
plt.show()
phases = np.arctan(pair_corr.imag/pair_corr.real)  #Why use arctan and not arctan2?
plt.pcolormesh(x,y,phases,cmap='seismic',vmin = np.min(phases), vmax = np.max(phases))
plt.title("Phase of pair correlation")
plt.colorbar()
plt.show()
# print profiling output
#stats = pstats.Stats(prof).strip_dirs().sort_stats("cumtime")
#stats.print_stats(10)

t2 = time.time()
print("Time taken", t2-t1)
