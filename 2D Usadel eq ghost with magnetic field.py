# -*- coding: utf-8 -*-
"""
Created on Thu May  4 13:17:21 2023

@author: Havard
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 13:46:25 2023

@author: Havard
"""

from numpy import sqrt, pi, abs, arctan2, max, conjugate, cos, where, gradient, min, save
import numpy as np

from scipy.integrate import trapz

import matplotlib.pyplot as plt

from linearized_Usadel_solver_2D import get_Usadel_solution, update_A_field
from Boundary_conditions_for_linearized_Usadel import g_bc_SSNSS, calculate_correct_A_field
from scipy import integrate as spint

#Set parameters for the run

Nx = 400
Ny = Nx
e = -1
Lx = 8
Ly = Lx
D = 1
n = 1
theta = -n*2*np.pi
eV = 0
e_ = -1
tol = 1e-3
gamma = 3
use_kl = True
x,dx = np.linspace(-Lx/2,Lx/2,Nx,retstep = True)
y, dy = np.linspace(-Ly/2,Ly/2,Ny,retstep = True)
xv, yv = np.meshgrid(x,y)
max_its = 1

#Set the value of the magnetic field density and the vector potential at the edges
B0 = n*np.pi/((Lx+(0*dx))*(Ly+(0*dx)))
B_and_bc = np.zeros((Ny+2, Nx+2,2))
B_and_bc[0,1:-1,0] = n*np.pi/(4*Lx)
B_and_bc[-1,1:-1,0] = -n*np.pi/(4*Lx)
B_and_bc[1:-1,0,1] = -n*np.pi/(4*Ly)
B_and_bc[1:-1,-1,1] = n*np.pi/(4*Ly)
B_and_bc[1:-1,1:-1,1] = B0

#Calculate the vector potential
A = calculate_correct_A_field(B_and_bc, Nx, Ny, dx, dy)[1:-1,1:-1]

multiplier = 1
A*=multiplier
B = np.gradient(A[:,:,1], dx, axis = 1) -  np.gradient(A[:,:,0], dy, axis = 0)

#Create an array with all the energies to solve for, we also need the negative energies
epsilons1 = np.linspace(eV,2,600,dtype = complex)
epsilons2 = np.linspace(2,30, 280, dtype = complex)
epsilons = np.append(epsilons1,epsilons2)

epsilons_minus = -epsilons

#Add an imaginary part to stabilize solutions and as experimentally noted by Dynes et al.
epsilons+= 1j*(1e-2)
epsilons_minus+= 1j*(1e-2)

#Create arrays to store the Green's function values and their gradients in
f_sols = np.zeros((epsilons.shape[0],Ny,Nx),dtype = complex)
f_sols_minus = np.zeros((epsilons.shape[0],Ny,Nx),dtype = complex)
f_grads = np.zeros((epsilons.shape[0],Ny,Nx,2),dtype = complex)
f_grads_minus = np.zeros((epsilons.shape[0],Ny,Nx,2),dtype = complex)

#Solve the linearized Usadel equation for positive energies
f_sols, f_grads = get_Usadel_solution(epsilons, f_sols, f_grads, g_bc_SSNSS, x, y, dx, dy, A, theta = theta, use_kl = use_kl,D=D, gamma = gamma)

#Solve the linearized Usadel equation for negative energies
f_sols_minus,f_grads_minus = get_Usadel_solution(epsilons_minus, f_sols_minus, f_grads_minus, g_bc_SSNSS, x, y, dx, dy, A, theta = theta,use_kl = use_kl,D=D, gamma = gamma)


#Calculate the form of the pair correlation
pair_corr = spint.trapz(f_sols-f_sols_minus, x = epsilons, axis = 0)
abs_corr = np.abs(pair_corr)

#Calculate the supercurrent in the x and y directions
current_x = e/2*spint.trapz((f_sols*np.conjugate(f_grads_minus[:,:,:,0])- f_sols_minus*np.conjugate(f_grads[:,:,:,0]) + 2*e*(A)[:,:,0]*1j*(f_sols*np.conjugate(f_sols_minus)-f_sols_minus*np.conjugate(f_sols))).real, x = epsilons, axis = 0).real
current_y = e/2*spint.trapz((f_sols*np.conjugate(f_grads_minus[:,:,:,1])- f_sols_minus*np.conjugate(f_grads[:,:,:,1]) + 2*e*(A)[:,:,1]*1j*(f_sols*np.conjugate(f_sols_minus)-f_sols_minus*np.conjugate(f_sols))).real, x = epsilons, axis = 0).real
abs_current = sqrt((current_x.real)**2+(current_y.real)**2)

circ_phase = -np.arctan2(yv,xv)
circulating_current = abs_current*cos((circ_phase - arctan2(current_x, current_y))%(2*pi))

#Calculate the spectral current density
plot_integrand = (f_sols*conjugate(f_grads_minus[:,:,:,1])- f_sols_minus*conjugate(f_grads[:,:,:,1]) + 2*e_*(A)[:,:,1]*1j*(f_sols*conjugate(f_sols_minus)-f_sols_minus*conjugate(f_sols))).real[:600, Ny//2,Nx//2:]

#Plot the spectral current density
plt.pcolormesh(xv[Ny//2, Nx//2:], epsilons[:600].real, plot_integrand, cmap = 'seismic', vmin = -max(abs(plot_integrand)), vmax = max(abs(plot_integrand)))
plt.colorbar()
plt.title(r'$L = $'+f'{Lx}' r'$\xi$')
plt.xlabel(r'x/\xi')
plt.ylabel(r'y/\xi')
plt.gca().set_aspect('equal')
plt.tight_layout()
plt.savefig(f'Spectral current density for n = {n} L={Lx} N={Nx} eV={eV}.pdf', bbox_inches = "tight")

plt.show()

#Calculate the A and B fields induced by the supercurrents
A_delta = update_A_field(xv, yv, current_x, current_y, dx, dy)
B_induced = gradient(A_delta[:,:,1], dx, axis = 1) -  gradient(A_delta[:,:,0], dy, axis = 0)

#Calculate the applied B field
B_app = gradient(A[:,:,1], dx, axis = 1) -  gradient(A[:,:,0], dy, axis = 0)

#Save data
save(f"Applied B field for n = {n} L={Lx} N={Nx} eV={eV}", B_app)
save(f"x current for n = {n} L={Lx} N={Nx} eV={eV}", current_x)
save(f"y current for n = {n} L={Lx} N={Nx} eV={eV}", current_y)

#Set the font size for plotting
plt.rcParams.update({'font.size': 20})

#Plot the applied magnetic field
plt.pcolormesh(x,y,B_app,cmap='RdYlBu_r', vmin = -max(abs(B_app)), vmax = max(abs(B_app)))
plt.colorbar()

plt.title('Applied magnetic field')
plt.xlabel('x')
plt.ylabel('y')
plt.gca().set_aspect('equal')
plt.savefig(f'Applied magnetic fields for n = {n} L={Lx} N={Nx} eV={eV}.pdf')
plt.gca().set_aspect('equal')
plt.savefig(f'Applied magnetic fields for n = {n} L={Lx} N={Nx} eV={eV}.png')

plt.close()

#Plot the induced magnetic field
plt.pcolormesh(x,y,B_induced,cmap='RdYlBu_r', vmin = -max(abs(B_induced)), vmax = max(abs(B_induced)))
plt.colorbar()
plt.title(f'eV = {eV}')
plt.xlabel('x')
plt.ylabel('y')
plt.gca().set_aspect('equal')
plt.savefig(f'Induced magnetic fields for n = {n} L={Lx} N={Nx} eV={eV}.pdf')
plt.gca().set_aspect('equal')
plt.savefig(f'Induced magnetic fields for n = {n} L={Lx} N={Nx} eV={eV}.png')

plt.close()

#Plot the circulation of the current
plt.pcolormesh(x,y,circulating_current, cmap = 'seismic', vmin = -max(abs(circulating_current)), vmax = max(abs(circulating_current)))
plt.colorbar()
plt.title(f'eV = {eV}')
plt.xlabel('x')
plt.ylabel('y')
plt.gca().set_aspect('equal')
plt.savefig(f'Circulating current for n = {n} L={Lx} N={Nx} eV={eV}.pdf')
plt.gca().set_aspect('equal')
plt.savefig(f'Circulating current for n = {n} L={Lx} N={Nx} eV={eV}.png')

plt.close()

#Plot the current density
plt.streamplot(xv, yv, current_x, current_y, color = abs_current, cmap ="viridis")
plt.xlim(min(x), max(x))
plt.colorbar()
plt.title(f'eV = {eV}')
plt.xlabel('x')
plt.ylabel('y')
plt.gca().set_aspect('equal')
plt.savefig(f'Streamplot of current for n = {n} L={Lx} N={Nx} eV={eV}.pdf')
plt.gca().set_aspect('equal')
plt.savefig(f'Streamplot of current for n = {n} L={Lx} N={Nx} eV={eV}.png')

plt.close()

eV = 0.15

#Find at what index the applied voltage is situated and only include energies larger than the applied voltage
cutoff_ind = max(where(epsilons.real<eV))+1
f_sols_new = f_sols[cutoff_ind:]
f_grads_new = f_grads[cutoff_ind:]
f_sols_minus_new = f_sols_minus[cutoff_ind:]
f_grads_minus_new = f_grads_minus[cutoff_ind:]
epsilons_new = epsilons[cutoff_ind:]

#Calculate the supercurrent in the x and y directions
current_x = e_*trapz((f_sols_new*conjugate(f_grads_minus_new[:,:,:,0])- f_sols_minus_new*conjugate(f_grads_new[:,:,:,0]) + 2*e_*(A)[:,:,0]*1j*(f_sols_new*conjugate(f_sols_minus_new)-f_sols_minus_new*conjugate(f_sols_new))).real, x = epsilons_new, axis = 0).real
current_y = e_*trapz((f_sols_new*conjugate(f_grads_minus_new[:,:,:,1])- f_sols_minus_new*conjugate(f_grads_new[:,:,:,1]) + 2*e_*(A)[:,:,1]*1j*(f_sols_new*conjugate(f_sols_minus_new)-f_sols_minus_new*conjugate(f_sols_new))).real, x = epsilons_new, axis = 0).real

abs_current = sqrt((current_x.real)**2+(current_y.real)**2)
circulating_current = abs_current*cos((circ_phase - arctan2(current_x, current_y))%(2*pi))

#Save data
save(f"x current for n = {n} L={Lx} N={Nx} eV={eV}", current_x)
save(f"y current for n = {n} L={Lx} N={Nx} eV={eV}", current_y)

#Calculate the A and B fields induced by the supercurrents
A_delta = update_A_field(xv, yv, current_x, current_y, dx, dy)
B_induced = gradient(A_delta[:,:,1], dx, axis = 1) -  gradient(A_delta[:,:,0], dy, axis = 0)
B_app = gradient(A[:,:,1], dx, axis = 1) -  gradient(A[:,:,0], dy, axis = 0)

#Plot the induced magnetic field
plt.pcolormesh(x,y,B_induced,cmap='RdYlBu_r', vmin = -max(abs(B_induced)), vmax = max(abs(B_induced)))
plt.colorbar()
plt.title(f'eV = {eV}')
plt.xlabel('x')
plt.ylabel('y')
plt.gca().set_aspect('equal')
plt.savefig(f'Induced magnetic fields for n = {n} L={Lx} N={Nx} eV={eV}.pdf')
plt.gca().set_aspect('equal')
plt.savefig(f'Induced magnetic fields for n = {n} L={Lx} N={Nx} eV={eV}.png')
plt.close()

#Plot the circulation of the current
plt.pcolormesh(x,y,circulating_current, cmap = 'seismic', vmin = -max(abs(circulating_current)), vmax = max(abs(circulating_current)))
plt.colorbar()
plt.title(f'eV = {eV}')
plt.xlabel('x')
plt.ylabel('y')
plt.gca().set_aspect('equal')
plt.savefig(f'Circulating current for n = {n} L={Lx} N={Nx} eV={eV}.pdf')
plt.gca().set_aspect('equal')
plt.savefig(f'Circulating current for n = {n} L={Lx} N={Nx} eV={eV}.png')
plt.close()

#Plot the current density
plt.streamplot(xv, yv, current_x, current_y, color = abs_current, cmap ="viridis")
plt.colorbar()
plt.title(f'eV = {eV}')
plt.xlabel('x')
plt.ylabel('y')
plt.gca().set_aspect('equal')
plt.savefig(f'Streamplot of current for n = {n} L={Lx} N={Nx} eV={eV}.pdf')
plt.gca().set_aspect('equal')
plt.savefig(f'Streamplot of current for n = {n} L={Lx} N={Nx} eV={eV}.png')
plt.close()

eV = 0.25

#Find at what index the applied voltage is situated and only include energies larger than the applied voltage
cutoff_ind = max(where(epsilons.real<eV))+1
f_sols_new = f_sols[cutoff_ind:]
f_grads_new = f_grads[cutoff_ind:]
f_sols_minus_new = f_sols_minus[cutoff_ind:]
f_grads_minus_new = f_grads_minus[cutoff_ind:]
epsilons_new = epsilons[cutoff_ind:]

#Calculate the supercurrent in the x and y directions
current_x = e_*trapz((f_sols_new*conjugate(f_grads_minus_new[:,:,:,0])- f_sols_minus_new*conjugate(f_grads_new[:,:,:,0]) + 2*e_*(A)[:,:,0]*1j*(f_sols_new*conjugate(f_sols_minus_new)-f_sols_minus_new*conjugate(f_sols_new))).real, x = epsilons_new, axis = 0).real
current_y = e_*trapz((f_sols_new*conjugate(f_grads_minus_new[:,:,:,1])- f_sols_minus_new*conjugate(f_grads_new[:,:,:,1]) + 2*e_*(A)[:,:,1]*1j*(f_sols_new*conjugate(f_sols_minus_new)-f_sols_minus_new*conjugate(f_sols_new))).real, x = epsilons_new, axis = 0).real

abs_current = sqrt((current_x.real)**2+(current_y.real)**2)
circulating_current = abs_current*cos((circ_phase - arctan2(current_x, current_y))%(2*pi))

#Save data
save(f"x current for n = {n} L={Lx} N={Nx} eV={eV}", current_x)
save(f"y current for n = {n} L={Lx} N={Nx} eV={eV}", current_y)

#Calculate the A and B fields induced by the supercurrents
A_delta = update_A_field(xv, yv, current_x, current_y, dx, dy)
B_induced = gradient(A_delta[:,:,1], dx, axis = 1) -  gradient(A_delta[:,:,0], dy, axis = 0)
B_app = gradient(A[:,:,1], dx, axis = 1) -  gradient(A[:,:,0], dy, axis = 0)

#Plot the induced magnetic field
plt.pcolormesh(x,y,B_induced,cmap='RdYlBu_r', vmin = -max(abs(B_induced)), vmax = max(abs(B_induced)))
plt.colorbar()
plt.title(f'eV = {eV}')
plt.xlabel('x')
plt.ylabel('y')
plt.gca().set_aspect('equal')
plt.savefig(f'Induced magnetic fields for n = {n} L={Lx} N={Nx} eV={eV}.pdf')
plt.gca().set_aspect('equal')
plt.savefig(f'Induced magnetic fields for n = {n} L={Lx} N={Nx} eV={eV}.png')
plt.close()

#Plot the circulation of the current
plt.pcolormesh(x,y,circulating_current, cmap = 'seismic', vmin = -max(abs(circulating_current)), vmax = max(abs(circulating_current)))
plt.colorbar()
plt.title(f'eV = {eV}')
plt.xlabel('x')
plt.ylabel('y')
plt.gca().set_aspect('equal')
plt.savefig(f'Circulating current for n = {n} L={Lx} N={Nx} eV={eV}.pdf')
plt.gca().set_aspect('equal')
plt.savefig(f'Circulating current for n = {n} L={Lx} N={Nx} eV={eV}.png')
plt.close()

#Plot the current density
plt.streamplot(xv, yv, current_x, current_y, color = abs_current, cmap ="viridis")
plt.colorbar()
plt.title(f'eV = {eV}')
plt.xlabel('x')
plt.ylabel('y')
plt.gca().set_aspect('equal')
plt.savefig(f'Streamplot of current for n = {n} L={Lx} N={Nx} eV={eV}.pdf')
plt.gca().set_aspect('equal')
plt.savefig(f'Streamplot of current for n = {n} L={Lx} N={Nx} eV={eV}.png')
plt.close()


eV = 0.35

#Find at what index the applied voltage is situated and only include energies larger than the applied voltage
cutoff_ind = max(where(epsilons.real<eV))+1
f_sols_new = f_sols[cutoff_ind:]
f_grads_new = f_grads[cutoff_ind:]
f_sols_minus_new = f_sols_minus[cutoff_ind:]
f_grads_minus_new = f_grads_minus[cutoff_ind:]
epsilons_new = epsilons[cutoff_ind:]

#Calculate the supercurrent in the x and y directions
current_x = e_*trapz((f_sols_new*conjugate(f_grads_minus_new[:,:,:,0])- f_sols_minus_new*conjugate(f_grads_new[:,:,:,0]) + 2*e_*(A)[:,:,0]*1j*(f_sols_new*conjugate(f_sols_minus_new)-f_sols_minus_new*conjugate(f_sols_new))).real, x = epsilons_new, axis = 0).real
current_y = e_*trapz((f_sols_new*conjugate(f_grads_minus_new[:,:,:,1])- f_sols_minus_new*conjugate(f_grads_new[:,:,:,1]) + 2*e_*(A)[:,:,1]*1j*(f_sols_new*conjugate(f_sols_minus_new)-f_sols_minus_new*conjugate(f_sols_new))).real, x = epsilons_new, axis = 0).real

abs_current = sqrt((current_x.real)**2+(current_y.real)**2)
circulating_current = abs_current*cos((circ_phase - arctan2(current_x, current_y))%(2*pi))

#Save data
save(f"x current for n = {n} L={Lx} N={Nx} eV={eV}", current_x)
save(f"y current for n = {n} L={Lx} N={Nx} eV={eV}", current_y)

#Calculate the A and B fields induced by the supercurrents
A_delta = update_A_field(xv, yv, current_x, current_y, dx, dy)
B_induced = gradient(A_delta[:,:,1], dx, axis = 1) -  gradient(A_delta[:,:,0], dy, axis = 0)
B_app = gradient(A[:,:,1], dx, axis = 1) -  gradient(A[:,:,0], dy, axis = 0)

#Plot the induced magnetic field
plt.pcolormesh(x,y,B_induced,cmap='RdYlBu_r', vmin = -max(abs(B_induced)), vmax = max(abs(B_induced)))
plt.colorbar()
plt.title(f'eV = {eV}')
plt.xlabel('x')
plt.ylabel('y')
plt.gca().set_aspect('equal')
plt.savefig(f'Induced magnetic fields for n = {n} L={Lx} N={Nx} eV={eV}.pdf')
plt.gca().set_aspect('equal')
plt.savefig(f'Induced magnetic fields for n = {n} L={Lx} N={Nx} eV={eV}.png')
plt.close()

#Plot the circulation of the current
plt.pcolormesh(x,y,circulating_current, cmap = 'seismic', vmin = -max(abs(circulating_current)), vmax = max(abs(circulating_current)))
plt.colorbar()
plt.title(f'eV = {eV}')
plt.xlabel('x')
plt.ylabel('y')
plt.gca().set_aspect('equal')
plt.savefig(f'Circulating current for n = {n} L={Lx} N={Nx} eV={eV}.pdf')
plt.gca().set_aspect('equal')
plt.savefig(f'Circulating current for n = {n} L={Lx} N={Nx} eV={eV}.png')
plt.close()

#Plot the current density
plt.streamplot(xv, yv, current_x, current_y, color = abs_current, cmap ="viridis")
plt.colorbar()
plt.title(f'eV = {eV}')
plt.xlabel('x')
plt.ylabel('y')
plt.gca().set_aspect('equal')
plt.savefig(f'Streamplot of current for n = {n} L={Lx} N={Nx} eV={eV}.pdf')
plt.gca().set_aspect('equal')
plt.savefig(f'Streamplot of current for n = {n} L={Lx} N={Nx} eV={eV}.png')
plt.close()



eV = 0.5

#Find at what index the applied voltage is situated and only include energies larger than the applied voltage
cutoff_ind = max(where(epsilons.real<eV))+1
f_sols_new = f_sols[cutoff_ind:]
f_grads_new = f_grads[cutoff_ind:]
f_sols_minus_new = f_sols_minus[cutoff_ind:]
f_grads_minus_new = f_grads_minus[cutoff_ind:]
epsilons_new = epsilons[cutoff_ind:]

#Calculate the supercurrent in the x and y directions
current_x = e_*trapz((f_sols_new*conjugate(f_grads_minus_new[:,:,:,0])- f_sols_minus_new*conjugate(f_grads_new[:,:,:,0]) + 2*e_*(A)[:,:,0]*1j*(f_sols_new*conjugate(f_sols_minus_new)-f_sols_minus_new*conjugate(f_sols_new))).real, x = epsilons_new, axis = 0).real
current_y = e_*trapz((f_sols_new*conjugate(f_grads_minus_new[:,:,:,1])- f_sols_minus_new*conjugate(f_grads_new[:,:,:,1]) + 2*e_*(A)[:,:,1]*1j*(f_sols_new*conjugate(f_sols_minus_new)-f_sols_minus_new*conjugate(f_sols_new))).real, x = epsilons_new, axis = 0).real

abs_current = sqrt((current_x.real)**2+(current_y.real)**2)
circulating_current = abs_current*cos((circ_phase - arctan2(current_x, current_y))%(2*pi))

#Save data
save(f"x current for n = {n} L={Lx} N={Nx} eV={eV}", current_x)
save(f"y current for n = {n} L={Lx} N={Nx} eV={eV}", current_y)

#Calculate the A and B fields induced by the supercurrents
A_delta = update_A_field(xv, yv, current_x, current_y, dx, dy)
B_induced = gradient(A_delta[:,:,1], dx, axis = 1) -  gradient(A_delta[:,:,0], dy, axis = 0)
B_app = gradient(A[:,:,1], dx, axis = 1) -  gradient(A[:,:,0], dy, axis = 0)

#Plot the induced magnetic field
plt.pcolormesh(x,y,B_induced,cmap='RdYlBu_r', vmin = -max(abs(B_induced)), vmax = max(abs(B_induced)))
plt.colorbar()
plt.title(f'eV = {eV}')
plt.xlabel('x')
plt.ylabel('y')
plt.gca().set_aspect('equal')
plt.savefig(f'Induced magnetic fields for n = {n} L={Lx} N={Nx} eV={eV}.pdf')
plt.gca().set_aspect('equal')
plt.savefig(f'Induced magnetic fields for n = {n} L={Lx} N={Nx} eV={eV}.png')
plt.close()

#Plot the circulation of the current
plt.pcolormesh(x,y,circulating_current, cmap = 'seismic', vmin = -max(abs(circulating_current)), vmax = max(abs(circulating_current)))
plt.colorbar()
plt.title(f'eV = {eV}')
plt.xlabel('x')
plt.ylabel('y')
plt.gca().set_aspect('equal')
plt.savefig(f'Circulating current for n = {n} L={Lx} N={Nx} eV={eV}.pdf')
plt.gca().set_aspect('equal')
plt.savefig(f'Circulating current for n = {n} L={Lx} N={Nx} eV={eV}.png')
plt.close()

#Plot the current density
plt.streamplot(xv, yv, current_x, current_y, color = abs_current, cmap ="viridis")
plt.colorbar()
plt.title(f'eV = {eV}')
plt.xlabel('x')
plt.ylabel('y')
plt.gca().set_aspect('equal')
plt.savefig(f'Streamplot of current for n = {n} L={Lx} N={Nx} eV={eV}.pdf')
plt.gca().set_aspect('equal')
plt.savefig(f'Streamplot of current for n = {n} L={Lx} N={Nx} eV={eV}.png')
plt.close()
