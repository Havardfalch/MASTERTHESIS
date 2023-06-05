# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 16:18:33 2023

@author: Havard
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 13:55:36 2023

@author: Havard
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate as spint
from linearized_Usadel_solver_2D import get_integrated_3d_Usadel_solution, update_A_field, A_field_integrated_usadel_parallel 
from Boundary_conditions_for_linearized_Usadel import f_integrated_z_direction

#Set parameters for the run
Nx = 400
Ny = Nx
e_ = -1
Lx = 20
Ly = Lx
Lz = 1
D = 1
G_T = 0.3
G_N = 1

threshold = 1
include_gbcs = False
kappa = 2.71
nu = 0.8

eV = 0
x,dx = np.linspace(-Lx/2,Lx/2,Nx,retstep = True)
y, dy = np.linspace(-Ly/2,Ly/2,Ny,retstep = True)
xv, yv = np.meshgrid(x,y)

sign = -1

"""
#Create the vortex pattern for 7 vortices in a larger lattice with 105 vortices
Lx2 = 70
Ly2 = Lx2

vortex_core_dist = (Lx/2)*3/4
long_axis_dist = sin(pi/3) * vortex_core_dist
short_axis_dist = sin(pi/6) * vortex_core_dist
#vortex_core_dist = 1.075 * np.sqrt(np.abs(np.pi/B_applied))
core_locs = []

for i in range(int((Lx2//2)//long_axis_dist)+1):
    for j in range(int((Ly2//2)//vortex_core_dist)+1):
        if i == 0 and j == 0:
            core_locs.append([0,0])
        elif (j*vortex_core_dist + (i%2)*short_axis_dist + threshold)>Lx2/2:
            pass
        elif j==0 and i%2==1:
            core_locs.append([j*vortex_core_dist + (i%2)*short_axis_dist, i * long_axis_dist])
            core_locs.append([-(j*vortex_core_dist + (i%2)*short_axis_dist), i * long_axis_dist])
            core_locs.append([j*vortex_core_dist + (i%2)*short_axis_dist, -i * long_axis_dist])
            core_locs.append([-(j*vortex_core_dist + (i%2)*short_axis_dist), -i * long_axis_dist])
        elif j==0:
            core_locs.append([(j*vortex_core_dist + (i%2)*short_axis_dist), i * long_axis_dist])
            core_locs.append([j*vortex_core_dist + (i%2)*short_axis_dist, -i * long_axis_dist])
        elif i==0:
            core_locs.append([j*vortex_core_dist + (i%2)*short_axis_dist, i * long_axis_dist])
            core_locs.append([-(j*vortex_core_dist + (i%2)*short_axis_dist), i * long_axis_dist])
        
        else:
            core_locs.append([j*vortex_core_dist + (i%2)*short_axis_dist, i * long_axis_dist])
            core_locs.append([-(j*vortex_core_dist + (i%2)*short_axis_dist), i * long_axis_dist])
            core_locs.append([j*vortex_core_dist + (i%2)*short_axis_dist, -i * long_axis_dist])
            core_locs.append([-(j*vortex_core_dist + (i%2)*short_axis_dist), -i * long_axis_dist])
print(len(core_locs))
core_locs = array(core_locs)

#Create vortex pattern for 23 vortices in the normal metal
core_locs = np.zeros((23,2))
for i in range(6):
    core_locs[i,0] = np.cos(i*2*np.pi/6)*np.max(x)*3/8
    core_locs[i,1] = np.sin(i*2*np.pi/6)*np.max(y)*3/8
for i in range(6):
    for j in range(2):
        core_locs[2*i+j+6,0] = np.cos(i*2*np.pi/6)*np.max(x)*3/8 + np.cos((i+j)*2*np.pi/6)*np.max(x)*3/8
        core_locs[2*i+j+6,1] = np.sin(i*2*np.pi/6)*np.max(y)*3/8 + np.sin((i+j)*2*np.pi/6)*np.max(y)*3/8
for i in range(4):
    core_locs[i+18,0] =  np.max(x)*6/8 * np.sign((i%2-1/2))
    core_locs[i+18,1] =  np.max(x)*6/8 * np.sin(1*np.pi/3) * np.sign(((i/3)%2-1/2))

#Create the vortex pattern for 7 vortices in the normal metal
core_locs = np.zeros((7,2))
for i in range(6):
    core_locs[i,0] = np.cos(i*2*np.pi/6)*np.max(x)*3/4
    core_locs[i,1] = np.sin(i*2*np.pi/6)*np.max(y)*3/4

"""

#Create a single isolated vortex
core_locs = np.zeros((1,2))

#Find the number of vortices
num_cores = core_locs.shape[0]

#Calculate the phase of the superconductor by adding the phase from each superconducting vortex
theta = 0
for i in range(core_locs.shape[0]):
    theta += sign * np.arctan2(yv-core_locs[i,1], xv-core_locs[i,0])
#Fix the phase in the range[0,2*pi)
theta = theta%(2*np.pi)

#Calculate the magnetic vector potential
A = np.zeros((Ny, Nx,2))
A = A_field_integrated_usadel_parallel(core_locs, A, x, y, Nx, Ny, dx, dy, threshold, kappa)

slc = np.max((1, Nx//20))

#For kappa<1 this fixes the vector potential to be the correct direction since then the log changes sign
multiplier = 1 * np.sign(np.log(kappa))
A*=multiplier


    
#Create an array with all the energies to solve for, we also need the negative energies
epsilons1 = np.linspace(eV,2,300,dtype = complex)
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
f_sols, f_grads = get_integrated_3d_Usadel_solution(epsilons, f_sols, f_grads, f_integrated_z_direction, x, y, dx, dy, A, theta = theta, D=D, e_ = e_, Lz = Lz, G_T = G_T, G_N = G_N, core_locs = core_locs, threshold = threshold, nu=nu, include_gbcs = include_gbcs)

#Solve the linearized Usadel equation for negative energies
f_sols_minus,f_grads_minus = get_integrated_3d_Usadel_solution(epsilons_minus, f_sols_minus, f_grads_minus, f_integrated_z_direction, x, y, dx, dy, A, theta = theta, D=D, e_ = e_, Lz = Lz, G_T = G_T, G_N = G_N, core_locs = core_locs, threshold = threshold, nu=nu, include_gbcs = include_gbcs)

#Calculate the form of the pair correlation
pair_corr = spint.trapz(f_sols-f_sols_minus, x = epsilons, axis = 0)
abs_corr = np.abs(pair_corr)

#Calculate the supercurrent in the x and y directions
current_x = e_*spint.trapz((f_sols*np.conjugate(f_grads_minus[:,:,:,0])- f_sols_minus*np.conjugate(f_grads[:,:,:,0]) + 2*e_*(A)[:,:,0]*1j*(f_sols*np.conjugate(f_sols_minus)-f_sols_minus*np.conjugate(f_sols))).real, x = epsilons, axis = 0).real
current_y = e_*spint.trapz((f_sols*np.conjugate(f_grads_minus[:,:,:,1])- f_sols_minus*np.conjugate(f_grads[:,:,:,1]) + 2*e_*(A)[:,:,1]*1j*(f_sols*np.conjugate(f_sols_minus)-f_sols_minus*np.conjugate(f_sols))).real, x = epsilons, axis = 0).real

abs_current = np.sqrt((current_x.real)**2+(current_y.real)**2)

#Find the difference in the phase in the currents and the superconducting vortex to more easily see if the current is reversed
circulating_current = abs_current*np.cos((theta - np.arctan2(-current_x, current_y)))

#Calculate the A and B fields induced by the supercurrents
A_delta = update_A_field(xv, yv, current_x, current_y, dx, dy)
B_induced = np.gradient(A_delta[:,:,1], dx, axis = 1) -  np.gradient(A_delta[:,:,0], dy, axis = 0)

#Calculate the applied B field
B_app = np.gradient(A[:,:,1], dx, axis = 1) -  np.gradient(A[:,:,0], dy, axis = 0)

#Save data
np.save(f"Applied B field for {num_cores} cores L={Lx} N={Nx} eV={eV} kappa={kappa}", B_app)
np.save(f"x current for {num_cores} cores L={Lx} N={Nx} eV={eV} kappa={kappa}", current_x)
np.save(f"y current for {num_cores} cores L={Lx} N={Nx} eV={eV} kappa={kappa}", current_y)

#Set the font size for plotting
plt.rcParams.update({'font.size': 20})


#Plot the applied magnetic field
plt.pcolormesh(x,y,B_app,cmap='RdYlBu_r', vmin = -np.max(abs(B_app)), vmax = np.max(abs(B_app)))
plt.colorbar()

plt.title('Applied magnetic field')
plt.xlabel('x')
plt.ylabel('y')
plt.gca().set_aspect('equal')
plt.savefig(f'Applied magnetic fields for {num_cores} cores L={Lx} N={Nx} eV={eV} kappa={kappa} nu={nu} G_T={G_T}.pdf')
plt.gca().set_aspect('equal')
plt.savefig(f'Applied magnetic fields for {num_cores} cores L={Lx} N={Nx} eV={eV} kappa={kappa} nu={nu} G_T={G_T}.png')

plt.close()

#Plot the induced magnetic field
plt.pcolormesh(x,y,B_induced,cmap='RdYlBu_r', vmin = -np.max(abs(B_induced)), vmax = np.max(abs(B_induced)))
plt.colorbar()
plt.title(f'eV = {eV}')
plt.xlabel('x')
plt.ylabel('y')
plt.gca().set_aspect('equal')
plt.savefig(f'Induced magnetic fields for {num_cores} cores L={Lx} N={Nx} eV={eV} kappa={kappa} nu={nu} G_T={G_T}.pdf')
plt.gca().set_aspect('equal')
plt.savefig(f'Induced magnetic fields for {num_cores} cores L={Lx} N={Nx} eV={eV} kappa={kappa} nu={nu} G_T={G_T}.png')

plt.close()

#Plot the circulation of the current
plt.pcolormesh(x,y,circulating_current, cmap = 'seismic', vmin = -np.max(abs(circulating_current)), vmax = np.max(abs(circulating_current)))
plt.colorbar()
plt.title(f'eV = {eV}')
plt.xlabel('x')
plt.ylabel('y')
plt.gca().set_aspect('equal')
plt.savefig(f'Circulating current for {num_cores} cores L={Lx} N={Nx} eV={eV} kappa={kappa} nu={nu} G_T={G_T}.pdf')
plt.gca().set_aspect('equal')
plt.savefig(f'Circulating current for {num_cores} cores L={Lx} N={Nx} eV={eV} kappa={kappa} nu={nu} G_T={G_T}.png')

plt.close()

#Plot the current density
plt.streamplot(xv, yv, current_x, current_y, color = abs_current, cmap = "viridis")
plt.xlim(min(x), np.max(x))
plt.colorbar()
plt.title(f'eV = {eV}')
plt.xlabel('x')
plt.ylabel('y')
plt.gca().set_aspect('equal')
plt.savefig(f'Streamplot of current for {num_cores} cores L={Lx} N={Nx} eV={eV} kappa={kappa} nu={nu} G_T={G_T}.pdf')
plt.gca().set_aspect('equal')
plt.savefig(f'Streamplot of current for {num_cores} cores L={Lx} N={Nx} eV={eV} kappa={kappa} nu={nu} G_T={G_T}.png')

plt.close()

eV = 0.15

#Find at what index the applied voltage is situated and only include energies larger than the applied voltage
cutoff_ind = np.max(np.where(epsilons.real<eV))+1
f_sols_new = f_sols[cutoff_ind:]
f_grads_new = f_grads[cutoff_ind:]
f_sols_minus_new = f_sols_minus[cutoff_ind:]
f_grads_minus_new = f_grads_minus[cutoff_ind:]
epsilons_new = epsilons[cutoff_ind:]


#Calculate the supercurrent in the x and y directions
current_x = e_*spint.trapz((f_sols_new*np.conjugate(f_grads_minus_new[:,:,:,0])- f_sols_minus_new*np.conjugate(f_grads_new[:,:,:,0]) + 2*e_*(A)[:,:,0]*1j*(f_sols_new*np.conjugate(f_sols_minus_new)-f_sols_minus_new*np.conjugate(f_sols_new))).real, x = epsilons_new, axis = 0).real
current_y = e_*spint.trapz((f_sols_new*np.conjugate(f_grads_minus_new[:,:,:,1])- f_sols_minus_new*np.conjugate(f_grads_new[:,:,:,1]) + 2*e_*(A)[:,:,1]*1j*(f_sols_new*np.conjugate(f_sols_minus_new)-f_sols_minus_new*np.conjugate(f_sols_new))).real, x = epsilons_new, axis = 0).real

abs_current = np.sqrt((current_x.real)**2+(current_y.real)**2)
circulating_current = abs_current*np.cos((theta - np.arctan2(current_x, current_y))%(2*np.pi))

#Save data
np.save(f"x current for {num_cores} cores L={Lx} N={Nx} eV={eV} kappa={kappa}", current_x)
np.save(f"y current for {num_cores} cores L={Lx} N={Nx} eV={eV} kappa={kappa}", current_y)

#Calculate the A and B fields induced by the supercurrents
A_delta = update_A_field(xv, yv, current_x, current_y, dx, dy)
B_induced = np.gradient(A_delta[:,:,1], dx, axis = 1) -  np.gradient(A_delta[:,:,0], dy, axis = 0)

#Plot the induced magnetic field
plt.pcolormesh(x,y,B_induced,cmap='RdYlBu_r', vmin = -np.max(abs(B_induced)), vmax = np.max(abs(B_induced)))
plt.colorbar()
plt.title(f'eV = {eV}')
plt.xlabel('x')
plt.ylabel('y')
plt.gca().set_aspect('equal')
plt.savefig(f'Induced magnetic fields for {num_cores} cores L={Lx} N={Nx} eV={eV} kappa={kappa} nu={nu} G_T={G_T}.pdf')
plt.gca().set_aspect('equal')
plt.savefig(f'Induced magnetic fields for {num_cores} cores L={Lx} N={Nx} eV={eV} kappa={kappa} nu={nu} G_T={G_T}.png')
plt.close()

#Plot the circulation of the current
plt.pcolormesh(x,y,circulating_current, cmap = 'seismic', vmin = -np.max(abs(circulating_current)), vmax = np.max(abs(circulating_current)))
plt.colorbar()
plt.title(f'eV = {eV}')
plt.xlabel('x')
plt.ylabel('y')
plt.gca().set_aspect('equal')
plt.savefig(f'Circulating current for {num_cores} cores L={Lx} N={Nx} eV={eV} kappa={kappa} nu={nu} G_T={G_T}.pdf')
plt.gca().set_aspect('equal')
plt.savefig(f'Circulating current for {num_cores} cores L={Lx} N={Nx} eV={eV} kappa={kappa} nu={nu} G_T={G_T}.png')
plt.close()

#Plot the current density
plt.streamplot(xv, yv, current_x, current_y, color = abs_current, cmap = "viridis")
plt.colorbar()
plt.title(f'eV = {eV}')
plt.xlabel('x')
plt.ylabel('y')
plt.gca().set_aspect('equal')
plt.savefig(f'Streamplot of current for {num_cores} cores L={Lx} N={Nx} eV={eV} kappa={kappa} nu={nu} G_T={G_T}.pdf')
plt.gca().set_aspect('equal')
plt.savefig(f'Streamplot of current for {num_cores} cores L={Lx} N={Nx} eV={eV} kappa={kappa} nu={nu} G_T={G_T}.png')
plt.close()

eV = 0.25

#Find at what index the applied voltage is situated and only include energies larger than the applied voltage
cutoff_ind = np.max(np.where(epsilons.real<eV))+1
f_sols_new = f_sols[cutoff_ind:]
f_grads_new = f_grads[cutoff_ind:]
f_sols_minus_new = f_sols_minus[cutoff_ind:]
f_grads_minus_new = f_grads_minus[cutoff_ind:]
epsilons_new = epsilons[cutoff_ind:]


#Calculate the supercurrent in the x and y directions
current_x = e_*spint.trapz((f_sols_new*np.conjugate(f_grads_minus_new[:,:,:,0])- f_sols_minus_new*np.conjugate(f_grads_new[:,:,:,0]) + 2*e_*(A)[:,:,0]*1j*(f_sols_new*np.conjugate(f_sols_minus_new)-f_sols_minus_new*np.conjugate(f_sols_new))).real, x = epsilons_new, axis = 0).real
current_y = e_*spint.trapz((f_sols_new*np.conjugate(f_grads_minus_new[:,:,:,1])- f_sols_minus_new*np.conjugate(f_grads_new[:,:,:,1]) + 2*e_*(A)[:,:,1]*1j*(f_sols_new*np.conjugate(f_sols_minus_new)-f_sols_minus_new*np.conjugate(f_sols_new))).real, x = epsilons_new, axis = 0).real

abs_current = np.sqrt((current_x.real)**2+(current_y.real)**2)
circulating_current = abs_current*np.cos((theta - np.arctan2(current_x, current_y))%(2*np.pi))

#Save data
np.save(f"x current for {num_cores} cores L={Lx} N={Nx} eV={eV} kappa={kappa}", current_x)
np.save(f"y current for {num_cores} cores L={Lx} N={Nx} eV={eV} kappa={kappa}", current_y)

#Calculate the A and B fields induced by the supercurrents
A_delta = update_A_field(xv, yv, current_x, current_y, dx, dy)
B_induced = np.gradient(A_delta[:,:,1], dx, axis = 1) -  np.gradient(A_delta[:,:,0], dy, axis = 0)

#Plot the induced magnetic field
plt.pcolormesh(x,y,B_induced,cmap='RdYlBu_r', vmin = -np.max(abs(B_induced)), vmax = np.max(abs(B_induced)))
plt.colorbar()
plt.title(f'eV = {eV}')
plt.xlabel('x')
plt.ylabel('y')
plt.gca().set_aspect('equal')
plt.savefig(f'Induced magnetic fields for {num_cores} cores L={Lx} N={Nx} eV={eV} kappa={kappa} nu={nu} G_T={G_T}.pdf')
plt.gca().set_aspect('equal')
plt.savefig(f'Induced magnetic fields for {num_cores} cores L={Lx} N={Nx} eV={eV} kappa={kappa} nu={nu} G_T={G_T}.png')
plt.close()

#Plot the circulation of the current
plt.pcolormesh(x,y,circulating_current, cmap = 'seismic', vmin = -np.max(abs(circulating_current)), vmax = np.max(abs(circulating_current)))
plt.colorbar()
plt.title(f'eV = {eV}')
plt.xlabel('x')
plt.ylabel('y')
plt.gca().set_aspect('equal')
plt.savefig(f'Circulating current for {num_cores} cores L={Lx} N={Nx} eV={eV} kappa={kappa} nu={nu} G_T={G_T}.pdf')
plt.gca().set_aspect('equal')
plt.savefig(f'Circulating current for {num_cores} cores L={Lx} N={Nx} eV={eV} kappa={kappa} nu={nu} G_T={G_T}.png')
plt.close()

#Plot the current density
plt.streamplot(xv, yv, current_x, current_y, color = abs_current, cmap = "viridis")
plt.colorbar()
plt.title(f'eV = {eV}')
plt.xlabel('x')
plt.ylabel('y')
plt.gca().set_aspect('equal')
plt.savefig(f'Streamplot of current for {num_cores} cores L={Lx} N={Nx} eV={eV} kappa={kappa} nu={nu} G_T={G_T}.pdf')
plt.gca().set_aspect('equal')
plt.savefig(f'Streamplot of current for {num_cores} cores L={Lx} N={Nx} eV={eV} kappa={kappa} nu={nu} G_T={G_T}.png')
plt.close()

eV = 0.35

#Find at what index the applied voltage is situated and only include energies larger than the applied voltage
cutoff_ind = np.max(np.where(epsilons.real<eV))+1
f_sols_new = f_sols[cutoff_ind:]
f_grads_new = f_grads[cutoff_ind:]
f_sols_minus_new = f_sols_minus[cutoff_ind:]
f_grads_minus_new = f_grads_minus[cutoff_ind:]
epsilons_new = epsilons[cutoff_ind:]


#Calculate the supercurrent in the x and y directions
current_x = e_*spint.trapz((f_sols_new*np.conjugate(f_grads_minus_new[:,:,:,0])- f_sols_minus_new*np.conjugate(f_grads_new[:,:,:,0]) + 2*e_*(A)[:,:,0]*1j*(f_sols_new*np.conjugate(f_sols_minus_new)-f_sols_minus_new*np.conjugate(f_sols_new))).real, x = epsilons_new, axis = 0).real
current_y = e_*spint.trapz((f_sols_new*np.conjugate(f_grads_minus_new[:,:,:,1])- f_sols_minus_new*np.conjugate(f_grads_new[:,:,:,1]) + 2*e_*(A)[:,:,1]*1j*(f_sols_new*np.conjugate(f_sols_minus_new)-f_sols_minus_new*np.conjugate(f_sols_new))).real, x = epsilons_new, axis = 0).real

abs_current = np.sqrt((current_x.real)**2+(current_y.real)**2)
circulating_current = abs_current*np.cos((theta - np.arctan2(current_x, current_y))%(2*np.pi))

#Save data
np.save(f"x current for {num_cores} cores L={Lx} N={Nx} eV={eV} kappa={kappa}", current_x)
np.save(f"y current for {num_cores} cores L={Lx} N={Nx} eV={eV} kappa={kappa}", current_y)

#Calculate the A and B fields induced by the supercurrents
A_delta = update_A_field(xv, yv, current_x, current_y, dx, dy)
B_induced = np.gradient(A_delta[:,:,1], dx, axis = 1) -  np.gradient(A_delta[:,:,0], dy, axis = 0)

#Plot the induced magnetic field
plt.pcolormesh(x,y,B_induced,cmap='RdYlBu_r', vmin = -np.max(abs(B_induced)), vmax = np.max(abs(B_induced)))
plt.colorbar()
plt.title(f'eV = {eV}')
plt.xlabel('x')
plt.ylabel('y')
plt.gca().set_aspect('equal')
plt.savefig(f'Induced magnetic fields for {num_cores} cores L={Lx} N={Nx} eV={eV} kappa={kappa} nu={nu} G_T={G_T}.pdf')
plt.gca().set_aspect('equal')
plt.savefig(f'Induced magnetic fields for {num_cores} cores L={Lx} N={Nx} eV={eV} kappa={kappa} nu={nu} G_T={G_T}.png')
plt.close()

#Plot the circulation of the current
plt.pcolormesh(x,y,circulating_current, cmap = 'seismic', vmin = -np.max(abs(circulating_current)), vmax = np.max(abs(circulating_current)))
plt.colorbar()
plt.title(f'eV = {eV}')
plt.xlabel('x')
plt.ylabel('y')
plt.gca().set_aspect('equal')
plt.savefig(f'Circulating current for {num_cores} cores L={Lx} N={Nx} eV={eV} kappa={kappa} nu={nu} G_T={G_T}.pdf')
plt.gca().set_aspect('equal')
plt.savefig(f'Circulating current for {num_cores} cores L={Lx} N={Nx} eV={eV} kappa={kappa} nu={nu} G_T={G_T}.png')
plt.close()

#Plot the current density
plt.streamplot(xv, yv, current_x, current_y, color = abs_current, cmap = "viridis")
plt.colorbar()
plt.title(f'eV = {eV}')
plt.xlabel('x')
plt.ylabel('y')
plt.gca().set_aspect('equal')
plt.savefig(f'Streamplot of current for {num_cores} cores L={Lx} N={Nx} eV={eV} kappa={kappa} nu={nu} G_T={G_T}.pdf')
plt.gca().set_aspect('equal')
plt.savefig(f'Streamplot of current for {num_cores} cores L={Lx} N={Nx} eV={eV} kappa={kappa} nu={nu} G_T={G_T}.png')
plt.close()


eV = 0.5

#Find at what index the applied voltage is situated and only include energies larger than the applied voltage
cutoff_ind = np.max(np.where(epsilons.real<eV))+1
f_sols_new = f_sols[cutoff_ind:]
f_grads_new = f_grads[cutoff_ind:]
f_sols_minus_new = f_sols_minus[cutoff_ind:]
f_grads_minus_new = f_grads_minus[cutoff_ind:]
epsilons_new = epsilons[cutoff_ind:]


#Calculate the supercurrent in the x and y directions
current_x = e_*spint.trapz((f_sols_new*np.conjugate(f_grads_minus_new[:,:,:,0])- f_sols_minus_new*np.conjugate(f_grads_new[:,:,:,0]) + 2*e_*(A)[:,:,0]*1j*(f_sols_new*np.conjugate(f_sols_minus_new)-f_sols_minus_new*np.conjugate(f_sols_new))).real, x = epsilons_new, axis = 0).real
current_y = e_*spint.trapz((f_sols_new*np.conjugate(f_grads_minus_new[:,:,:,1])- f_sols_minus_new*np.conjugate(f_grads_new[:,:,:,1]) + 2*e_*(A)[:,:,1]*1j*(f_sols_new*np.conjugate(f_sols_minus_new)-f_sols_minus_new*np.conjugate(f_sols_new))).real, x = epsilons_new, axis = 0).real

abs_current = np.sqrt((current_x.real)**2+(current_y.real)**2)
circulating_current = abs_current*np.cos((theta - np.arctan2(current_x, current_y))%(2*np.pi))

#Save data
np.save(f"x current for {num_cores} cores L={Lx} N={Nx} eV={eV} kappa={kappa}", current_x)
np.save(f"y current for {num_cores} cores L={Lx} N={Nx} eV={eV} kappa={kappa}", current_y)

#Calculate the A and B fields induced by the supercurrents
A_delta = update_A_field(xv, yv, current_x, current_y, dx, dy)
B_induced = np.gradient(A_delta[:,:,1], dx, axis = 1) -  np.gradient(A_delta[:,:,0], dy, axis = 0)

#Plot the induced magnetic field
plt.pcolormesh(x,y,B_induced,cmap='RdYlBu_r', vmin = -np.max(abs(B_induced)), vmax = np.max(abs(B_induced)))
plt.colorbar()
plt.title(f'eV = {eV}')
plt.xlabel('x')
plt.ylabel('y')
plt.gca().set_aspect('equal')
plt.savefig(f'Induced magnetic fields for {num_cores} cores L={Lx} N={Nx} eV={eV} kappa={kappa} nu={nu} G_T={G_T}.pdf')
plt.gca().set_aspect('equal')
plt.savefig(f'Induced magnetic fields for {num_cores} cores L={Lx} N={Nx} eV={eV} kappa={kappa} nu={nu} G_T={G_T}.png')
plt.close()

#Plot the circulation of the current
plt.pcolormesh(x,y,circulating_current, cmap = 'seismic', vmin = -np.max(abs(circulating_current)), vmax = np.max(abs(circulating_current)))
plt.colorbar()
plt.title(f'eV = {eV}')
plt.xlabel('x')
plt.ylabel('y')
plt.gca().set_aspect('equal')
plt.savefig(f'Circulating current for {num_cores} cores L={Lx} N={Nx} eV={eV} kappa={kappa} nu={nu} G_T={G_T}.pdf')
plt.gca().set_aspect('equal')
plt.savefig(f'Circulating current for {num_cores} cores L={Lx} N={Nx} eV={eV} kappa={kappa} nu={nu} G_T={G_T}.png')
plt.close()

#Plot the current density
plt.streamplot(xv, yv, current_x, current_y, color = abs_current, cmap = "viridis")
plt.colorbar()
plt.title(f'eV = {eV}')
plt.xlabel('x')
plt.ylabel('y')
plt.gca().set_aspect('equal')
plt.savefig(f'Streamplot of current for {num_cores} cores L={Lx} N={Nx} eV={eV} kappa={kappa} nu={nu} G_T={G_T}.pdf')
plt.gca().set_aspect('equal')
plt.savefig(f'Streamplot of current for {num_cores} cores L={Lx} N={Nx} eV={eV} kappa={kappa} nu={nu} G_T={G_T}.png')
