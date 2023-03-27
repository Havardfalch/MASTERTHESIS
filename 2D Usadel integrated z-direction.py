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
from scipy import special as spspecial
from linearized_Usadel_solver_2D import get_integrated_3d_Usadel_solution
from Boundary_conditions_for_linearized_Usadel import f_integrated_z_direction
import time
from tqdm import tqdm



import cProfile as profile
import pstats
t1 = time.time()
 


def int_func_for_A_x(t,x,y,lambd, threshold = 1, kappa=1, phi_0 = -np.pi):
    if np.sqrt((t*x)**2+(t*y)**2)<=threshold:
        return t*y*np.log(kappa)*phi_0/(2*np.pi*lambd**2)
    else:
        return t*y*phi_0/(2*np.pi*lambd**2) * spspecial.kn(0, np.sqrt((t*x)**2+(t*y)**2)/lambd) * np.log(kappa)/spspecial.kn(0,threshold/lambd)
def int_func_for_A_y(t,x,y,lambd, threshold = 1, kappa=1, phi_0 = -np.pi):
    if np.sqrt((t*x)**2+(t*y)**2)<=threshold:
        return -t*x*np.log(kappa)*phi_0/(2*np.pi*lambd**2)
    else:
        return -t*x*phi_0/(2*np.pi*lambd**2) * spspecial.kn(0, np.sqrt((t*x)**2+(t*y)**2)/lambd) * np.log(kappa)/spspecial.kn(0,threshold/lambd)

def B_strength(x,y,lambd, threshold = 1, kappa=1, phi_0 = -np.pi):
    if np.sqrt(x**2+y**2)<=threshold:
        return -np.log(kappa)*phi_0/(2*np.pi*lambd**2)
    else:
        return -phi_0/(2*np.pi*lambd**2) * spspecial.kn(0, np.sqrt((x)**2+(y)**2)/lambd) * np.log(kappa)/spspecial.kn(0,threshold/lambd)


def A_field_integrated_usadel(core_locs, A, x, y, Nx, Ny, dx, dy, threshold = 1, kappa = 2.71, phi_0 = -np.pi, ksi = 1):
    xv,yv = np.meshgrid(x,y)
    lambd = kappa*ksi
    for i in tqdm(range(xv.shape[0])):
        for j in range(xv.shape[1]):
            for k in range(core_locs.shape[0]):
                x_rel = xv[i,j]-core_locs[k,0]
                y_rel = yv[i,j]-core_locs[k,1]
                
                A[i,j,0] += spint.quad(int_func_for_A_x,0,1, args = (x_rel, y_rel, lambd, threshold, kappa, phi_0))[0]
                A[i,j,1] += spint.quad(int_func_for_A_y,0,1, args = (x_rel, y_rel, lambd, threshold, kappa, phi_0))[0]
                
    return A

#prof = profile.Profile()
#prof.enable()

Nx = 301
Ny = Nx
e_ = -1
Lx = 20
Ly = Lx
Lz = 1
D = 1
G_T = 0.01
#Endre denne 
G_N = 1

threshold = 1
include_gbcs = False

kappa = 12

nu = 0.25
print(nu, 1/(2*np.sqrt(2)))
plt.plot(np.linspace(0, Lx/2,200), np.tanh(nu*(np.linspace(0, Lx/2,200))))
plt.grid()
plt.title("tanh(nu*x)")
plt.show()


eV = 1

x,dx = np.linspace(-Lx/2,Lx/2,Nx,retstep = True)
y, dy = np.linspace(-Ly/2,Ly/2,Ny,retstep = True)
xv, yv = np.meshgrid(x,y)


vortex_core_dist = 1.1 * np.sqrt(np.abs(1/(np.log(kappa)*np.pi/(2*np.pi*kappa**2))))
long_axis_dist = np.sin(np.pi/3) * vortex_core_dist
short_axis_dist = np.sin(np.pi/6) * vortex_core_dist
B_applied = 100
#vortex_core_dist = 1.075 * np.sqrt(np.abs(np.pi/B_applied))






sign = 1
#theta = sign * np.arctan2(yv,xv)

long_axis_locs = np.array(np.where(((x[:-1]%long_axis_dist) - (x[1:]%long_axis_dist))>0))
short_axis_locs = np.array(np.where(((y[:-1]%short_axis_dist) - (y[1:]%short_axis_dist))>0))
#print(long_axis_locs, long_axis_locs[0,:-1]-long_axis_locs[0,1:], round(np.average(long_axis_locs[0,:-1]-long_axis_locs[0,1:])))
#print(short_axis_locs, short_axis_locs[0,:-1]- short_axis_locs[0,1:], round(np.average(short_axis_locs[0,:-1]- short_axis_locs[0,1:])))
#print(short_axis_dist, long_axis_dist, vortex_core_dist)
core_locs = []
#i = 0 
#while (i*vortex_core_dist+threshold)<np.max(x):
#    core_locs.append(np.array([long_axis_locs*((i+1)%2) + (i+1)//2  ]))

for i in range(int((Lx//2)//long_axis_dist)+1):
    for j in range(int((Ly//2)//vortex_core_dist)+1):
        if i == 0 and j == 0:
            core_locs.append([0,0])
        elif (j*vortex_core_dist + (i%2)*short_axis_dist + threshold)>Lx/2:
            pass
        else:
            core_locs.append([j*vortex_core_dist + (i%2)*short_axis_dist, i * long_axis_dist])
            core_locs.append([-(j*vortex_core_dist + (i%2)*short_axis_dist), i * long_axis_dist])
            core_locs.append([j*vortex_core_dist + (i%2)*short_axis_dist, -i * long_axis_dist])
            core_locs.append([-(j*vortex_core_dist + (i%2)*short_axis_dist), -i * long_axis_dist])
print(core_locs)
core_locs = np.array(core_locs)



core_locs = np.zeros((7,2))
for i in range(6):
    core_locs[i,0] = np.cos(i*2*np.pi/6)*np.max(x)*2/3
    core_locs[i,1] = np.sin(i*2*np.pi/6)*np.max(y)*2/3
"""
core_locs[0,0] = x[Nx//4]
core_locs[0,1] = y[Ny//4]

core_locs[1,0] = x[3*Nx//4]
core_locs[1,1] = y[Ny//4]

core_locs[2,0] = x[Nx//4]
core_locs[2,1] = y[3*Ny//4]

core_locs[3,0] = x[3*Nx//4]
core_locs[3,1] = y[3*Ny//4]
"""
plt.scatter(core_locs[:,0], core_locs[:,1])
plt.xlim(np.min(x), np.max(x))
plt.ylim(np.min(y), np.max(y))
plt.show()

core_locs = np.zeros((1,2))
theta = 0
for i in range(core_locs.shape[0]):
    theta += sign * np.arctan2(yv-core_locs[i,1], xv-core_locs[i,0])
theta = theta%(2*np.pi)-np.pi

A = np.zeros((Ny, Nx,2))
A = A_field_integrated_usadel(core_locs, A, x, y, Nx, Ny, dx, dy, threshold, kappa)
slc = np.max((1, Nx//20))
multiplier = 1 * np.sign(np.log(kappa))
A*=multiplier


B = ((A[1:-1,2:,1]-A[1:-1,:-2,1])/(2*dx) +(A[:-2, 1:-1, 0]- A[2:,1:-1,0])/(2*dy))

print("Total B field", np.sum(B)*dx*dy)
#A *=np.pi*core_locs.shape[0]/(np.sum(B)*dx*dy)*np.abs(multiplier)
#B = ((A[1:-1,2:,1]-A[1:-1,:-2,1])/(2*dx) +(A[:-2, 1:-1, 0]- A[2:,1:-1,0])/(2*dy))
print("Total normalized B field", np.sum(B)*dx*dy)
B_strength_vec = np.vectorize(B_strength)
B_exact = B_strength_vec(x, y[Ny//2], kappa, threshold, kappa)*np.sign(np.log(kappa))
B_exact *= np.max(B)/np.max(B_exact)
plt.plot(x, B_exact, label =  "Exact form")
plt.plot(x[1:-1], B[:,Ny//2], label = "Finite difference form")
plt.title("B field at y = " + str(y[Ny//2]))
plt.legend()
plt.grid()
plt.show()

div = ((A[1:-1,2:,0]-A[1:-1,:-2,0])/(2*dx) - (A[:-2, 1:-1, 1]- A[2:,1:-1,1])/(2*dy))
print("Max B", np.max(B), "Min B", np.min(B))
print("Max div", np.max(np.abs(div)))
print("Applied voltage", eV)
print("Size in x:", Lx, "Points in x:", Nx, "Size in y:", Ly, "Points in y:", Ny)
print("Max of A", np.max(np.linalg.norm(A, axis = 2)), "Multiplier", multiplier)
print("SC phase sign", sign)
print("Vortex core size", threshold, "Vortex core distance", vortex_core_dist)
print("Kappa", kappa, "Nu", nu, "G_T", G_T)
if include_gbcs:
    print("Using g_bcs term")
else:
    print("Not using g_bcs term")
epsilons1 = np.linspace(eV,2,300,dtype = complex)
epsilons2 = np.linspace(2,30, 280, dtype = complex)
epsilons = np.append(epsilons1,epsilons2)


epsilons_minus = -epsilons

epsilons+= 1j*(1e-2)
epsilons_minus+= 1j*(1e-2)



#prof = profile.Profile()
#prof.enable()

f_sols = np.zeros((epsilons.shape[0],Ny,Nx),dtype = complex)
f_sols_minus = np.zeros((epsilons.shape[0],Ny,Nx),dtype = complex)
f_grads = np.zeros((epsilons.shape[0],Ny,Nx,2),dtype = complex)
f_grads_minus = np.zeros((epsilons.shape[0],Ny,Nx,2),dtype = complex)


f_sols, f_grads = get_integrated_3d_Usadel_solution(epsilons, f_sols, f_grads, f_integrated_z_direction, x, y, dx, dy, A, theta = theta, D=D, e_ = e_, Lz = Lz, G_T = G_T, G_N = G_N, core_locs = core_locs, threshold = threshold, nu=nu, include_gbcs = include_gbcs)

f_sols_minus,f_grads_minus = get_integrated_3d_Usadel_solution(epsilons_minus, f_sols_minus, f_grads_minus, f_integrated_z_direction, x, y, dx, dy, A, theta = theta, D=D, e_ = e_, Lz = Lz, G_T = G_T, G_N = G_N, core_locs = core_locs, threshold = threshold, nu=nu, include_gbcs = include_gbcs)

pair_corr = spint.trapz(f_sols-f_sols_minus, x = epsilons, axis = 0)
abs_corr = np.abs(pair_corr)
current_x = e_*spint.trapz((f_sols*np.conjugate(f_grads_minus[:,:,:,0])- f_sols_minus*np.conjugate(f_grads[:,:,:,0]) + 2*e_*(A)[:,:,0]*1j*(f_sols*np.conjugate(f_sols_minus)-f_sols_minus*np.conjugate(f_sols))).real, x = epsilons, axis = 0).real
current_y = e_*spint.trapz((f_sols*np.conjugate(f_grads_minus[:,:,:,1])- f_sols_minus*np.conjugate(f_grads[:,:,:,1]) + 2*e_*(A)[:,:,1]*1j*(f_sols*np.conjugate(f_sols_minus)-f_sols_minus*np.conjugate(f_sols))).real, x = epsilons, axis = 0).real

abs_current = np.sqrt((current_x.real)**2+(current_y.real)**2)
circulating_current = abs_current*np.cos((np.arctan2(yv,xv) - np.arctan2(-current_x, current_y))%(2*np.pi))


print("Max inverted current", np.min(circulating_current))

#prof.disable()

#☺
div_current_x = current_x[:,:-2]- current_x[:,2:]
div_current_y = current_y[:-2]- current_y[2:]

"""
plt.pcolormesh(x,y,current_x.real,cmap='seismic', vmin = -np.max(np.abs(current_x.real)), vmax = np.max(np.abs(current_x.real)))
plt.colorbar()
plt.title("Current in x-direction")
plt.show()

plt.pcolormesh(x,y,current_y.real,cmap='seismic', vmin = -np.max(np.abs(current_y.real)), vmax = np.max(np.abs(current_y.real)))
plt.colorbar()
plt.title("Current in y-direction")
plt.show()
"""

plt.quiver(x[slc//2::slc],y[slc//2::slc],current_x[slc//2::slc,slc//2::slc].real, current_y[slc//2::slc,slc//2::slc].real)
plt.scatter(0,0)
plt.title("Supercurrent")
plt.show()

plt.quiver(x[slc//2::slc],y[slc//2::slc],(current_x[slc//2::slc,slc//2::slc].real*0+1), (current_y[slc//2::slc,slc//2::slc].real*0+1), angles = (np.arctan2(current_y[slc//2::slc,slc//2::slc].real, current_x[slc//2::slc,slc//2::slc].real)*180.0/np.pi))
plt.scatter(0,0)
plt.title("Direction of supercurrent")
plt.show()

plt.quiver(x[slc//2::slc],y[slc//2::slc],(A) [slc//2::slc,slc//2::slc,0], (A)[slc//2::slc,slc//2::slc,1])
plt.scatter(0,0)
plt.title("Applied magnetic vector potential")
plt.show()

plt.pcolormesh(x[1:-1],y[1:-1],B,cmap='seismic', vmin = np.min(B), vmax = np.max(B))
plt.colorbar()
plt.title("Magnetic field")
plt.show()
"""
plt.pcolormesh(x[1:-1],y[1:-1],div,cmap='seismic', vmin = np.min(div), vmax = np.max(div))
plt.colorbar()
plt.title("Divergence")
plt.show()

plt.pcolormesh(x[1:-1],y[1:-1],np.abs(B),norm=colors.LogNorm(vmin = np.min(np.abs(B))+1e-6, vmax = np.max(np.abs(B))), cmap = 'seismic')
plt.colorbar()
plt.title("B logatritmically")
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
plt.pcolormesh(x[1:-1],y[1:-1], div_current_x.real[1:-1] + div_current_y.real[:,1:-1],cmap='seismic')
plt.colorbar()
plt.title("Divergence of current")
plt.show()

plt.pcolormesh(x,y,abs_current, cmap = 'seismic')
plt.colorbar()
plt.title("Absolute value of current")
plt.show()

plt.pcolormesh(x,y,circulating_current, cmap = 'seismic', vmin = -np.max(np.abs(circulating_current)), vmax = np.max(np.abs(circulating_current)))
plt.colorbar()
plt.title("Circulation of current")
plt.show()

plt.pcolormesh(x,y,circulating_current, cmap = 'tab20', vmin = -np.max(np.abs(circulating_current)), vmax = np.max(np.abs(circulating_current)))
plt.colorbar()
plt.title("Circulation of current")
plt.show()
"""
plt.pcolormesh(x,y,abs_current,norm=colors.LogNorm(vmin=np.sort(abs_current.flatten())[1], vmax=np.max(abs_current)), cmap = 'seismic')
plt.colorbar()
plt.title("Absolute value of current")
plt.show()
"""
plt.pcolormesh(x,y,abs_corr,cmap='seismic',vmin = np.sort(abs_corr.flatten())[1], vmax = np.max(np.abs(abs_corr)))
plt.title("Absolute value of pair correlation")
plt.colorbar()
plt.show()
"""
plt.pcolormesh(x,y,abs_corr,cmap='seismic', norm=colors.LogNorm(vmin = np.sort(abs_corr.flatten())[1], vmax = np.max(np.abs(abs_corr))))
plt.title("Absolute value of pair correlation")
plt.colorbar()
plt.show()
"""
phases = np.arctan2(pair_corr.imag,pair_corr.real)
plt.pcolormesh(x,y,phases,cmap='seismic',vmin = np.min(phases), vmax = np.max(phases))
plt.title("Phase of pair correlation")
plt.colorbar()
plt.show()

plt.pcolormesh(x,y,theta,cmap='seismic',vmin = np.min(theta), vmax = np.max(theta))
plt.title("Phase from SC")
plt.colorbar()
plt.show()


plt.pcolormesh(x,y,(np.arctan2(yv,xv) - np.arctan2(-current_x, current_y))%(2*np.pi),cmap='seismic')
plt.title("Phase difference in space and currents")
plt.colorbar()
plt.show()
# print profiling output
#stats = pstats.Stats(prof).strip_dirs().sort_stats("cumtime")
#stats.print_stats(10)

xv, yv = np.meshgrid(x,y)
plt.streamplot(xv, yv, current_x, current_y, color = abs_current, cmap = "viridis")
plt.colorbar()
plt.title("Streamplot of current")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

plt.streamplot(xv, yv, current_x, current_y, color = abs_current, cmap = "seismic")
plt.colorbar()
plt.title("Streamplot of current")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

plt.quiver(x[Nx//2-10:Nx//2+10],y[Ny//2-10:Ny//2+10],current_x[Ny//2-10:Ny//2+10,Nx//2-10:Nx//2+10].real, current_y[Ny//2-10:Ny//2+10,Nx//2-10:Nx//2+10].real)
plt.scatter(0,0)
plt.title("Supercurrent")
plt.show()
plt.streamplot(xv[Ny//2-20:Ny//2+20,Nx//2-20:Nx//2+20], yv[Ny//2-20:Ny//2+20,Nx//2-20:Nx//2+20], current_x[Ny//2-20:Ny//2+20,Nx//2-20:Nx//2+20].real, current_y[Ny//2-20:Ny//2+20,Nx//2-20:Nx//2+20].real, color = abs_current[Ny//2-20:Ny//2+20,Nx//2-20:Nx//2+20], cmap = "seismic")
plt.colorbar()
plt.title("Streamplot of current")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

N_from_center = 40
plt.streamplot(xv[Ny//2-N_from_center:Ny//2+N_from_center,Nx//2-N_from_center:Nx//2+N_from_center], yv[Ny//2-N_from_center:Ny//2+N_from_center,Nx//2-N_from_center:Nx//2+N_from_center], current_x[Ny//2-N_from_center:Ny//2+N_from_center,Nx//2-N_from_center:Nx//2+N_from_center].real, current_y[Ny//2-N_from_center:Ny//2+N_from_center,Nx//2-N_from_center:Nx//2+N_from_center].real, color = abs_current[Ny//2-N_from_center:Ny//2+N_from_center,Nx//2-N_from_center:Nx//2+N_from_center], cmap = "seismic")
plt.colorbar()
plt.title("Streamplot of current")
plt.xlabel("x")
plt.ylabel("y")
plt.show()


plt.pcolormesh(xv[Ny//2-N_from_center:Ny//2+N_from_center,Nx//2-N_from_center:Nx//2+N_from_center],yv[Ny//2-N_from_center:Ny//2+N_from_center,Nx//2-N_from_center:Nx//2+N_from_center],circulating_current[Ny//2-N_from_center:Ny//2+N_from_center,Nx//2-N_from_center:Nx//2+N_from_center], cmap = 'tab20', vmin = -np.max(np.abs(circulating_current[Ny//2-N_from_center:Ny//2+N_from_center,Nx//2-N_from_center:Nx//2+N_from_center])), vmax = np.max(np.abs(circulating_current[Ny//2-N_from_center:Ny//2+N_from_center,Nx//2-N_from_center:Nx//2+N_from_center])))
plt.colorbar()
plt.title("Circulation of current")
plt.show()

t2 = time.time()
print("Time taken", t2-t1)
