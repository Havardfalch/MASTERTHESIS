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



from joblib import Parallel, delayed


import cProfile as profile
import pstats
t1 = time.time()
 


def int_func_for_A_x(t,x,y,lambd, threshold = 1, kappa=2.71, phi_0 = -np.pi):
    """
    Function used for generating the x-component of the vector potential A by integrating over the magnetic field using Poincare's lemma.
    It returns the value of the magnetic field at t*x and t*y multiplied by t*y.

    Parameters
    ----------
    t : Float
        Integration variable, should be between 0 and 1.
    x : Float
        x-coordinate.
    y : Float
        y-coordinate.
    lambd : Float
        The magnetic penetration depth.
    threshold : Float, optional
        The size of the vortex core where the magnetic field is constant, measured in units of coherence lengths. The default is 1.
    kappa : Float, optional
        The value of the Ginzburg-Landau parameter kappa, which is the magnetic penetration depth lambda divided by the coherence length.
        As we measure in the scale of the coherence length this is equal to the magnetic penetration depth. The default is 2.71.
    phi_0 : Float, optional
        The magnetic flux quantum. In natural units this is equal to -pi. The default is -pi.

    Returns
    -------
    Float
        The value of the magnetic field at point (tx,ty) multiplied by the integration variable t and y.

    """
    if np.sqrt((t*x)**2+(t*y)**2)<=threshold:
        return t*y*np.log(kappa)*phi_0/(2*np.pi*lambd**2)
    else:
        return t*y*phi_0/(2*np.pi*lambd**2) * spspecial.kn(0, np.sqrt((t*x)**2+(t*y)**2)/lambd) * np.log(kappa)/spspecial.kn(0,threshold/lambd)
    
def int_func_for_A_y(t,x,y,lambd, threshold = 1, kappa=2.71, phi_0 = -np.pi):
    """
    Function used for generating the y-component of the vector potential A by integrating over the magnetic field using Poincare's lemma.
    It returns the value of the magnetic field at t*x and t*y multiplied by -t*x.

    Parameters
    ----------
    t : Float
        Integration variable, should be between 0 and 1.
    x : Float
        x-coordinate.
    y : Float
        y-coordinate.
    lambd : Float
        The magnetic penetration depth.
    threshold : Float, optional
        The size of the vortex core where the magnetic field is constant, measured in units of coherence lengths. The default is 1.
    kappa : Float, optional
        The value of the Ginzburg-Landau parameter kappa, which is the magnetic penetration depth lambda divided by the coherence length.
        As we measure in the scale of the coherence length this is equal to the magnetic penetration depth. The default is 2.71.
    phi_0 : Float, optional
        The magnetic flux quantum. In natural units this is equal to -pi. The default is -pi.

    Returns
    -------
    Float
        The value of the magnetic field at point (tx,ty) multiplied by the integration variable t and -x.

    """
    if np.sqrt((t*x)**2+(t*y)**2)<=threshold:
        return -t*x*np.log(kappa)*phi_0/(2*np.pi*lambd**2)
    else:
        return -t*x*phi_0/(2*np.pi*lambd**2) * spspecial.kn(0, np.sqrt((t*x)**2+(t*y)**2)/lambd) * np.log(kappa)/spspecial.kn(0,threshold/lambd)

def B_strength(x,y,lambd, threshold = 1, kappa=1, phi_0 = -np.pi):
    if np.sqrt(x**2+y**2)<=threshold:
        return -np.log(kappa)*phi_0/(2*np.pi*lambd**2)
    else:
        return -phi_0/(2*np.pi*lambd**2) * spspecial.kn(0, np.sqrt((x)**2+(y)**2)/lambd) * np.log(kappa)/spspecial.kn(0,threshold/lambd)

def A_field_integrated_usadel_parallel_call(i,core_locs, A, x, y, Nx, Ny, dx, dy, threshold = 1, kappa = 2.71, phi_0 = -np.pi, ksi = 1):
    """
    Helper function for calculating the magnetic vector potential from the magnetic field using Poincare's lemma.
    For a given index of the y-coordinate it loops over all x coordinates and all superconducting vortex cores and adds the contribution
    from each vortex core to the vector potential in each spatial point it loops through.
    
    
    
    Parameters
    ----------
    i : Integer
        Index of the y coordinate to calculate the magnetic vector potential for.
    core_locs : 2D array of size (Number of vortices, 2)
        Array containing the location of the cortex cores.
    A : 3 dimensional numpy array of dimension (Ny, Nx, 2)
        Array of 0's to be filled with the calculated values of the magnetic vector potential,
        [:,:,0] components are the x-compontents while [:,:,1] are the y-components.
    x : 1D array of length Nx
        The x coordinates of the system.
    y : 1D array of length Ny
        The y coordinates of the system.
    Nx : Integer
        Number of grid points in the x-direction.
    Ny : Integer
        Number of grid points in the y-direction.
    dx : Float
        Steplength in the x-direction.
    dy : Float
        Steplength in the y-direction.
    threshold : Float, optional
        The size of the vortex core where the magnetic field is constant, measured in units of coherence lengths. The default is 1.
    kappa : Float, optional
        The value of the Ginzburg-Landau parameter kappa, which is the magnetic penetration depth lambda divided by the coherence length.
        As we measure in the scale of the coherence length this is equal to the magnetic penetration depth. The default is 2.71.
    phi_0 : Float, optional
        The magnetic flux quantum. In natural units this is equal to -pi. The default is -pi.
    ksi : Float, optional
        The coherence length. As we measure all lengths relative the coherence length this is 1. The default is 1.
    
    Returns
    -------
    A : 3 dimensional numpy array of dimension (Ny, Nx, 2)
        Array filled with the value of the vector potential at each point,
        [:,:,0] components are the x-compontents while [:,:,1] are the y-components.
    
    """
    B = np.zeros_like(A)
    xv,yv = np.meshgrid(x,y)
    lambd = kappa*ksi
    
    for j in range(xv.shape[1]):
        for k in range(core_locs.shape[0]):
            x_rel = xv[i,j]-core_locs[k,0]
            y_rel = yv[i,j]-core_locs[k,1]
            
            B[i,j,0] += spint.quad(int_func_for_A_x,0,1, args = (x_rel, y_rel, lambd, threshold, kappa, phi_0))[0]
            B[i,j,1] += spint.quad(int_func_for_A_y,0,1, args = (x_rel, y_rel, lambd, threshold, kappa, phi_0))[0]
    return B

def A_field_integrated_usadel_parallel(core_locs, A, x, y, Nx, Ny, dx, dy, threshold = 1, kappa = 2.71, phi_0 = -np.pi, ksi = 1):
    """
    Function for calculating the magnetic vector potential from the magnetic field using Poincare's lemma.
    It loops over all spacial coordinates and all superconducting vortex cores and adds the contribution
    from each vortex core to the vector potential in each spatial point.
    This function uses the joblib library to parallellize the first loop.
    
    
    Parameters
    ----------
    core_locs : 2D array of size (Number of vortices, 2)
        Array containing the location of the cortex cores.
    A : 3 dimensional numpy array of dimension (Ny, Nx, 2)
        Array of 0's to be filled with the calculated values of the magnetic vector potential,
        [:,:,0] components are the x-compontents while [:,:,1] are the y-components.
    x : 1D array of length Nx
        The x coordinates of the system.
    y : 1D array of length Ny
        The y coordinates of the system.
    Nx : Integer
        Number of grid points in the x-direction.
    Ny : Integer
        Number of grid points in the y-direction.
    dx : Float
        Steplength in the x-direction.
    dy : Float
        Steplength in the y-direction.
    threshold : Float, optional
        The size of the vortex core where the magnetic field is constant, measured in units of coherence lengths. The default is 1.
    kappa : Float, optional
        The value of the Ginzburg-Landau parameter kappa, which is the magnetic penetration depth lambda divided by the coherence length.
        As we measure in the scale of the coherence length this is equal to the magnetic penetration depth. The default is 2.71.
    phi_0 : Float, optional
        The magnetic flux quantum. In natural units this is equal to -pi. The default is -pi.
    ksi : Float, optional
        The coherence length. As we measure all lengths relative the coherence length this is 1. The default is 1.
    
    Returns
    -------
    A : 3 dimensional numpy array of dimension (Ny, Nx, 2)
        Array filled with the value of the vector potential at each point,
        [:,:,0] components are the x-compontents while [:,:,1] are the y-components.
    
    """
    A = np.sum(np.array(Parallel(n_jobs=4)(delayed(A_field_integrated_usadel_parallel_call)(i,core_locs, A, x, y, Nx, Ny, dx, dy, threshold, kappa, phi_0, ksi) for i in tqdm(range(y.shape[0])))), axis = 0)
    return A

def A_field_integrated_usadel(core_locs, A, x, y, Nx, Ny, dx, dy, threshold = 1, kappa = 2.71, phi_0 = -np.pi, ksi = 1):
    """
    Function for calculating the magnetic vector potential from the magnetic field using Poincare's lemma.
    It loops over all spacial coordinates and all superconducting vortex cores and adds the contribution
    from each vortex core to the vector potential in each spatial point. 
    

    Parameters
    ----------
    core_locs : 2D array of size (Number of vortices, 2)
        Array containing the location of the cortex cores.
    A : 3 dimensional numpy array of dimension (Ny, Nx, 2)
        Array of 0's to be filled with the calculated values of the magnetic vector potential,
        [:,:,0] components are the x-compontents while [:,:,1] are the y-components.
    x : 1D array of length Nx
        The x coordinates of the system.
    y : 1D array of length Ny
        The y coordinates of the system.
    Nx : Integer
        Number of grid points in the x-direction.
    Ny : Integer
        Number of grid points in the y-direction.
    dx : Float
        Steplength in the x-direction.
    dy : Float
        Steplength in the y-direction.
    threshold : Float, optional
        The size of the vortex core where the magnetic field is constant, measured in units of coherence lengths. The default is 1.
    kappa : Float, optional
        The value of the Ginzburg-Landau parameter kappa, which is the magnetic penetration depth lambda divided by the coherence length.
        As we measure in the scale of the coherence length this is equal to the magnetic penetration depth. The default is 2.71.
    phi_0 : Float, optional
        The magnetic flux quantum. In natural units this is equal to -pi. The default is -pi.
    ksi : Float, optional
        The coherence length. As we measure all lengths relative the coherence length this is 1. The default is 1.

    Returns
    -------
    A : 3 dimensional numpy array of dimension (Ny, Nx, 2)
        Array filled with the value of the vector potential at each point,
        [:,:,0] components are the x-compontents while [:,:,1] are the y-components.

    """
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

#Set parameters for the run
Nx = 100
Ny = Nx
e_ = -1
Lx = 20
Ly = Lx
Lz = 1
D = 1
G_T = 0.3
#Endre denne 
G_N = 1

threshold = 1
include_gbcs = False
kappa = 2.71
nu = 0.8

eV = 0
x,dx = np.linspace(-Lx/2,Lx/2,Nx,retstep = True)
y, dy = np.linspace(-Ly/2,Ly/2,Ny,retstep = True)
xv, yv = np.meshgrid(x,y)

"""
Lx2 = 70
Ly2 = Lx2

vortex_core_dist = 12
long_axis_dist = np.sin(np.pi/3) * vortex_core_dist
short_axis_dist = np.sin(np.pi/6) * vortex_core_dist
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

core_locs = np.array(core_locs)
print(np.where(core_locs[:,0]==np.max(core_locs[:,0])))

core_locs = np.delete(core_locs, np.where(core_locs[:,0]==np.max(core_locs[:,0])),0)
core_locs = np.delete(core_locs, np.where(core_locs[:,0]==np.max(core_locs[:,0])),0)
core_locs[:,0] += vortex_core_dist/2

core_locs = np.zeros((23,2))
for i in range(6):
    core_locs[i,0] = np.cos(i*2*np.pi/6)*np.max(x)*5/12
    core_locs[i,1] = np.sin(i*2*np.pi/6)*np.max(y)*5/12
for i in range(6):
    for j in range(2):
        core_locs[2*i+j+6,0] = np.cos(i*2*np.pi/6)*np.max(x)*5/12 + np.cos((i+j)*2*np.pi/6)*np.max(x)*5/12
        core_locs[2*i+j+6,1] = np.sin(i*2*np.pi/6)*np.max(y)*5/12 + np.sin((i+j)*2*np.pi/6)*np.max(y)*5/12
for i in range(4):
    core_locs[i+18,0] =  np.max(x)*10/12 * np.sign((i%2-1/2))
    core_locs[i+18,1] =  np.max(x)*10/12 * np.sin(1*np.pi/3) * np.sign(((i/3)%2-1/2))


#Specify the center of the superconducting vortices 
core_locs = np.zeros((7,2))
for i in range(6):
    core_locs[i,0] = np.cos(i*2*np.pi/6)*np.max(x)*5/6
    core_locs[i,1] = np.sin(i*2*np.pi/6)*np.max(y)*5/6

plt.scatter(core_locs[:,0], core_locs[:,1])
#plt.xlim(np.min(x), np.max(x))
#plt.ylim(np.min(y), np.max(y))
plt.grid()
plt.show()
"""
core_locs = np.zeros((1,2))

#Calculate the phase of the superconductor by adding the phase from each superconducting vortex
theta = 0
for i in range(core_locs.shape[0]):
    theta += np.arctan2(yv-core_locs[i,1], xv-core_locs[i,0])
#Fix the phase in the range[0,2*pi)
theta = theta%(2*np.pi)

#Calculate the magnetic vector potential
A = np.zeros((Ny, Nx,2))
A = A_field_integrated_usadel_parallel(core_locs, A, x, y, Nx, Ny, dx, dy, threshold, kappa)

slc = np.max((1, Nx//20))

#For kappa<1 this fixes the vector potential to be the correct direction since then the log changes sign
multiplier = 1 * np.sign(np.log(kappa))
A*=multiplier

plt.plot(x, A[Ny//2,:,0], label = "Ax")
plt.plot(x, A[Ny//2,:,1], label = "Ay")
plt.grid()
plt.legend()
plt.show()


B = ((A[1:-1,2:,1]-A[1:-1,:-2,1])/(2*dx) +(A[:-2, 1:-1, 0]- A[2:,1:-1,0])/(2*dy))

#Plotting to be removed at some point, just to see the form at y=0
B_strength_vec = np.vectorize(B_strength)
B_exact = B_strength_vec(x, y[Ny//2], kappa, threshold, kappa)*np.sign(np.log(kappa))
B_exact *= np.max(B)/np.max(B_exact)
plt.plot(x, B_exact, label =  "Exact form")
plt.plot(x[1:-1], B[Ny//2], label = "Finite difference form")
plt.title("B field at y = " + str(y[Ny//2]))
plt.legend()
plt.grid()
plt.show()

#Calculate the divergence of the magnetic field, this should be 0
div = ((A[1:-1,2:,0]-A[1:-1,:-2,0])/(2*dx) - (A[:-2, 1:-1, 1]- A[2:,1:-1,1])/(2*dy))

#Printing information to keep track of what parameters were used, should be removed at some point
print("Max B", np.max(B), "Min B", np.min(B))
print("Max div", np.max(np.abs(div)))
print("Applied voltage", eV)
print("Size in x:", Lx, "Points in x:", Nx, "Size in y:", Ly, "Points in y:", Ny)
print("Max of A", np.max(np.linalg.norm(A, axis = 2)), "Multiplier", multiplier)
print("Kappa", kappa, "Nu", nu, "G_T", G_T)
if include_gbcs:
    print("Using g_bcs term")
else:
    print("Not using g_bcs term")
    
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

#Calculate the current if the A-field part is removed. This should be taken away
current_x_no_A = e_*spint.trapz((f_sols*np.conjugate(f_grads_minus[:,:,:,0])- f_sols_minus*np.conjugate(f_grads[:,:,:,0]) ).real, x = epsilons, axis = 0).real
current_y_no_A = e_*spint.trapz((f_sols*np.conjugate(f_grads_minus[:,:,:,1])- f_sols_minus*np.conjugate(f_grads[:,:,:,1]) ).real, x = epsilons, axis = 0).real

abs_current_no_A = np.sqrt((current_x_no_A.real)**2+(current_y_no_A.real)**2)
circulating_current_no_A = abs_current*np.cos((theta - np.arctan2(-current_x_no_A, current_y_no_A)%(2*np.pi))%(2*np.pi))


print("Max inverted current", np.min(circulating_current))

#prof.disable()

#☺
#Calculate the divergence of the current
div_current_x = current_x[:,:-2]- current_x[:,2:]
div_current_y = current_y[:-2]- current_y[2:]

#Plots of lots of different stuff

plt.quiver(x[slc//2::slc],y[slc//2::slc],current_x[slc//2::slc,slc//2::slc].real, current_y[slc//2::slc,slc//2::slc].real)
plt.scatter(0,0)
plt.title("Supercurrent")
plt.show()
plt.quiver(x[slc//2::slc],y[slc//2::slc],current_x_no_A[slc//2::slc,slc//2::slc].real, current_y_no_A[slc//2::slc,slc//2::slc].real)
plt.scatter(0,0)
plt.title("Supercurrent without A field")
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

plt.pcolormesh(x[1:-1],y[1:-1], np.sqrt(div_current_x.real[1:-1]**2 + div_current_y.real[:,1:-1]**2),cmap='seismic')
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

plt.streamplot(xv, yv, current_x_no_A, current_y_no_A, color = abs_current_no_A, cmap = "viridis")
plt.colorbar()
plt.title("Streamplot of current without A field")
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

"""
center_right = core_locs[1]
print(center_right)
x_center_right = np.argmin(np.abs(x-center_right[0]))
y_center_right = np.argmin(np.abs(y-center_right[1]))

N_from_center = 40
plt.streamplot(xv[y_center_right-N_from_center:y_center_right+N_from_center,x_center_right-N_from_center:x_center_right+N_from_center], yv[y_center_right-N_from_center:y_center_right+N_from_center,x_center_right-N_from_center:x_center_right+N_from_center], current_x[y_center_right-N_from_center:y_center_right+N_from_center,x_center_right-N_from_center:x_center_right+N_from_center].real, current_y[y_center_right-N_from_center:y_center_right+N_from_center,x_center_right-N_from_center:x_center_right+N_from_center].real, color = abs_current[y_center_right-N_from_center:y_center_right+N_from_center,x_center_right-N_from_center:x_center_right+N_from_center], cmap = "seismic")
plt.colorbar()
plt.title("Streamplot of current")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
"""
N_from_center = 40
plt.streamplot(xv[Ny//2-N_from_center:Ny//2+N_from_center,Nx//2-N_from_center:Nx//2+N_from_center], yv[Ny//2-N_from_center:Ny//2+N_from_center,Nx//2-N_from_center:Nx//2+N_from_center], current_x[Ny//2-N_from_center:Ny//2+N_from_center,Nx//2-N_from_center:Nx//2+N_from_center].real, current_y[Ny//2-N_from_center:Ny//2+N_from_center,Nx//2-N_from_center:Nx//2+N_from_center].real, color = abs_current[Ny//2-N_from_center:Ny//2+N_from_center,Nx//2-N_from_center:Nx//2+N_from_center], cmap = "seismic")
plt.colorbar()
plt.title("Streamplot of current")
plt.xlabel("x")
plt.ylabel("y")
plt.show()


plt.pcolormesh(xv[Ny//2-N_from_center:Ny//2+N_from_center,Nx//2-N_from_center:Nx//2+N_from_center],yv[Ny//2-N_from_center:Ny//2+N_from_center,Nx//2-N_from_center:Nx//2+N_from_center],circulating_current[Ny//2-N_from_center:Ny//2+N_from_center,Nx//2-N_from_center:Nx//2+N_from_center], cmap = 'seismic', vmin = -np.max(np.abs(circulating_current[Ny//2-N_from_center:Ny//2+N_from_center,Nx//2-N_from_center:Nx//2+N_from_center])), vmax = np.max(np.abs(circulating_current[Ny//2-N_from_center:Ny//2+N_from_center,Nx//2-N_from_center:Nx//2+N_from_center])))
plt.colorbar()
plt.title("Circulation of current")
plt.show()



plt.plot(x[Nx//2:],abs_corr[Ny//2, Nx//2:], label = "Pair corr")
psi_0 = abs_corr[Ny//2, -1]
plt.plot(x[Nx//2:],psi_0 * np.tanh(0.1*x[Nx//2:]), label = "Nu = 0.1")
plt.plot(x[Nx//2:],psi_0 * np.tanh(0.25*x[Nx//2:]), label = "Nu = 0.25")
plt.plot(x[Nx//2:],psi_0 * np.tanh(0.8*x[Nx//2:]), label = "Nu = 0.8")
plt.plot(x[Nx//2:],psi_0 * np.tanh(1*x[Nx//2:]), label = "Nu = 1")
plt.plot(x[Nx//2:],psi_0 * np.tanh(1.5*x[Nx//2:]), label = "Nu = 1.5")
plt.legend()
plt.grid()
plt.show()

t2 = time.time()
print("Time taken", t2-t1)
