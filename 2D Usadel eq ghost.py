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
from scipy import sparse
from scipy import integrate as spint
from scipy.sparse import linalg as splinalg

import time

t1 = time.time()


def ghost_two_d_difference_matrix_usadel_neuman(x, y, dx, dy, A, eps=1, D=1, e_=1):
    #Assuming dx = dy for simplicity
    diag_x_shape = x.shape[0]+2  #The x-dimension of the final matrix
    diag_y_shape = y.shape[0]+2  #The y-dimension of the final matrix

    #Create the diagonal of the matrix and fill it with the proper values
    diag = np.zeros((diag_y_shape,diag_x_shape), dtype = complex)       
    
    #All points excluding the ghost points follow normal finite difference scheme
    diag[1:-1,1:-1] = -(2/(dx**2) + 2/(dy**2)  + 2j*eps/D -  4*e_*np.sum(A*A, axis = 2)) 
    #The ghost points at the end are different due to boundary conditions
    diag[0,:] = -1/(2*dy)
    diag[-1,:] = 1/(2*dy)
    diag[:,0] = 1/(2*dx)
    diag[:,-1] = 1/(2*dx)
    diag = np.reshape(diag, diag_x_shape*diag_y_shape)
    
    #Create the upper and lower diagonals and fil them with values according to the finite difference scheme
    #This corresponds the the points with x+dx and x-dx when looking at a points (x,y)
    upper_diag = np.zeros((diag_y_shape,diag_x_shape), dtype = complex)
    upper_diag[1:-1,1:-1] = (1/(dx**2)- 2j*e_*A[:,:,0]/(dx))
    lower_diag = np.zeros_like(upper_diag)
    lower_diag[1:-1,1:-1] = (1/(dx**2) + 2j*e_*A[:,:,0]/(dx))
    #Create the diagonals corresponding to y+dy and y-dy when looking at a points (x,y)
    #and fill them with values according to the finite difference scheme
    upup_diag = np.zeros_like(upper_diag)
    upup_diag[1:-1,1:-1] = (1/(dy**2) - 2j*e_*A[:,:,1]/(dy))
    
    lowlow_diag = np.zeros_like(upper_diag)
    lowlow_diag[1:-1,1:-1] = (1/(dy**2) + 2j*e_*A[:,:,1]/(dy))
    
    #Create and fill the diagonals corresponding to the boundary conditions with ghost points
    #These are offset by 2*dx and 2*dy
    top_bc_diag = np.zeros_like(upper_diag)
    bottom_bc_diag = np.zeros_like(upper_diag)
    right_bc_diag = np.zeros_like(upper_diag)
    left_bc_diag = np.zeros_like(upper_diag)
    
    top_bc_diag[-1,1:-1] = -1/(2*dy)
    bottom_bc_diag[0,1:-1] = 1/(2*dy)
    #right_bc_diag[1:-1,-1] = -1/(2*dx)
    #left_bc_diag[1:-1,0] = 1/(2*dx)
    
    #Reshape all arrays to be one dimensional and fix them to be the proper length
    upper_diag = np.reshape(upper_diag,diag.shape[0])[:-1]
    lower_diag = np.reshape(lower_diag, diag.shape[0])[1:]
    upup_diag = np.reshape(upup_diag, diag.shape[0])[:-diag_x_shape]
    lowlow_diag = np.reshape(lowlow_diag, diag.shape[0])[diag_x_shape:]
    
    top_bc_diag = np.reshape(top_bc_diag,diag.shape[0])[2*diag_x_shape:]
    bottom_bc_diag = np.reshape(bottom_bc_diag,diag.shape[0])[:-2*diag_x_shape]
    right_bc_diag = np.reshape(right_bc_diag, diag.shape[0])[2:]
    left_bc_diag = np.reshape(left_bc_diag, diag.shape[0])[:-2]
    
    #Create the final matrix with all elements in their proper places
    difference_matrix = sparse.diags((diag,upper_diag,lower_diag, upup_diag, lowlow_diag, top_bc_diag, bottom_bc_diag, right_bc_diag, left_bc_diag), 
                                     [0, 1, -1, diag_x_shape, -diag_x_shape, -2*diag_x_shape, 2*diag_x_shape, -2, 2])
    return difference_matrix
def get_Usadel_solution(epsilons_fun, f_sols_f,f_grads_f, x, y, dx, dy, A, theta = 0, D=1, e=-1):
    Nx = x.shape[0]
    Ny = y.shape[0]
    for i in range(epsilons_fun.shape[0]):
        g_diff_mat = ghost_two_d_difference_matrix_usadel_neuman(x, y, dx, dy, A, eps=epsilons_fun[i], D=D, e_=e).tocsc()
        g2=np.zeros(g_diff_mat.shape[0], dtype = complex)
        g2 = np.reshape(g2,(Ny+2,Nx+2))
        g2[1:Ny+1,0] = -10*np.sinh(np.arctanh(1/epsilons_fun[i]))* np.exp(1j*theta/2)
        g2[1:Ny+1,-1] = -10*np.sinh(np.arctanh(1/epsilons_fun[i])) * np.exp(-1j*theta/2)
        #g2[0,1:Nx+1] = -0.01/(2*dx)#np.sinh(np.arctanh(1/epsilons_fun[i]))# * np.exp(-1j*theta/2)
        #g2[Ny//2,Nx//2] = -(2/dx**2+2/dy**2) #-np.sinh(np.arctanh(1/epsilons_fun[i]))* np.exp(1j*theta/2)
        #g2[Ny//4,Nx//4] = (2/dx**2+2/dy**2)/4#-np.sinh(np.arctanh(1/epsilons_fun[i])) * np.exp(-1j*theta/2)
        #g2[3*Ny//4,3*Nx//4] = (2/dx**2+2/dy**2)/4#-np.sinh(np.arctanh(1/epsilons_fun[i])) * np.exp(-1j*theta/2)
        #g2[3*Ny//4,Nx//4] = (2/dx**2+2/dy**2)/4
        #g2[Ny//4,3*Nx//4] = (2/dx**2+2/dy**2)/4
        g2 = np.reshape(g2,(Nx+2)*(Ny+2))
        
        f_out= splinalg.spsolve(g_diff_mat,g2)
        f_grads_f[i] = grad_f(f_out.copy(), f_grads_f[i], x, y, dx, dy)
        f_new = np.reshape(np.reshape(f_out, (Ny+2,Nx+2))[1:-1,1:-1], (Ny,Nx))
        f_sols_f[i] = f_new
    return f_sols_f, f_grads
def grad_f(f, f_out, x, y, dx, dy):
    f_mat = np.reshape(f, (y.shape[0]+2, x.shape[0]+2))
    der_x_f = (f_mat[1:-1,2:] - f_mat[1:-1, :-2])/(2*dx)
    der_y_f = (f_mat[2:,1:-1] - f_mat[:-2,1:-1])/(2*dy)
    f_out[:,:,0] = der_x_f
    f_out[:,:,1] = der_y_f
    return f_out

Nx = 100
Ny = Nx//10
x,dx = np.linspace(0,2,Nx,retstep = True)
y, dy = np.linspace(0,4,Ny,retstep = True)
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
epsilons[0]+= 1e-5
#epsilons = np.array((1.5,2))
epsilons_minus = -1*epsilons
f_sols = np.zeros((epsilons.shape[0],Ny,Nx),dtype = complex)
f_sols_minus = np.zeros((epsilons.shape[0],Ny,Nx),dtype = complex)
f_grads = np.zeros((epsilons.shape[0],Ny,Nx,2),dtype = complex)
f_grads_minus = np.zeros((epsilons.shape[0],Ny,Nx,2),dtype = complex)

#theta = np.pi*6/2+0j
thetas = np.linspace(0,4*np.pi,256)
currents = np.zeros(thetas.shape[0], dtype = complex)
for i in range(thetas.shape[0]):
    if (i)%(thetas.shape[0]//4)==0 and i>0:
        print(i)
    theta = thetas[i]
    f_sols, f_grads = get_Usadel_solution(epsilons, f_sols, f_grads, x, y, dx, dy, A, theta = theta)
    

    f_sols_minus,f_grads_minus = get_Usadel_solution(epsilons_minus, f_sols_minus, f_grads_minus, x, y, dx, dy, A, theta = theta)



    current_x = np.trapz(f_sols*np.conjugate(f_grads_minus[:,:,:,0])- f_sols_minus*np.conjugate(f_grads[:,:,:,0]), x = epsilons, axis = 0)
    current_y = np.trapz(np.real(f_sols*np.conjugate(f_grads_minus[:,:,:,1])- f_sols_minus*np.conjugate(f_grads[:,:,:,1])), x = epsilons, axis = 0)
    currents[i] = np.average(current_x)
print("\n \n Max current", np.max(currents))
plt.plot(thetas, currents.real)
plt.title("Real current")
plt.show()
plt.plot(thetas, currents.imag)
plt.title("Imaginary part of current")
plt.show()
f_av = np.sum(f_sols, axis = 1)/f_sols.shape[1]
pair_corr = spint.trapz(f_sols-f_sols_minus, x = epsilons, axis = 0)

av_f = spint.trapz(f_sols, dx = dy, axis = 1)
plt.pcolormesh(x,y,current_x.real,cmap='seismic')
plt.colorbar()
plt.title("Current in x-direction")
plt.show()

plt.plot(x,np.average(current_x.real,axis = 0))
plt.title("Current in x-direction")
plt.show()

plt.pcolormesh(x,y,current_y.real,cmap='seismic')
plt.colorbar()
plt.title("Current in y-direction")
plt.show()



plt.pcolormesh(x,y,np.average(f_sols.real,axis = 0),cmap='seismic')
plt.title("Real part of avg solution")
plt.colorbar()
plt.show()
plt.pcolormesh(x,y,np.average(f_sols.imag,axis = 0),cmap='seismic')
plt.title("Imaginary part of average solution")
plt.colorbar()
plt.show()

t2 = time.time()
print("Time taken", t2-t1)
