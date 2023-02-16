# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 08:53:44 2023

@author: Havard
"""
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg
from tqdm import tqdm

def ghost_two_d_difference_matrix_usadel_neuman(x, y, dx, dy, A, eps, D=1, e_=-1, use_kl = True):
    """
    Create the matrix describing the finite difference version of the Usadel equation
    Ghost points are used, i. e. an extra point is added outside the lattice in each row and column
    These are used to take bboundary conditions into account
    First all the necessary diagonals are created and transformed to rectangular forms corresponding to the padded grid
    Then they are filled with the correct values and finally transformed back to one dimensional arrays and fixed to the right length
    Finally the full matrix is created in sparse form

    Parameters
    ----------
    x : 1D array of length Nx
        The x coordinates of the system.
    y : 1D array of length Ny
        The y coordinates of the system.
    dx : Float
        Steplength in the x-direction.
    dy : Float
        Steplength in the y-direction.
    A : 3 dimensional numpy array of dimension (Ny, Nx, 2)
        Describes the vector potential at each point,
        [:,:,0] components are the x-compontents while [:,:,1] are the y-components.
    eps : Float, optional
        The value of the energy. The default is 1.
    D : Float, optional
        The value of the diffusion constant. The default is 1.
    e_ : Float, optional
        The value of the electron charge. The default is -1.

    Returns
    -------
    difference_matrix : 2D complex sparse array of shape ((Nx+2)*( Ny+2),(Nx+2)*( Ny+2))
        Contains the prefactors for the finite difference version of the Usadel equation to be solved when given boundary conditions.
    """
    
    diag_x_shape = x.shape[0]+2  #The x-dimension of the final matrix
    diag_y_shape = y.shape[0]+2  #The y-dimension of the final matrix

    #Create the diagonal of the matrix and fill it with the proper values
    diag = np.zeros((diag_y_shape,diag_x_shape), dtype = complex)       
    
    #All points excluding the ghost points follow normal finite difference scheme
    diag[1:-1,1:-1] = -(2/(dx**2) + 2/(dy**2)  + 1j*2*eps/D -  4*e_*np.sum(A*A, axis = 2)) 
    #The ghost points at the end are different due to boundary conditions
    diag[0,:] = 1/(2*dy)
    diag[-1,:] = 1/(2*dy)
    diag[:,0] = 1/(2*dx)
    diag[:,-1] = 1/(2*dx)
    if use_kl:
        diag[0,:] = -1/(2*dy)
        diag[:,0] = -1/(2*dx)
        
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
    if use_kl:
        top_bc_diag[-1,1:-1] = -1/(2*dy)
        bottom_bc_diag[0,1:-1] = 1/(2*dy)
        right_bc_diag[1:-1,-1] = -1/(2*dx)
        left_bc_diag[1:-1,0] = 1/(2*dx)
    
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

def grad_f(f, f_out, x, y, dx, dy):
    """
    Calculates the gradients in the x and y directions of f.
    
    Parameters
    ----------
    f : 1D complex array of length (Nx+2)*(Ny+2)
        The value of the off diagonal components of the greens function at all points in the system.
    f_out : 3D complex array of shape (Ny, Nx, 2)
        An array to save the values of the gradients in all points in the system in.
    x : 1D array of length Nx
        The x coordinates of the system.
    y : 1D array of length Ny
        The y coordinates of the system.
    dx : Float
        Steplength in the x-direction.
    dy : Float
        Steplength in the y-direction.

    Returns
    -------
    f_out : D array of shape (Ny, Nx, 2)
        The values of the gradient in all points in the system.

    """
    
    f_mat = np.reshape(f, (y.shape[0]+2, x.shape[0]+2))
    der_x_f = (f_mat[1:-1,2:] - f_mat[1:-1, :-2])/(2*dx)
    der_y_f = (f_mat[2:,1:-1] - f_mat[:-2,1:-1])/(2*dy)
    f_out[:,:,0] = der_x_f
    f_out[:,:,1] = der_y_f
    return f_out

def get_Usadel_solution(epsilons_fun, f_sols_f,f_grads_f, bc_fun, x, y, dx, dy, A, theta = 0, D=1, e=-1, use_kl = False, gamma = 3):
    """
    Solves the finite difference version of the Usadel equation using ghost points given
    the vector potential A and a discretized space (x,y), for the values of the energies
    given in epsilons_fun and boundary conditions given by bc_fun.
    First the discretized matrix is found using the function 
    ghost_two_d_difference_matrix_usadel_neuman and using the boundary conditions given
    by bc_fun the solution is found using scipy.sparse.linalge.spsolve. Then the ghost 
    points are removed from the edges and the solution is saved. This is done for all
    energies given.    

    Parameters
    ----------
    epsilons_fun : 1D numpy array of shape Ne
        Contains all the energies the Usadel equation is solved for.
    f_sols_f : 3D complex numpy array of shape (Ne, Ny, Nx)
        An array to save the value of the off diagonal components of the greens function for all energies and all points in the system.
    f_grads_f :  4D complex numpy array of shape (Ne, Ny, Nx, 2).
        An array to salve the gradient of the off diagonal components of the greens function for all energies and all points in the system.
        [:,:,:,0] is the x-component of the gradient, while [:,:,:,1] is the y-component of the gradient.
    bc_fun : Function, returns 1D complex array of shape (Nx+2)*(Ny+2)
        Function for calculating the boundary conditions for the given value of the energy epsilon and phase difference theta.
    x : 1D array of length Nx
        The x coordinates of the system.
    y : 1D array of length Ny
        The y coordinates of the system.
    dx : Float
        Steplength in the x-direction.
    dy : Float
        Steplength in the y-direction.
    A : 3 dimensional numpy array of dimension (Ny, Nx, 2)
        Describes the vector potential at each point,
        [:,:,0] components are the x-compontents while [:,:,1] are the y-components.
    theta : Float, optional
        The phase shift difference between the two superconductors in case of a SNS system. The default is 0.
    D : Float, optional
        The value of the diffusion constant. The default is 1.
    e_ : Float, optional
        The value of the electron charge. The default is -1.

    Returns
    -------
    f_sols_f : 3D complex numpy array of shape (Ne, Ny, Nx)
        An array contaning the value of the off diagonal components of the greens function for all energies and all points in the system.
    f_grads_f :  4D complex numpy array of shape (Ne, Ny, Nx, 2).
        An array containing the gradient of the off diagonal components of the greens function for all energies and all points in the system.
        [:,:,:,0] is the x-component of the gradient, while [:,:,:,1] is the y-component of the gradient.

    """
    Nx = x.shape[0]
    Ny = y.shape[0]
    for i in tqdm(range(epsilons_fun.shape[0])):
        g_diff_mat = ghost_two_d_difference_matrix_usadel_neuman(x, y, dx, dy, A, epsilons_fun[i], D=D, e_=e, use_kl = use_kl).tocsc()
        g = bc_fun(epsilons_fun[i], theta, Nx, Ny, use_kl, gamma = gamma)
        f_out= splinalg.spsolve(-g_diff_mat,g, use_umfpack=False)
        f_grads_f[i] = grad_f(f_out.copy(), f_grads_f[i], x, y, dx, dy)
        f_new = np.reshape(np.reshape(f_out, (Ny+2,Nx+2))[1:-1,1:-1], (Ny,Nx))
        f_sols_f[i] = f_new
    return f_sols_f, f_grads_f

def update_A_field(x_grid, y_grid, x_current, y_current, dx, dy, mu_0 = 1):
    """
    Calculates the change in the magnetic vector potential A due to the currents in the system.

    Parameters
    ----------
    x_grid : 2d array of shape (Ny,Nx)
        Contains the x_coordinate of each grid point.
    y_grid : 2d array of shape (Ny,Nx)
        Contains the y_coordinate of each grid point.
    x_current : 2d array of shape (Ny,Nx)
        Contains the current in the x direction for each grid point.
    y_current : 2d array of shape (Ny,Nx)
        Contains the current in the y direction for each grid point.
    dx : Float
        The spacing between grid points in the x direction.
    dy : Float
        The spacing between grid points in the x direction.
    mu_0 : Flota, optional
        The value of the vacuum permeability. The default is 1.

    Returns
    -------
    delta_A : 3d array of shape (Ny,Nx,2)
        The magnetic vector potential induced by the currents in the material.

    """
    delta_A = np.zeros((x_current.shape[0], x_current.shape[1], 2), dtype = float)
    for i in range(x_current.shape[0]):
        for j in range(x_current.shape[1]):
            divisor = ((x_grid[i,j]-x_grid)**2+(y_grid[i,j]-y_grid)**2)
            divisor[divisor==0.0] = np.min(divisor + (divisor==0)*np.max(divisor)) #Avoid 0 distance and thus infinite contribution
            distances = 1/np.sqrt(divisor)
            delta_A[i,j,0] = mu_0/(4*np.pi) * np.sum(x_current/distances)*dx*dy
            delta_A[i,j,1] = mu_0/(4*np.pi) * np.sum(y_current/distances)*dx*dy
    return delta_A
