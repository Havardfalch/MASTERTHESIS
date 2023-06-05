# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 08:53:44 2023

@author: Havard
"""
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg
from scipy import integrate as spint
from scipy import special as spspecial
from joblib import Parallel, delayed
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
    use_kl : Bool, optional
        Boolean deciding which type of boundary condition to use.
        If True Kuprianov-Lukichev boundary conditions will be used.
        If False transparent boundary conditions will be used.
        The default is True.

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
    diag[1:-1,1:-1] = -(2/(dx**2) + 2/(dy**2)   +  4*e_**2*np.sum(A*A, axis = 2)) + 1j*2*(eps)/D
    #The ghost points at the end are different due to boundary conditions
    diag[0,:] = 1/(2*dy)
    diag[-1,:] = 1/(2*dy)
    diag[:,0] = 1/(2*dx)
    diag[:,-1] = 1/(2*dx)
    if use_kl:
        diag[0,:] = -1/(2*dy)
        diag[:,0] = -1/(2*dx)
        
    #Reshape the diagonal to have the proper shape
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
    #The arrays to the rights ot the diagonal remove values from the end of the array while
    #arrays to the left remove from the start. This is since arrays to the right represent
    #points later in the flattened grid while the ones to the left refer to points earlier
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
    #Reshape the Green's function to have the same shape as the system with ghost points
    f_mat = np.reshape(f, (y.shape[0]+2, x.shape[0]+2))
    
    #Calculate the gradients in the x and y direction
    der_x_f = (f_mat[1:-1,2:] - f_mat[1:-1, :-2])/(2*dx)
    der_y_f = (f_mat[2:,1:-1] - f_mat[:-2,1:-1])/(2*dy)
    
    #Fill the given matrix with the gradients in the correct positions
    f_out[:,:,0] = der_x_f
    f_out[:,:,1] = der_y_f
    
    return f_out

def get_Usadel_solution(epsilons_fun, f_sols_f,f_grads_f, bc_fun, x, y, dx, dy, A, theta = 0, D=1, e=-1, use_kl = True, gamma = 3):
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
    use_kl : Bool, optional
        Boolean deciding which type of boundary condition to use.
        If True Kuprianov-Lukichev boundary conditions will be used.
        If False transparent boundary conditions will be used.
        The default is True.
    gamma : Float, optional
        The value of the conductance of the interface in Kuprianov Lukichev boundary
        conditions. The default is 3.

    Returns
    -------
    f_sols_f : 3D complex numpy array of shape (Ne, Ny, Nx)
        An array contaning the value of the off diagonal components of the greens function for all energies and all points in the system.
    f_grads_f :  4D complex numpy array of shape (Ne, Ny, Nx, 2).
        An array containing the gradient of the off diagonal components of the greens function for all energies and all points in the system.
        [:,:,:,0] is the x-component of the gradient, while [:,:,:,1] is the y-component of the gradient.

    """
    #Create integers with the dimensions of the system
    Nx = x.shape[0]
    Ny = y.shape[0]
    #Loop through all energies to solve the linearized Usadel equation for
    #tqdm creates a progressbar to see how far in the simulation one has gotten
    for i in tqdm(range(epsilons_fun.shape[0])):
        #Create the matrix describing the finite difference version of the linearized Usadel
        #equation where the lhs contains all coefficents in front of the various f_ij
        #using ghost points so the boundary conditions are also included
        g_diff_mat = ghost_two_d_difference_matrix_usadel_neuman(x, y, dx, dy, A, epsilons_fun[i], D=D, e_=e, use_kl = use_kl).tocsc()
        
        #Create the rhs that is independent of f and also contains the boundary conditions
        g = bc_fun(epsilons_fun[i], theta, Nx, Ny, use_kl, gamma = gamma)
        
        #Solve the matrix equation system
        f_out= splinalg.spsolve(g_diff_mat,g, use_umfpack=False)
        
        #Calculate the gradients
        f_grads_f[i] = grad_f(f_out.copy(), f_grads_f[i], x, y, dx, dy)
        
        #Reshape the solution into the same shape as the system and remove the ghost points
        f_sols_f[i] = np.reshape(np.reshape(f_out, (Ny+2,Nx+2))[1:-1,1:-1], (Ny,Nx))
        
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
    
    #Create an arary to store the A field induced by the currents with the same shape as 
    #the system and space to save both the x and y component
    delta_A = np.zeros((x_current.shape[0], x_current.shape[1], 2), dtype = float)
    #Loop through the indices for the y-coordinate, tqdm creates a progressbar.
    for i in tqdm(range(x_current.shape[0])):
        #Loop through the indices of the x-coordinate
        for j in range(x_current.shape[1]):
            #Calculate the square of the distance from a grid point to all other grid points
            distances = np.sqrt((x_grid[i,j]-x_grid)**2+(y_grid[i,j]-y_grid)**2)
            #Set the smallest distance which should be zero equal to the second smallest 
            #distance. This is done to avoid infinities when dividing by the distance
            distances[distances==0.0] = np.inf #np.min(distances + (distances==0)*np.max(distances))
            #Calculate the inverse of the distance
            inverse_distances = 1/distances
            #Calculate the contribution to the A-field in the x and y direction using 
            #Maxwells equations. The multiplication with dx and dy is to take into account
            #that we are integrating
            delta_A[i,j,0] = mu_0/(4*np.pi) * np.trapz(np.trapz(x_current*inverse_distances, x=x_grid[0], axis = 1), x = y_grid[:,0])
            delta_A[i,j,1] = mu_0/(4*np.pi) * np.trapz(np.trapz(y_current*inverse_distances, x=x_grid[0], axis = 1), x = y_grid[:,0])
            
            """delta_A[i,j,0] = mu_0/(4*np.pi) * np.sum(x_current/inverse_distances)*dx*dy
            delta_A[i,j,1] = mu_0/(4*np.pi) * np.sum(y_current/inverse_distances)*dx*dy"""
    return delta_A

def get_integrated_3d_Usadel_solution(epsilons_fun, f_sols_f,f_grads_f, f_bcs_fun, x, y, dx, dy, A, theta = 0, D=1, e_=-1, Lz=1, G_T=1, G_N=1, core_locs = np.array([[None,None]]), threshold = 1, nu = 1, include_gbcs = False):
    """
    Solves the finite difference version of the Usadel equation using ghost points given
    the vector potential A and a discretized space (x,y), for the values of the energies
    given in epsilons_fun and boundary conditions given by bc_fun.
    First the discretized matrix is found using the function 
    ghost_two_d_difference_matrix_usadel_neuman and using the boundary conditions and right
    hand side given by f_bcs_fun the solution is found using scipy.sparse.linalge.spsolve.
    Then the ghost points are removed from the edges and the solution is saved. This is done
    for all energies given in epsilons_fun.    

    Parameters
    ----------
    epsilons_fun : 1D numpy array of shape Ne
        Contains all the energies the Usadel equation is solved for.
    f_sols_f : 3D complex numpy array of shape (Ne, Ny, Nx)
        An array to save the value of the off diagonal components of the greens function for all energies and all points in the system.
    f_grads_f :  4D complex numpy array of shape (Ne, Ny, Nx, 2).
        An array to salve the gradient of the off diagonal components of the greens function for all energies and all points in the system.
        [:,:,:,0] is the x-component of the gradient, while [:,:,:,1] is the y-component of the gradient.
    f_bcs_fun : Function, returns 1D complex array of shape (Nx+2)*(Ny+2)
        Function for calculating the right hand side of the Usadel equation for the given value of the
        energy epsilon and complex phase theta.
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
    Lz : Float, optional
        The thickness of the normal metal thin film. The default is 1.
    G_T : Float, optional
        Tunneling conductance between the normal metal and the superconductor. The default is 1.
    G_N : Float, optional
        Conductance of the norma metal. The default is 1.
    core_locs : 2D array of size (Number of vortices, 2)
        Array containing the location of the cortex cores. The default is [[None,None]]
    threshold : Float, optional
        The size of the vortex cores meausured in number of coherence lengths. The default is 1.
    nu : Float, optional
        A number containing information about how fast the superconductivity recovers outside of a vortex.
        The default is 1.
    include_gbcs : Bool, optional
        A boolean deciding if the g_bcs term from the integrated Kuprianov-Lukichev boundary
        conditions is included. The default is False.

    Returns
    -------
    f_sols_f : 3D complex numpy array of shape (Ne, Ny, Nx)
        An array contaning the value of the off diagonal components of the greens function for all energies and all points in the system.
    f_grads_f :  4D complex numpy array of shape (Ne, Ny, Nx, 2).
        An array containing the gradient of the off diagonal components of the greens function for all energies and all points in the system.
        [:,:,:,0] is the x-component of the gradient, while [:,:,:,1] is the y-component of the gradient.

    """
    #Create integers with the dimensions of the system
    Nx = x.shape[0]
    Ny = y.shape[0]
    #Loop through all energies to solve the linearized Usadel equation for
    #tqdm creates a progressbar to see how far in the simulation one has gotten
    for i in tqdm(range(epsilons_fun.shape[0])):
        #Create the matrix describing the finite difference version of the linearized Usadel
        #equation where the lhs contains all coefficents in front of the various f_ij
        #using ghost points so the boundary conditions are also included
        g_diff_mat = ghost_two_d_integrated_difference_matrix_usadel(x, y, dx, dy, A, epsilons_fun[i], D=D, e_=e_, Lz=Lz, G_T=G_T, G_N=G_N, core_locs = core_locs, threshold=threshold, nu=nu, include_gbcs = include_gbcs).tocsc()
        
        #Create the rhs that is independent of f and also contains the boundary conditions
        f_bcs = f_bcs_fun(epsilons_fun[i], theta, Nx, Ny, x, y, Lz, G_T, G_N, core_locs, threshold, nu)
        
        #Solve the matrix equation system
        f_out= splinalg.spsolve(g_diff_mat,f_bcs, use_umfpack=False)
        
        #Calculate the gradients
        f_grads_f[i] = grad_f(f_out.copy(), f_grads_f[i], x, y, dx, dy)
        
        #Reshape the solution into the same shape as the system and remove the ghost points
        f_sols_f[i] = np.reshape(np.reshape(f_out, (Ny+2,Nx+2))[1:-1,1:-1], (Ny,Nx))
        
    return f_sols_f, f_grads_f

def ghost_two_d_integrated_difference_matrix_usadel(x, y, dx, dy, A, eps, D=1, e_=-1, Lz=1, G_T = 1, G_N = 1, core_locs = [[None,None]], threshold = 1, nu = 1, include_gbcs = False):
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
    Lz : Float, optional
        The thickness of the normal metal thin film. The default is 1.
    G_T : Float, optional
        Tunneling conductance between the normal metal and the superconductor. The default is 1.
    G_N : Float, optional
        Conductance of the norma metal. The default is 1.
    core_locs : 2D array of size (Number of vortices, 2)
        Array containing the location of the cortex cores. The default is [[None,None]]
    threshold : Float, optional
        The size of the vortex cores meausured in number of coherence lengths. The default is 1.
    nu : Float, optional
        A number containing information about how fast the superconductivity recovers outside of a vortex.
        The default is 1.
    include_gbcs : Bool, optional
        A boolean deciding if the g_bcs term from the integrated Kuprianov-Lukichev boundary
        conditions is included. The default is False.

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
    diag[1:-1,1:-1] = -(2/(dx**2) + 2/(dy**2)   +  4*e_**2*np.sum(A*A, axis = 2)) + 1j*2*(eps)/D

    if core_locs[0,0]!=None and include_gbcs:
            delta = 1
            xv, yv = np.meshgrid(x,y)
            for i in range(core_locs.shape[0]):
                distance = np.sqrt((xv- core_locs[i,0])**2 + (yv - core_locs[i,1])**2)
                delta *= np.tanh(nu*(distance))
            g_bcs_contrib = (1/(Lz**2)) * (G_T/G_N) * np.cosh(np.arctanh(delta/eps))
            diag[1:-1,1:-1] += g_bcs_contrib 
     
    #The ghost points at the end are different due to boundary conditions
    diag[0,:] = -1/(2*dx)
    diag[-1,:] = 1/(2*dx)
    diag[:,0] = -1/(2*dy)
    diag[:,-1] = 1/(2*dy)
        
    #Reshape the diagonal to have the proper shape
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
    right_bc_diag[1:-1,-1] = -1/(2*dx)
    left_bc_diag[1:-1,0] = 1/(2*dx)
    
    lowlow_diag[-1,1:-1] = -2j*e_*A[-1,:, 1]
    upup_diag[0,1:-1] = -2j*e_*A[0,:, 1]
    upper_diag[1:-1,0] = -2j*e_*A[:,0,0]
    lower_diag[1:-1,-1] = -2j*e_*A[:,-1,0]

    
    #Reshape all arrays to be one dimensional and fix them to be the proper length
    #The arrays to the rights ot the diagonal remove values from the end of the array while
    #arrays to the left remove from the start. This is since arrays to the right represent
    #points later in the flattened grid while the ones to the left refer to points earlier
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
    #Create an array to store the vector potential in
    A_new = np.zeros_like(A)
    
    #Create a mesh of the grid points
    xv,yv = np.meshgrid(x,y)
    
    #Calculate lambda
    lambd = kappa*ksi
    
    #Loop over the x coordinates
    for j in range(xv.shape[1]):
        #Loop over the vortices in the normal metal
        for k in range(core_locs.shape[0]):
            #Calculate the x-distance from the center of the vortex
            x_rel = xv[i,j]-core_locs[k,0]
            #Calculate the y-distance from the center of the vortex
            y_rel = yv[i,j]-core_locs[k,1]
            
            #Calculate the contribution to x component of the vector potential from one vortex at one gridpoint
            A_new[i,j,0] += spint.quad(int_func_for_A_x,0,1, args = (x_rel, y_rel, lambd, threshold, kappa, phi_0))[0]
            #Calculate the contribution to y component of the vector potential from one vortex at one gridpoint
            A_new[i,j,1] += spint.quad(int_func_for_A_y,0,1, args = (x_rel, y_rel, lambd, threshold, kappa, phi_0))[0]
    return A_new

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
    #Create a mesh of the grid points
    xv,yv = np.meshgrid(x,y)
    #Calculate lambda
    lambd = kappa*ksi
    #Loop over the y coordinates, tqdm creates a progressbar
    for i in tqdm(range(xv.shape[0])):
        #Loop over the x coordinates
        for j in range(xv.shape[1]):
            #Loop over the vortices in the normal metal
            for k in range(core_locs.shape[0]):
                #Calculate the x-distance from the center of the vortex
                x_rel = xv[i,j]-core_locs[k,0]
                #Calculate the y-distance from the center of the vortex
                y_rel = yv[i,j]-core_locs[k,1]
                
                #Calculate the contribution to x component of the vector potential from one vortex at one gridpoint
                A[i,j,0] += spint.quad(int_func_for_A_x,0,1, args = (x_rel, y_rel, lambd, threshold, kappa, phi_0))[0]
                #Calculate the contribution to y component of the vector potential from one vortex at one gridpoint
                A[i,j,1] += spint.quad(int_func_for_A_y,0,1, args = (x_rel, y_rel, lambd, threshold, kappa, phi_0))[0]
    return A
