# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 10:41:59 2023

@author: Havard
"""
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg


def g_bc_SN(epsilon, theta, Nx, Ny, gamma = None):
    """
    Calculate the boundary condition for a normal metal influenced by a superconductor via
    the proximity effect when the superconductor is only on the left side of the normal metal.

    Parameters
    ----------
    epsilon : Float
        The value of the energy.
    theta : Float
        Not included in this boundary condition. The phase winding in the supercondctor.
    Nx : Integer
        Number of grid points in the x-direction.
    Ny : TYPE
        Number of grid points in the y-direction.
    gamma : TYPE, optional
        Not included in this boundary condition. The value of the conductance of the
        interface in Kuprianov Lukichev boundary condition. The default is None.

    Returns
    -------
    g : 1D array of shape ((Nx+2)*(Ny+2))
        Array containing the values of the boundary condition.

    """
    
    #Create an array to store all the boundary condition values in including the ghost points
    g=np.zeros((Ny+2, Nx+2), dtype = complex)
    
    #Fill the points corresponding to the superconductor
    g[1:Ny+1,0] = -np.sinh(np.arctanh(1/epsilon))
    
    #Reshape into a one dimensional array to work with scipy's spsolve
    g = np.reshape(g,(Nx+2)*(Ny+2))
    return g


def g_bc_SNS(epsilon, theta, Nx, Ny, gamma = None):
    """
    Calculate the boundary condition at energy epsilon for a normal metal influenced by
    a superconductor via the proximity effect when there are superconductors on the 
    left and right hand side of the normal metal and these superconductors have a phase
    difference theta.

    Parameters
    ----------
    epsilon : Float
        The value of the energy.
    theta : Float
        The phase differencce between the two supercondctors.
    Nx : Integer
        Number of grid points in the x-direction.
    Ny : TYPE
        Number of grid points in the y-direction.
    gamma : TYPE, optional
        Not included in this boundary condition. The value of the conductance of the
        interface in Kuprianov Lukichev boundary condition. The default is None.

    Returns
    -------
    g : 1D array of shape ((Nx+2)*(Ny+2))
        Array containing the values of the boundary condition.
    """
    
    #Create an array to store all the boundary condition values in including the ghost points
    g=np.zeros((Ny+2, Nx+2), dtype = complex)
   
    #Fill the points corresponding to the left superconductor and give it the proper phase
    g[1:Ny+1,0] = -np.sinh(np.arctanh(1/epsilon))* np.exp(1j*theta/2)/3
    
    #Fill the points corresponding to the right superconductor and give it the proper phase
    g[1:Ny+1,-1] = -np.sinh(np.arctanh(1/epsilon)) * np.exp(-1j*theta/2)/3

    #Reshape into a one dimensional array to work with scipy's spsolve
    g = np.reshape(g,(Nx+2)*(Ny+2))
    return g

def f_integrated_z_direction(epsilon, theta, Nx, Ny, x, y, Lz=1, G_T=0.3, G_N=1, core_locs = [[None, None]], threshold  = 1, nu=1):
    """
    Calculate the right hand side of the linearized Usadel equation when a thin film normal
    metal is placed on top of a superconductor and the Kuprianov-Lukichev boundary conditions
    have been integrated out to become a part of the Usadel equation. As ghost points are
    used and vacuum is surrounding the normal metal 

    Parameters
    ----------
    epsilon : Float
        The value of the energy.
    theta : Float
        The complex phase of the underlying superconductor.
    Nx : Integer
        Number of grid points in the x-direction.
    Ny : TYPE
        Number of grid points in the y-direction.
    x : 1D array of length Nx
        The x coordinates of the system.
    y : 1D array of length Ny
        The y coordinates of the system.
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

    Returns
    -------
    f : 1D array of length (Nx+2)*(Ny+2)
        An array containing the right hand side of the linearized Usadel equation.

    """
    f=np.zeros((Ny+2, Nx+2), dtype = complex)
    delta = 1
    if core_locs[0,0]!= None:
        xv, yv = np.meshgrid(x,y)
        for i in range(core_locs.shape[0]):
            distance = np.sqrt((xv- core_locs[i,0])**2 + (yv - core_locs[i,1])**2)
            delta *= np.tanh(nu*(distance)) #*(distance>threshold)
    #Check if there should be a minus sign here
    f[1:-1, 1:-1] = -(G_T/G_N) *(1/(Lz**2))*np.sinh(np.arctanh(delta/epsilon))*np.exp(1j*theta)
    f = np.reshape(f, (Nx+2)*(Ny+2))
    return f

def g_bc_SSNSS(epsilon, theta, Nx, Ny, use_kl = True, gamma = 3):
    """
    Calculate the boundary condition at energy epsilon for a normal metal influenced by
    a superconductor via the proximity effect when there are superconductors on all 
    four sides of the normal metal and the superconductor has a continously varying phase
    with total phase winding of theta.

    Parameters
    ----------
    epsilon : Float
        The value of the energy.
    theta : Float
        The total phase winding of the surrounding superconductor.
    Nx : Integer
        Number of grid points in the x-direction.
    Ny : TYPE
        Number of grid points in the y-direction.
    use_kl : Bool, optional
        Boolean deciding which type of boundary condition to use.
        If True Kuprianov-Lukichev boundary conditions will be used.
        If False transparent boundary conditions will be used.
        The default is True.
    gamma : TYPE, optional
        The value of the conductance of the interface in Kuprianov Lukichev boundary
        condition. The default is 3.

    Returns
    -------
    g : 1D array of shape ((Nx+2)*(Ny+2))
        Array containing the values of the boundary condition.
    """
    #Create an array to store all the boundary condition values in including the ghost points
    g=np.zeros((Ny+2, Nx+2), dtype = complex)
    #Create an array with continously varying phase with total phase winding theta
    phase = np.exp(1j*(np.linspace(0,theta,(2*Ny)+(2*Nx), endpoint = False)))
    if use_kl:
        #Fill the points corresponding to the bottom row of ghost points
        #There is no minus sign since the normal metal is over (larger y than) the
        #superconductor
        g[0,1:Nx+1] = np.sinh(np.arctanh(1/epsilon)) *phase[0:Nx]
        
        #Fill the points corresponding to the right row of ghost points
        #There is a minus sign since the normal metal is on the left (smaller x) of the
        #superconductor
        g[1:Ny+1,-1] = -np.sinh(np.arctanh(1/epsilon)) *phase[Nx:Ny+Nx]
        
        #Fill the points corresponding to the top row of ghost points, the phase is flipped
        #since it goes clockwise and thus increases when x decreases
        #There is a minus sign since the normal metal is below (smaller y than) the
        #superconductor
        g[-1,1:Nx+1] = -np.sinh(np.arctanh(1/epsilon))*np.flip(phase[Ny+Nx:Ny+(2*Nx)])
        
        #Fill the points corresponding to the left row of ghost points, the phase is flipped
        #since it goes clockwise and thus increases when y decreases
        #Fill the points corresponding to the right row of ghost points
        #There is no minus sign since the normal metal is on the right (larger x) of the
        #superconductor
        g[1:Ny+1,0] = np.sinh(np.arctanh(1/epsilon))*np.flip(phase[Ny+(2*Nx):(2*Ny)+(2*Nx)])
        
        #Divide by gamma from Kuprianov-Lukichev boundary conditions
        g/=gamma
    else:
        #Fill the points corresponding to the bottom row of ghost points
        g[0,1:Nx+1] = -np.sinh(np.arctanh(1/epsilon)) *phase[0:Nx]
        #Fill the points corresponding to the right row of ghost points
        g[1:Ny+1,-1] = -np.sinh(np.arctanh(1/epsilon)) *phase[Nx:Ny+Nx]
        #Fill the points corresponding to the top row of ghost points, the phase is flipped
        #since it goes clockwise and thus increases when x decreases
        g[-1,1:Nx+1] = -np.sinh(np.arctanh(1/epsilon))*np.flip(phase[Ny+Nx:Ny+(2*Nx)])
        #Fill the points corresponding to the left row of ghost points, the phase is flipped
        #since it goes clockwise and thus increases when y decreases
        g[1:Ny+1,0] = -np.sinh(np.arctanh(1/epsilon))*np.flip(phase[Ny+(2*Nx):(2*Ny)+(2*Nx)])
    
    #Reshape into a one dimensional array to work with scipy's spsolve
    g = np.reshape(g,(Nx+2)*(Ny+2))
    
    return g

def calculate_correct_A_field(B_and_bc, Nx, Ny, dx, dy):
    """
    Calculates the divergence free (Coloumb gauge) vector potential for a normal metal
    surrounded by a superconductor on all four sides. It does this for a given strength
    of the magnetic field B and given boundary conditions for the value of the vector
    potential in the surrounding superconductor.
    This is done by writing the zereo divergence condition and that the curl of the A-field
    equals the B-field using finite differences and using ghost points to take the boundary
    conditions into account. These equations are collected into a single matrix equation
    which is then solve. Except for the boundary conditions every other equation is 
    a divergence condition and every other equation is an equation for the B-field.
    This does not work when Nx and Ny are odd and Nx+1 and Ny+1 are divisible by 4.

    Parameters
    ----------
    B_and_bc : 3D array of shape (Ny+2, Nx+2, 2)
        Contains the value of the B field and the boundary conditions of the A-field.
        The values in B[0], B[-1], B[:,0] and B[:,-1] are the boundary conditions.
        In B[1:-1,1:-1, 1] are the values of the B-field which should be the same constant.
        B[1:-1, 1:-1, 0] should be zero since this is part of the divergence condition.
    Nx : Integer
        Number of grid points in the x-direction.
    Ny : TYPE
        Number of grid points in the y-direction.
    dx : Float
        Steplength in the x-direction.
    dy : Float
        Steplength in the y-direction.

    Returns
    -------
    A : 3D array of shape (Ny+2, Nx+2, 2)
        Contains the values of the vector potential with the correct value in the 
        ghost points to have zero current in the surrounding superconductor.
        A[j, i, 0] gives the x_component of the A_field at (x_i, y_j) and
        A[j, i, 1] gives the y_component of the A_field at (x_i, y_j)

    """
    #Create the diagonal to take into account boundary conditions
    diag = np.zeros_like(B_and_bc)
    diag[0] = 1
    diag[-1] = 1
    diag[:,0] = 1
    diag[:,-1] = 1
    
    #Create the upper and lower diagonals and fil them with values according to the finite difference scheme
    #This corresponds the the points with x+dx and x-dx when looking at a points (x,y)
    right = np.zeros_like(diag)
    right[1:-1,1:-1, 0] = 1/dx
    right[1:-1,1:-1, 1] = (1/(2*dx))

    left = np.zeros_like(diag)
    left[1:-1,1:-1, 0] = -1/dx
    left[1:-1,1:-1, 1] = -(1/(2*dx))
    #Create the diagonals corresponding to y+dy and y-dy when looking at a points (x,y)
    #and fill them with values according to the finite difference scheme
    #Due to the different offset for the x and y components we need two different arrays for each
    over_x = np.zeros_like(diag)
    over_y = np.zeros_like(diag)
    over_y[1:-1,1:-1, 0] = 1/dy
    over_x[1:-1,1:-1, 1] = -(1/(2*dy))
    
    
    under_x = np.zeros_like(diag)
    under_y = np.zeros_like(diag)
    under_y[1:-1,1:-1, 0] = -1/dy
    under_x[1:-1,1:-1, 1] = (1/(2*dy))
    
    #If the number of points Nx and Ny are odd the middle point should be 0 by symmetry
    if Nx%2==1 and Ny%2==1:
        right[Ny//2+1, Nx//2+1] = 0
        left[Ny//2+1, Nx//2+1] = 0
        over_x[Ny//2+1, Nx//2+1] = 0
        over_y[Ny//2+1, Nx//2+1] = 0
        under_x[Ny//2+1, Nx//2+1] = 0
        under_y[Ny//2+1, Nx//2+1] = 0
        diag[Nx//2+1, Ny//2+1] = 1
        B_and_bc[Nx//2+1, Ny//2+1] = 0
      
    #Reshape all arrays to be one dimensional and fix them to be the proper length
    #The arrays to the rights ot the diagonal remove values from the end of the array while
    #arrays to the left remove from the start. This is since arrays to the right represent
    #points later in the flattened grid while the ones to the left refer to points earlier
    diag = np.reshape(diag, diag.shape[0]*diag.shape[1]*diag.shape[2])
    right= np.reshape(right, diag.shape[0])[:-2]
    left  = np.reshape(left, diag.shape[0])[2:]
    over_x  = np.reshape(over_x, diag.shape[0])[:-2*(Nx+2)+1]
    over_y  = np.reshape(over_y, diag.shape[0])[:-2*(Nx+2)-1]
    under_x  = np.reshape(under_x, diag.shape[0])[2*(Nx+2)+1:]
    under_y  = np.reshape(under_y, diag.shape[0])[2*(Nx+2)-1:]
    
    #Create a matrix containing all linear equations
    difference_matrix = sparse.diags((diag,right, left, over_x, over_y, under_x, under_y), 
                                     [0, 2, -2, 2*(Nx+2)-1, 2*(Nx+2)+1,-2*(Nx+2)-1, -2*(Nx+2)+1]).tocsc()
    
    #Create a vector with the rhs of the equation
    g = np.reshape(B_and_bc, diag.shape[0])

    #Solve the matrix equation to find the vector potential
    A = np.reshape(splinalg.spsolve(difference_matrix, g, use_umfpack = True), (Ny+2, Nx+2,2))
    return A