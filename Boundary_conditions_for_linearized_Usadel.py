# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 10:41:59 2023

@author: Havard
"""
import numpy as np

def g_bc_SN(epsilon, theta, Nx, Ny, gamma = None):
    g=np.zeros((Ny+2, Nx+2), dtype = complex)
    g[1:Ny+1,0] = -np.sinh(np.arctanh(1/epsilon))
    g = np.reshape(g,(Nx+2)*(Ny+2))
    return g


def g_bc_SNS(epsilon, theta, Nx, Ny, gamma = None):
    g=np.zeros((Ny+2, Nx+2), dtype = complex)
    g[1:Ny+1,0] = -np.sinh(np.arctanh(1/epsilon))* np.exp(1j*theta/2)/3
    g[1:Ny+1,-1] = -np.sinh(np.arctanh(1/epsilon)) * np.exp(-1j*theta/2)/3

    g = np.reshape(g,(Nx+2)*(Ny+2))
    return g


def g_bc_SSNSS(epsilon, theta, Nx, Ny, use_kl = False, gamma = 3):
    """Boundary condition for a normal metal with a superconductor on all 4 sides with a continously varying phase
    """
    g=np.zeros((Ny+2, Nx+2), dtype = complex)
    phase = np.exp(1j*(np.linspace(0,theta,(2*Ny)+(2*Nx), endpoint = False)))
    if use_kl:
        g[0,1:Nx+1] = np.sinh(np.arctanh(1/epsilon)) *phase[0:Nx]
        g[1:Ny+1,-1] = -np.sinh(np.arctanh(1/epsilon)) *phase[Nx:Ny+Nx]
        g[-1,1:Nx+1] = -np.sinh(np.arctanh(1/epsilon))*np.flip(phase[Ny+Nx:Ny+(2*Nx)])
        g[1:Ny+1,0] = np.sinh(np.arctanh(1/epsilon))*np.flip(phase[Ny+(2*Nx):(2*Ny)+(2*Nx)])
        g/=gamma
    else:
        g[0,1:Nx+1] = -np.sinh(np.arctanh(1/epsilon)) *phase[0:Nx]
        g[1:Ny+1,-1] = -np.sinh(np.arctanh(1/epsilon)) *phase[Nx:Ny+Nx]
        g[-1,1:Nx+1] = -np.sinh(np.arctanh(1/epsilon))*np.flip(phase[Ny+Nx:Ny+(2*Nx)])
        g[1:Ny+1,0] = -np.sinh(np.arctanh(1/epsilon))*np.flip(phase[Ny+(2*Nx):(2*Ny)+(2*Nx)])
    g = np.reshape(g,(Nx+2)*(Ny+2))
    
    return g