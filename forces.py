#!/usr/bin/env python
# Author: Daniel Pasut <daniel.pasut@uoit.ca>

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import linalg as LA
from pylab import *
from tqdm import tqdm
import sys
from scipy.integrate import odeint
from scipy.integrate import ode

# Solenoid
gs = 30  # Grid spacing
R = 0.5  # Radius of loop (mm)
wr = 0.1  # Radius of wire (mm)
p = 0.1  # Pitch of wire, centre-to-centre (mm)
N = 100  # Number of segments in single loop of wire
n = int(sys.argv[1])#1  # Number of loops of wire
theta = np.empty(n*N)
mu = 1  # Magnetic susceptibility
I = 10  # Current
C = mu*I/(4*np.pi)

# Geometry
xmin = -2.1
xmax = 2.1
ymin = -2.1
ymax = 2.1
zmin = -1.1
zmax =  p*n*2+1.1
x = np.linspace(xmin, xmax, gs)  # Positions for x
y = np.linspace(ymin, ymax, gs)  # Positions for y
z = np.linspace(zmin, zmax, gs)  # Positions for z
Y, Z = np.meshgrid(y, z, indexing='ij')  # Grid for y/z

# Cell/beads
Rb = 3.5e-6
Rc = 3.5e-5
muo = np.pi*4e-7
muw = 10e-3
chi = 0.17
rhob = 1.5e3
rhoc = 1020
rhow = 1000
g = 9.81

MagConst = (4*np.pi*Rb**3*chi/(3*muo))
ViscConst = 6*np.pi*muw*Rc
m = (4/3)*np.pi*Rb**3*rhob + (4/3)*np.pi*(Rc**3-Rb**3)*rhoc
Vc = (4/3)*np.pi*Rc**3
Vb = (4/3)*np.pi*Rb**3

# Book keeping
BdotGradB = np.zeros([1, 3])
V = np.zeros(3)
tstep = 0.01
t0 = 0
t1 = 1
t = np.arange(t0, t1, tstep)

Vt = g*(rhoc-rhow)*Vc/(6*np.pi*muw*Rc) #terminal velocity
L = 1e-2

# Function to do summation over all segments of wire
def find_B(pos, theta, R, N, wr):
    cross = 0
    for k in range(1, theta.size):
        rs = np.array([R*np.cos(theta[k]-np.pi/N),
                       R*np.sin(theta[k]-np.pi/N),
                       (p*(theta[k]-np.pi/N))/np.pi])
        r = pos - rs
        dl = np.array([R*(np.cos(theta[k])-np.cos(theta[k-1])),
                       R*(np.sin(theta[k])-np.sin(theta[k-1])),
                       p/N])
        cross += C * np.cross(dl, r) / LA.norm(r)**3
    return cross



def find_BdotGradB(pos):
    h = np.pi*R/N

    Bx, By, Bz = find_B(pos, theta, R, N, wr)

    Bx_right, By_right, Bz_right = find_B(pos + [0,h,0], theta, R, N, wr)
    Bx_left, By_left, Bz_left = find_B(pos - [0,h,0], theta, R, N, wr)
    Bx_up, By_up, Bz_up = find_B(pos + [0,0,h], theta, R, N, wr)
    Bx_down, By_down, Bz_down = find_B(pos - [0,0,h], theta, R, N, wr)

    bxy = (Bx_right - Bx_left) / 2*h
    byy = (By_right - By_left) / 2*h
    bzy = (Bz_right - Bz_left) / 2*h

    bxz = (Bx_up - Bx_down) / 2*h
    byz = (By_up - By_down) / 2*h
    bzz = (Bz_up - Bz_down) / 2*h

    # X derivatives calculated by divergence and curl of B
    bxx = -byy - bzz
    byx =  bxy
    bzx =  - bxz
    return [Bx*bxx + By*bxy + Bz*bxz, Bx*byx + By*byy + Bz*byz, Bx*bzx + By*bzy + Bz*bzz]


def func(X, t):
    xval = 0
    yval = 0
    zval = -1
    return [xval, yval, zval]
if __name__ == '__main__':
    run()
