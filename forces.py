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

def funcmag(X, t):
    pos = np.array([X[0], X[1], X[2]])
    BdotGradB = find_BdotGradB(pos)
    xval = (1/Vt)*(Vb*chi/(muw*muo*6*np.pi*Rc))*(1/L)*BdotGradB[0]
    yval = (1/Vt)*(Vb*chi/(muw*muo*6*np.pi*Rc))*(1/L)*BdotGradB[1]
    zval = -1 + (1/Vt)*(Vb*chi/(muw*muo*6*np.pi*Rc))*(1/L)*BdotGradB[2]
    return [xval, yval, zval]


def run(X0):
    X = odeint(func, X0, t)
    Xmag = odeint(funcmag,X0,t)
    #print(Xmag[:,2])
    return X, Xmag




if __name__ == '__main__':
    for i in range(0, theta.size):
        theta[i] = i*2*np.pi/N


   #fig = plt.subplots(figsize=(20, 16), dpi=600)

    #plt.plot(t,X[:,2],'r-', lw=5, label='No Magnetic Field')
    #plt.plot(t,Xmag[Xmag[:,2]>2*p*n,2],'b-', lw=5, label='Magnetic Field')
    #plt.axhline(y=2, color='k', linestyle='-',lw=5, label='Bottom of Dish')
    #plt.xlabel('t/T', fontsize=30)
    #plt.ylabel('Z/L', fontsize=30)
    #plt.tick_params(axis='both', which='major', labelsize=30)
    #plt.legend(['No Magnetic Field', 'Magnetic Field', 'Bottom of Dish'], fontsize=30)
    #plt.savefig('diff.png', transparent=True,
    #            bbox_inches='tight', pad_inches=0)
    #plt.savefig('diff.jpg', bbox_inches='tight', pad_inches=0)
    #plt.show()


    fig = plt.figure()#figsize=(20, 16), dpi=600, facecolor='w', edgecolor='k')
    ax = fig.gca(projection='3d')

    X0 = [0, 0, 2*p*n+1]
    X, Xmag = run(X0)
    ax.plot(X[:,0],X[:,1],X[:,2], label='test',LineWidth=3)

    for i in tqdm(range(0, 10)):
        thetaval = i*2*np.pi/10
        X0= [np.cos(thetaval), np.sin(thetaval), 2*p*n+1]
        X, Xmag = run(X0)

        ax.plot(Xmag[Xmag[:,2]>2*p*n,0],Xmag[Xmag[:,2]>2*p*n,1] ,Xmag[Xmag[:,2]>2*p*n,2], label='wire', LineWidth=5)

    ax.set_xlabel('\n' + 'X axis')#, fontsize=30, linespacing=4)
    ax.set_ylabel('\n' + 'Y axis')#, fontsize=30, linespacing=4)
    ax.set_zlabel('\n' + 'Z axis')#, fontsize=30, linespacing=4)
    #ax.xaxis._axinfo['label']['space_factor'] = 100
    #plt.legend(['Magnetic Field', 'No Magnetic Field'], fontsize=30)

    #plt.tick_params(axis='both', which='major', labelsize=30)
    #plt.savefig('traj_test.png', transparent=True,
    #            bbox_inches='tight', pad_inches=0)
    plt.show()
