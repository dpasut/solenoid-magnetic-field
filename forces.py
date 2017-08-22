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

# Solenoid
gs = 30  # Grid spacing
R = 0.5  # Radius of loop (mm)
wr = 0.1  # Radius of wire (mm)
p = 0.1  # Pitch of wire, centre-to-centre (mm)
N = 100  # Number of segments in single loop of wire
n = int(sys.argv[1])#1  # Number of loops of wire
theta = np.empty(n*N)
mu = 1  # Magnetic susceptibility
I = 1  # Current
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
h = (ymax - ymin)/gs

# Cell/beads
Rb = 1
Rc = 1
muo = 1
muw = 1
chi = 1
rhob = 1
rhoc = 1
rhow = 1
g = 9.81

MagConst = (4*np.pi*Rb**3*chi/(3*muo))
ViscConst = 6*np.pi*muw*Rc
m = (4/3)*np.pi*Rb**3*rhob + (4/3)*np.pi*(Rc**3-Rb**3)*rhoc

# Book keeping
maxcount = 10000
position = np.zeros([maxcount, 3])
position[0] = [xmax, ymax, zmax]
BdotGradB = np.zeros([maxcount, 3])
disp = np.zeros(3)
V = np.zeros(3)
tstep = 0.1



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


def find_pos(xforce, yforce, zforce, pos, t):
    accx = xforce/m
    accy = yforce/m
    accz = zforce/m

    disp[0] = V[0]*t + (1/2)*accx*t**2
    disp[1] = V[1]*t + (1/2)*accy*t**2
    disp[2] = V[2]*t + (1/2)*accz*t**2

    if LA.norm(disp) < 1e-4:
        running = False
    else:
        running = True

    newposx = pos[0] + disp[0]
    newposy = pos[1] + disp[1]
    newposz = pos[2] + disp[2]

    return newposx, newposy, newposz, running


def run():
    count = 0
    while True:
        t = count * tstep
        pos = position[count]
        Bx, By, Bz = find_B(pos, theta, R, N, wr)
        Bx_right, By_right, Bz_right = find_B(pos, theta, R, N, wr)
        Bx_left, By_left, Bz_left = find_B(pos, theta, R, N, wr)
        Bx_up, By_up, Bz_up = find_B(pos, theta, R, N, wr)
        Bx_down, By_down, Bz_down = find_B(pos, theta, R, N, wr)

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

        BdotGradB[count, 0] = Bx*bxx + By*bxy + Bz*bxz
        BdotGradB[count, 1] = Bx*byx + By*byy + Bz*byz
        BdotGradB[count, 2] = Bx*bzx + By*bzy + Bz*bzz

        if count > 0:
            V[0] = (position[count, 0] - position[count -1, 0])/tstep
            V[1] = (position[count, 1] - position[count -1, 1])/tstep
            V[2] = (position[count, 2] - position[count -1, 2])/tstep
        else:
            V[0] = 0
            V[1] = 0
            V[2] = 0

        Forcex = MagConst*BdotGradB[count, 0] + ViscConst*V[0]
        Forcey = MagConst*BdotGradB[count, 1] + ViscConst*V[1]
        Forcez = -m*g + (4/3)*np.pi*Rc**3*rhow*g + MagConst*BdotGradB[count, 0] + ViscConst*V[2]

        print(Forcex, Forcey, Forcez)

        position[count + 1, 0], position[count + 1, 1], position[count + 1, 2], running= find_pos(Forcex, Forcey, Forcez, pos, t)

        print(count)
        #print(position[count])

        if count == maxcount:
            print("count exceded")
            break
        elif position[count + 1, 2] <= 2*p*n:
            print('position too low')
            break
        elif running == False:
            print('not moving')
            break
        else:
            count += 1

if __name__ == '__main__':
    run()
