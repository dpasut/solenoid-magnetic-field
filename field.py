#!/usr/bin/env python
# Author: Daniel Pasut <daniel.pasut@uoit.ca>

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import linalg as LA
from pylab import *
from tqdm import tqdm

gs = 30 # Grid spacing
R = 1 # Radius of loop (mm)
wr = 0.1 # Radius of wire (mm)
p = 0.1 # Pitch of wire, centre-to-centre (mm)
N = 100 # Number of segments in single loop of wire
n = 5 # Number of loops of wire
theta = np.empty(n*N)
mu = 1 # Magnetic suceptibility
I = -10 # Current
C = mu*I/(4*np.pi)

x = np.linspace(-2,2,gs) # Positions for x
y = np.linspace(-2.1,2.1,gs) # Positions for y
z = np.linspace(-1.1,p*n*2+1.1,gs) # Positions for z
Y, Z = np.meshgrid(y,z, indexing = 'ij') # Grid for y/z

# x's are all zero, looking at plane
Bx = np.zeros([y.size, z.size]) # x components don't change
By = np.zeros([y.size, z.size]) # y components of field matrix
Bz = np.zeros([y.size, z.size]) # z components of field matrix
norms = np.zeros([y.size,z.size]) # matrix for norms at each point


# Function to do summation over all segments of wire
def find_B(pos,theta,R,N,wr):
    cross = 0
    for k in range(1,theta.size):
        rs = np.array([R*np.cos(theta[k]-np.pi/N),
                       R*np.sin(theta[k]-np.pi/N),
                       (p*(theta[k]-np.pi/N))/np.pi])
        r = rs - pos
        dl = np.array([R*(np.cos(theta[k])-np.cos(theta[k-1])),
                           R*(np.sin(theta[k])-np.sin(theta[k-1])),
                           p/N])
        if LA.norm(r) <= 1.35*wr:
            inwire = np.array([0, 0, 0])
            return inwire
        else:
            cross += C*np.cross(dl,r)/ LA.norm(r)**3
    return cross


# Plot the solenoid in 3-D
for i in range(0,theta.size):
    theta[i] = i*2*np.pi/N
wire = np.array([R*np.cos(theta), R*np.sin(theta),p*theta/np.pi])

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(wire[0],wire[1] , wire[2], label='wire')
ax.legend()
plt.savefig('wire-loop.svg',transparent=True, bbox_inches='tight', pad_inches=0)
plt.savefig('wire-loop.jpg', bbox_inches='tight', pad_inches=0)
plt.show()


# Calculate the magnetic field and find norms
for i in tqdm(range(y.size)):
   for j in range(z.size):
       pos = np.array([0, y[i], z[j]])
       Bx[i,j], By[i,j], Bz[i,j] = find_B(pos, theta, R, N, wr)
       norms[i,j] = LA.norm([Bx[i,j], By[i,j], Bz[i,j]])


# Plot quiver diagram
fig, ax = plt.subplots()

for i in range(n):
    circ = plt.Circle((R*np.sin(np.pi/2),(4*i+1)*p/2), radius=wr, color='k', alpha=0.5)
    ax.add_patch(circ)
    circ = plt.Circle((R*np.sin(3*np.pi/2),(4*i+3)*p/2), radius=wr, color='k', alpha=0.5)
    ax.add_patch(circ)
    plt.plot(R*np.sin(np.pi/2),(4*i+1)*p/2,'ok',R*np.sin(3*np.pi/2),(4*i+3)*p/2,'*k')

ax.quiver(Y, Z, By/norms, Bz/norms)
xlim( (Y.min(), Y.max()) )  # set the xlim to xmin, xmax
ylim( (Z.min(), Z.max()) )
ax.set(aspect=1, title='Quiver Plot - field lines')
plt.savefig('field-loop.svg',transparent=True, bbox_inches='tight', pad_inches=0)
plt.savefig('field-loop.jpg', bbox_inches='tight', pad_inches=0)
plt.show()
