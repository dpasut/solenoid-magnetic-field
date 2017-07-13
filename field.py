# Author: Daniel Pasut <daniel.pasut@uoit.ca>

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import linalg as LA
from pylab import *


R = 1 # Radius of loop (mm)
wr = 0 # Radius of wire (mm)
p = 0.1 # Pitch of wire, centre-to-centre (mm)
N = 100 # Number of segments in single loop of wire
n = 5 # Number of loops of wire
theta = np.empty(n*N)
mu = 1 # Magnetic suceptibility
I = 10 # Current
C = mu*I/(4*np.pi)

x = np.linspace(-2,2,30) # Positions for x
y = np.linspace(-2,2,30) # Positions for y
z = np.linspace(-2,p*n*2+2,30) # Positions for z
Y, Z = np.mgrid[-2:2:30j,-2:p*n*2+2:30j] # Grid for y/z

# x's are all zero, looking at plane
Bx = np.zeros([y.size, z.size]) # x components don't change
By = np.zeros([y.size, z.size]) # y components of field matrix
Bz = np.zeros([y.size, z.size]) # z components of field matrix
norms = np.zeros([y.size,z.size]) # matrix for norms at each point


# Function to do summation over all
def find_B(x,y,z,theta,R,N):
    cross = 0
    for k in range(1,theta.size):

        dl = np.array([R*(np.cos(theta[k])-np.cos(theta[k-1])),
                           R*(np.sin(theta[k])-np.sin(theta[k-1])),
                           p/N])
        rs = np.array([R*np.cos(theta[k]-np.pi/N),
                       R*np.sin(theta[k]-np.pi/N),
                       (p*(theta[k]-np.pi/N))/np.pi])
        r = rs - np.array([0, y, z])
        cross += C*np.cross(dl,r)/ LA.norm(r)**3
    return cross



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

for i in range(y.size):
   for j in range(z.size):
       Bx[i,j], By[i,j], Bz[i,j] = find_B(0,y[i],z[j],theta,R,N)
       print(i,j)
       norms[i,j] = LA.norm([Bx[i,j], By[i,j], Bz[i,j]])

fig, ax = plt.subplots()

for i in range(n):
    plt.plot(R*np.sin(np.pi/2),(4*i+1)*p/2,'ok',R*np.sin(3*np.pi/2),(4*i+3)*p/2,'+k')
ax.quiver(Y, Z, By/norms, Bz/norms)
ax.set(aspect=1, title='Quiver Plot - field lines')
plt.savefig('field-loop.svg',transparent=True, bbox_inches='tight', pad_inches=0)
plt.savefig('field-loop.jpg', bbox_inches='tight', pad_inches=0)
plt.show()
