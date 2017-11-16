#!/usr/bin/env python
# Author: Daniel Pasut <daniel.pasut@uoit.ca>

import numpy as np
#import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import linalg as LA
from pylab import *
from tqdm import tqdm
import sys

gs = 30  # Grid spacing
R = 0.5  # Radius of loop (mm)
wr = 0.1  # Radius of wire (mm)
p = 0.1  # Pitch of wire, centre-to-centre (mm)
N = 100  # Number of segments in single loop of wire
n = int(sys.argv[1])#1  # Number of loops of wire
theta = np.empty(n*N)
mu = 1  # Magnetic susceptibility
I = -1  # Current
C = mu*I/(4*np.pi)
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
# x's are all zero, looking at plane
Bx = np.zeros([gs, gs])  # x components don't change
By = np.zeros([gs, gs])  # y components of field matrix
Bz = np.zeros([gs, gs])  # z components of field matrix
norms = np.zeros([gs, gs])  # matrix for norms at each point

Fx = np.zeros([gs, gs])
Fy = np.zeros([gs, gs])
Fz = np.zeros([gs, gs])

values = np.zeros([4,gs])
xvals = np.arange(gs)
insidez = 0.


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
        if LA.norm(r) <= 1.35*wr:
            inwire = np.array([0, 0, 0])
            return inwire
        else:
            cross += C * np.cross(dl, r) / LA.norm(r)**3
    return cross


# Plot the solenoid in 3-D
def plot_solenoid():
    wire = np.array([R*np.cos(theta), R*np.sin(theta), p*theta/np.pi])

    fig = plt.figure(figsize=(20, 16), dpi=600, facecolor='w', edgecolor='k')
    ax = fig.gca(projection='3d')
    ax.plot(wire[0], wire[1], wire[2], label='wire', LineWidth=7)

    ax.set_xlabel('\n' + 'X axis', fontsize=30, linespacing=4)
    ax.set_ylabel('\n' + 'Y axis', fontsize=30, linespacing=4)
    ax.set_zlabel('\n' + 'Z axis', fontsize=30, linespacing=4)
    ax.xaxis._axinfo['label']['space_factor'] = 100

    plt.tick_params(axis='both', which='major', labelsize=30)
    plt.savefig('wire-loop.png', transparent=True,
                bbox_inches='tight', pad_inches=0)
    #plt.show()


# Calculate the magnetic field and find norms
def find_field():
    for j in tqdm(range(y.size)):
        for k in range(z.size):
            pos = np.array([0, y[j], z[k]])
            Bx[j, k], By[j, k], Bz[j, k] = find_B(pos, theta, R, N, wr)
            norms[j, k] = LA.norm([Bx[j, k], By[j, k], Bz[j, k]])
            if k == z.size/2:
                values[0, j] = Bx[j, k]
                values[1, j] = By[j, k]
                values[2, j] = Bz[j, k]
                values[3, j] = abs(y[j]) - R
                if j == y.size/2:
                    insidez = Bz[j, k]
    return insidez


# Calculate Forces
def find_BdotGradB():
    for j in tqdm(range(1, y.size - 1)):
        for k in range(1, z.size - 1):
            bxy = (Bx[j+1, k] - Bx[j-1, k]) / 2*h
            byy = (By[j+1, k] - By[j-1, k]) / 2*h
            bzy = (Bz[j+1, k] - Bz[j-1, k]) / 2*h

            bxz = (Bx[j, k+1] - Bx[j, k-1]) / 2*h
            byz = (By[j, k+1] - By[j, k-1]) / 2*h
            bzz = (Bz[j, k+1] - Bz[j, k-1]) / 2*h

            # X derivatives calculated by divergence and curl of B
            bxx = -byy - bzz
            byx =  bxy
            bzx =  - bxz

            Fx[j, k] = Bx[j,k]*bxx + By[j,k]*bxy + Bz[j,k]*bxz
            Fy[j, k] = Bx[j,k]*byx + By[j,k]*byy + Bz[j,k]*byz
            Fz[j, k] = Bx[j,k]*bzx + By[j,k]*bzy + Bz[j,k]*bzz

            pos = np.array([0, y[j], z[k]])

            for q in range(1, theta.size):
                rs = np.array([R*np.cos(theta[q]-np.pi/N),
                       R*np.sin(theta[q]-np.pi/N),
                       (p*(theta[q]-np.pi/N))/np.pi])
                r = pos - rs

                if LA.norm(r) <= 1.25*wr:
                    Fx[j, k] = 0
                    Fy[j, k] = 0
                    Fz[j, k] = 0



# Plot quiver diagram
def plot_field():
    print(By[29])
    with open('field.dat','a+') as f:
        f.write("y z u v\n")
        for j in range(size(y)):
            for k in range(size(z)):
                f.write("%.4f %.4f %.4f %.4f\n" % (y[j], z[k], By[j,k], Bz[j,k]))


    fig, ax = plt.subplots(figsize=(20, 16), dpi=600)

    for i in range(n):
        circ = plt.Circle((R*np.sin(np.pi/2), (4*i+1)*p/2), radius=wr,
                          color='k', alpha=0.5)
        ax.add_patch(circ)
        circ = plt.Circle((R*np.sin(3*np.pi/2), (4*i+3)*p/2), radius=wr,
                          color='k', alpha=0.5)
        ax.add_patch(circ)
        plt.plot(R*np.sin(np.pi/2), (4*i+1)*p/2, '*k', R*np.sin(3*np.pi/2),
                 (4*i+3)*p/2, 'ok')
    ax.quiver(Y, Z, By, Bz)
    ax.set_xlim((ymin, ymax))  # set the xlim to xmin, xmax
    ax.set_ylim((zmin, zmax))

    ax.spines['bottom'].set_color('k')
    ax.spines['top'].set_color('white')
    #ax.set(aspect=1, title='Quiver Plot - field lines')
    plt.xlabel('Y axis', fontsize=30)
    plt.ylabel('Z axis', fontsize=30)
    plt.tick_params(axis='both', which='major', labelsize=30)
    plt.savefig('field-loop-test.png', transparent=True,
                bbox_inches='tight', pad_inches=0)
    plt.savefig('field-loop-test.jpg', bbox_inches='tight', pad_inches=0)
    plt.show()

# Plot quiver diagram
def plot_forces():
    fig, ax = plt.subplots(figsize=(20, 16), dpi=600)

 #   for i in range(n):
 #       circ = plt.Circle((R*np.sin(np.pi/2), (4*i+1)*p/2), radius=wr,
 #                         color='k', alpha=0.5)
 #       ax.add_patch(circ)
 #       circ = plt.Circle((R*np.sin(3*np.pi/2), (4*i+3)*p/2), radius=wr,
 #                         color='k', alpha=0.5)
 #       ax.add_patch(circ)
 #       plt.plot(R*np.sin(np.pi/2), (4*i+1)*p/2, 'ok', R*np.sin(3*np.pi/2),
 #                (4*i+3)*p/2, '*k')

    ax.quiver(Y, Z, Fy, Fz)
    ax.set_xlim((ymin +0.5, ymax -0.5))  # set the xlim to xmin, xmax
    ax.set_ylim((2*p*n, zmax))
    ax.spines['bottom'].set_color('k')
    ax.spines['top'].set_color('white')
    plt.xlabel('Y axis', fontsize=30)
    plt.ylabel('Z axis', fontsize=30)
    plt.tick_params(axis='both', which='major', labelsize=30)
    ax.set(aspect=1)
    plt.savefig('forces-loop.png', transparent=True,
                bbox_inches='tight', pad_inches=0)
    plt.savefig('forces-loop.jpg', bbox_inches='tight', pad_inches=0)
    #plt.show()

if __name__ == '__main__':
    for i in range(0, theta.size):
        theta[i] = i*2*np.pi/N

    #plot_solenoid()

    insidez = find_field()
    find_BdotGradB()

    plot_field()
    #plot_forces()
    #plt.figure()
    #plt.plot(xvals[18:gs], values[0, 18:gs],'b-',label='x component')
    #plt.plot(xvals[18:gs], values[0, 18:gs]*values[3, 18:gs],'g-',label='x component times distance')
    #plt.plot(xvals[18:gs], values[1, 18:gs],'g-', label='y component')
    #plt.plot(xvals, values[2],'r-')
    #plt.legend(['x component of magnetic field', 'x component times distance to wire'])
    #plt.savefig('test2.jpg', bbox_inches='tight', pad_inches=0)
    #plt.show()

    #with open('values.txt', 'a') as f:
    #    f.write("%i %5.4f %5.4f\n" %(n, min(values[0, 18:gs]*values[3, 18:gs]), insidez))
    #print(n, min(values[0, 18:gs]*values[3, 18:gs]), insidez)

    #with open('values.txt', 'r') as g:
    #    nvals, xcompXdist, bzvals = np.loadtxt('values.txt', delimiter=' ', unpack=True)

    #plt.figure()
    #plt.plot(nvals, xcompXdist,'b-',label='x component')
    #plt.plot(nvals, bzvals/nvals, 'r-')
    #plt.legend(['minimum x component times distance to wire'])
    #plt.title('x component times distance to wire for various number of loops')
    #plt.xlabel('Number of loops')
    #plt.ylabel('Magnitude of Magnetic Field')
   #plt.savefig('xcomp_times_dist.jpg', bbox_inches='tight', pad_inches=0)
    #plt.show()
#    print(xvals[19:gs], values[0, 19:gs], values[3,19:gs])
    #plt.figure()
    #plt.plot(xvals[19:gs], np.log(abs(values[0, 19:gs])) - np.log(abs(values[3, 19:gs])))
    #plt.plot(np.log(abs(values[0, 19:gs])), np.log(abs(values[3, 19:gs])))
    #plt.savefig('loglog.jpg', bbox_inches='tight', pad_inches=0)
    #plt.show()

    #plt.figure()
    #plt.plot(xvals[19:gs], np.log(abs(values[0, 19:gs])) + np.log(abs(values[3, 19:gs])))
    #plt.plot(np.log(abs(values[0, 19:gs])), np.log(abs(values[3, 19:gs])))
    #plt.savefig('log.jpg', bbox_inches='tight', pad_inches=0)
    #plt.show()
