#!/usr/local/bin/python3.7

import numpy as np
from numpy.linalg import solve as sol
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d

plt.rcParams['figure.figsize'] = 8, 5
plt.rcParams['axes.grid'] = True
plt.rcParams.update({'font.size':10})

def Z(U):
#    return 0.25 * (U[0] - 2) ** 2 + (U[1] - 1) ** 2 
    return 2 * (np.exp(-U[0]**2 - U[1]**2) - np.exp(-(U[0] - 1)**2 - (U[1] - 1)**2)) 

def grad(U, h):
    a = (Z(U + h * np.array([1, 0])) - Z(U)) / h
    b = (Z(U + h * np.array([0, 1])) - Z(U)) / h
    return np.array([a, b]) 

abs0, ord0 = -1.99, 3.01
absg, absd, ordg, ordd, reso = -2.5, 3.5, -2.5, 3.5, 100
X = np.array([abs0, ord0])
eps = 0.0001
step = 0.33

Y = np.array([abs0, ord0])
for i in range(50001):
    if i % 1000 == 0 :
        print(i,'#', '(%.4f, %.4f) --> %.8f' % (X[0], X[1], Z(X)))
    X -= step * grad(X, eps)
    Y = np.vstack((Y,X))

plt.figure()
w = np.ogrid[absg:absd:reso*1j, ordg:ordd:reso*1j]
lev = np.linspace(-1, 1, 11)
#lev = np.logspace(-2., 0., 5)
cont = plt.contour(w[:][0].flatten(), w[:][1].flatten(), Z(w), levels=lev)
plt.clabel(cont)
plt.axis('scaled')
plt.plot(Y[:,0], Y[:,1], 'r-.') ; plt.savefig('Newt1.pdf') ; plt.show()

w0, w1 = np.meshgrid(np.linspace(absg,absd,reso), np.linspace(ordg,ordd,reso))
w = np.array([w0, w1])
ax = plt.axes(projection='3d')
ax.plot_surface(w0, w1,  Z(w), alpha=.6, cmap = 'viridis') 
ax.plot3D(Y[:,0], Y[:,1], Z(Y.T), 'r-.') ; plt.savefig('Newt2.pdf') ; plt.show()
#ax.plot_wireframe(w[:][0].flatten(), w[:][1].flatten(),  Z(w)) ; plt.show()


'''
def Z(U):
    return (U[0] - 1) ** 2 + 0.25 * (U[1] - 2) ** 2 

def grad(U):
    a = 2 * (U[0] - 1)
    b = 0.5 * (U[1] - 2)
    return np.array([a, b])

X = np.array([0., 0.])
step = 0.1

for i in range(101):
    if i % 10 == 0 :
        print(i,'#', '(%.2f, %.2f) --> %.4f' % (X[0], X[1], Z(X)))
    X -= step * grad(X)
'''


'''
def Z(U):
    a = (U[0] - 1) ** 2 + 0.25 * (U[1] - 2) ** 2 
    b = (U[0] - 1) ** 2 + 0.5 * (U[1] - 2) ** 2 
    return np.array([a, b])

def Jac(U):
    a = 2 * (U[0] - 1)
    b = 0.5 * (U[1] - 2)
    c = 2 * (U[0] - 1)
    d = (U[1] - 1) 
    return np.array([[a, b], [c, d]])

X = np.array([-1., -1.])

for i in range(51):
    print(i, '#', '(%.2f, %.2f) --> (%.2f, %.2f)' \
          % (X[0], X[1], Z(X)[0], Z(X)[1]) ) 
    Y = sol(Jac(X), -Z(X))
    X += Y  
'''
