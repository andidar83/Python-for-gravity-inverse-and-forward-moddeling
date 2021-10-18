# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 11:01:22 2020

@author: Andidar83
"""

import math
import cmath
import time
import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.interpolate as intr
import matplotlib.cm as cm

data = np.loadtxt('gravity_anomaly.txt', skiprows=1)

X =data[:,0]
Y = data[:,1]
obs = data[:,2]
shape1 = (169,1)
x_grid = np.linspace(np.min(X), np.max(X), 50)
y_grid = np.linspace(np.min(Y),np.max(Y), 50)
Xi, Yi = np.meshgrid(x_grid, y_grid)

#membuat data vertikal
obsarray=np.array(obs)
d = obsarray.reshape(shape1)

#membuat matrix karnel

#Kolom 1 dngan nilai 1
satu =  [1]*169
satuarray=np.array(satu)
K1 = satuarray.reshape(shape1)

#kolom2
Xarray=np.array(X)
K2 = Xarray.reshape(shape1)
#kolom3
Yarray=np.array(Y)
K3 = Yarray.reshape(shape1)
#kolom4
K4 = K2 * K3
#kolom5
K5 = (K2)**2
#kolom6
K6 = (K3)**2
#Kolom 7
K7 = (K2)**3
#Kolom 8
K8 = ((K2)**2*(K3))
#Kolom 9
K9 = ((K2)*(K3)**2)
#Kolom 10
K10 =(K3)**3

G = np.concatenate((K1, K2, K3, K4, K5, K6), axis=1)
G2 = np.concatenate((K1,K2,K3,K4,K5,K6,K7,K8,K9,K10), axis=1)


#INVERSI
def inverseUD(data,Gx):
    Gt = Gx.transpose() # G transpose
    Gi = np.linalg.pinv(np.dot(Gx,Gt)) # [GGt+dampingI]-1
    m = np.dot(np.dot(Gt,Gi),data) # Gt*[GGt+dampingI]-1* d
    return m

Kpolinom = inverseUD(d,G)
Kpolinom2 = inverseUD(d,G2)

def polinom2(xi,yi):
    gregi2 = Kpolinom[0,:] + Kpolinom[1,:]*xi + Kpolinom[2,:]*yi + Kpolinom[3,:]*yi*xi + Kpolinom[4,:]*(xi)**2 + Kpolinom[5,:]*(yi)**2  
    return gregi2


def polinom3(xi2,yi2):
    gregi3 = Kpolinom2[0,:] + Kpolinom2[1,:]*xi2 + Kpolinom2[2,:]*yi2 + Kpolinom2[3,:]*yi2*xi2 + Kpolinom2[4,:]*(xi2)**2 + Kpolinom2[5,:]*(yi2)**2 + Kpolinom2[6,:]*xi2**3 + Kpolinom2[7,:]*(xi2)**2*(yi2) + Kpolinom2[8,:]*(xi2)*(yi2)**2 + Kpolinom2[9,:]*(yi2)**3
    return gregi3


Greg2 = polinom2(X, Y)
Greg3 = polinom3(X, Y)


#Anomali Bouger Default
obsarray=np.array(obs)
interBAD=intr.Rbf(X,Y,obsarray, smooth = 0.2)
BougerAD = interBAD(Xi,Yi)

plt.figure()

plt.contourf(Xi,Yi,BougerAD, 50, cmap=cm.jet)
cb1 = plt.colorbar()
cb1.set_label('mgal', labelpad=-40, y=1.05, rotation=0)
plt.contour(Xi, Yi, BougerAD, colors='black', linewidths=0.3, levels = 50)
plt.title('Kontur Anomali Bouger')

#Anomali Regional Orde 2
interBA=intr.Rbf(X,Y,Greg2, smooth = 0.2)
BougerA2 = interBA(Xi,Yi)

plt.figure()
plt.contourf(Xi,Yi,BougerA2, 50, cmap=cm.jet)
cb2 = plt.colorbar()
cb2.set_label('mgal', labelpad=-40, y=1.05, rotation=0)
plt.contour(Xi, Yi, BougerA2, colors='black', linewidths=1, levels = 50)
plt.title('Kontur TSA Orde 2')

#Anomali Regional Orde 3
interBA3=intr.Rbf(X,Y,Greg3, smooth = 0.2)
BougerA3 = interBA3(Xi,Yi)

plt.figure()
plt.contourf(Xi,Yi,BougerA3, 50, cmap=cm.jet)
cb3 = plt.colorbar()
cb3.set_label('mgal', labelpad=-40, y=1.05, rotation=0)
plt.contour(Xi, Yi, BougerA3, colors='black', linewidths=1, levels = 50)
plt.title('Kontur TSA Orde 3')


print(Kpolinom2)

"""
#NO 2
t = ([[43.21,43.30,37.35,35.55,41.35,42.24, 42.23, 38.50, 35.60, 40.00]])
d = np.transpose(t)


Gx = np.array([[1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              [0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1],
              [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0],
              [0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0],
              [0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0],
              [0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0],
              [0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]])

G= Gx*20


def inverseUD(di,Gx):
    Gt = Gx.transpose() # G transpose
    Gi = np.linalg.pinv(np.dot(Gt,Gx)) # [GtG]-1
    m = np.dot(np.dot(Gi,Gt),di) # [GtG]-1*Gt* d
    return m   

Si = inverseUD(d,G)
Sr = Si.reshape(5,5)
S = Sr[::-1]
print(S)

x_grid1 = np.linspace(0,100,6)
y_grid1 = np.linspace(0,100,6)
Xp, Yp = np.meshgrid(x_grid1, y_grid1)


plt.figure(figsize=(5,5))
plt.pcolor(Xp, Yp, S,cmap = cm.jet)
plt.xticks(np.linspace(0,100,6))
plt.yticks(np.linspace(0,100,6))
clb = plt.colorbar()
clb.set_label('Sm', labelpad=-40, y=1.05, rotation=0)
plt.title('Penamang Horizontal GPR')
plt.show()

"""







