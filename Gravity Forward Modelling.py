# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 19:46:18 2020

@author: Andidar83
"""

import numpy as np
import math
import matplotlib.pyplot as plt

"""
# Soal : Buatlah perkiraan model bawah permukaan menggunakan beberapa buah bola homogen sebagaimana sehingga didapat perbandingan antara kurva teoritik dengan kurva data gravitasi (observasi) yang diberikan pada file datacontoh.txt.
#Hitung nilai selisih kuadrat totalnya (E). Semakin kecil nilai tersebut berarti semakin baik kecocokannya.Lakukan pengulangan sebanyak yang dibutuhkan hingga kurva teoritik dan kurva data berhimpit
import matplotlib.mlab as ml
import scipy
import random
g= np.loadtxt("gdat.txt")
gdat = np.array(g)
x = np.loadtxt("xi.txt")
xi = np.array(x)

plt.plot(xi,gdat,'o')



G = 6.67*10**-11
pi= math.pi
R1 = 250
rho1 = 250
x1 = 500
z1 = 500

R2 = 400
rho2 = 980
x2 = 2000
z2 = 1000

R3 = 25
rho3 = -250
x3 = 3500
z3 = 300

def gravbola(rad,densi,dis,dep):
    nom = (4/3)*pi*rad**3*dep*densi
    den = ((xi-dis)**2+dep**2)**(3/2)
    grav = G*(nom/den)
    mgal = grav*10**5
    return mgal

g1 = gravbola(R1,rho1,x1,z1)
g2 = gravbola(R2,rho2,x2,z2)
g3 = gravbola(R3,rho3,x3,z3)
gteo = g1+g2+g3

err = []
for i in range(len(gteo)):
    E = (gteo[i]-gdat[i])**2
    err.append(E)
    
erro = sum(err)

plt.plot(xi,gteo,'-')
plt.xlabel("Distence (m)")
plt.ylabel("g (mgal)")
plt.legend(['g observed','g predicted'])
print("Error = ",erro)
"""

# Soal beda lagi, yang ini bikin pengaruh gravitasi dari 3 bola dengan densitas 
#yang berbeda beda, kemudian melakukan inversi untuk memperkirakan densitas, jika
#inversinya berhasil maka densitas hasil inversi akan = densitas yang kita definisikan di awal

xi= np.linspace(0,4000,20)
yi= np.linspace(0,4000,20)
X, Y = np.meshgrid(xi, yi) 
 
#FORWARD MODDELING

G = 6.67*10**-11
pi= math.pi
R1 = 450
rho1 = 1300
x1 = 2700
y1 = 1000
z1 = 500

R2 = 500
rho2 = 3500
x2 = 1000
y2 = 2000
z2 = 1500

R3 = 1000
rho3 = 1000
x3 = 3500
y3 = 3500
z3 = 2000

def gravbola(rad,densi,dis,disy,dep):
    nom = (4/3)*pi*rad**3*dep*densi
    den = ((X-dis)**2+(Y-disy)**2+dep**2)**(3/2)
    grav = G*(nom/den)
    mgal = grav*10**5
    return mgal

g1 = gravbola(R1,rho1,x1,y1,z1)
g2 = gravbola(R2,rho2,x2,y2,z2)
g3 = gravbola(R3,rho3,x3,y3,z3)
gteo = g1+g2+g3
gflat = gteo.flatten() 
gflatarray = np.array(gflat)

shape1 = (400,1)
gteoT = gflatarray.reshape(shape1)

Z = np.reshape(gteo, (-1, 20))

plt.contourf(X,Y,Z)
plt.scatter(X, Y, marker = 'd', s = 20, zorder = 10)

#########################################################################
#INVERSION
# Observed Data
xstation = np.linspace(0,4000,20)
data = np.array(gteo[9])


def gravbola(rad,dis,disy,dep):
    nom = (4/3)*pi*rad**3*dep
    den = ((X-dis)**2+(Y-disy)**2+dep**2)**(3/2)
    grav = G*(nom/den)
    karnel = grav*10**5
    return karnel

G1 = gravbola(R1,x1,y1,z1)
G1flat = G1.flatten()
G1flatarray = np.array(G1flat)
G1T = G1flatarray.reshape(shape1)


G2 = gravbola(R2,x2,y2,z2)
G2flat = G2.flatten()
G2flatarray = np.array(G2flat)
G2T = G2flatarray.reshape(shape1)

G3 = gravbola(R3,x3,y3,z3)
G3flat = G3.flatten()
G3flatarray = np.array(G3flat)
G3T = G3flatarray.reshape(shape1)

Gkarnel = o = np.concatenate((G1T, G2T, G3T), axis=1)

def inverseUD(d,Gx):
    Gt = Gx.transpose() # G transpose
    Gi = np.linalg.pinv(np.dot(Gt,Gx)) # [GGt+dampingI]-1
    m = np.dot(np.dot(Gi,Gt),d) # Gt*[GGt+dampingI]-1* d
    return m

rho = inverseUD(gteoT,Gkarnel)

print(rho)



