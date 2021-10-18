# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 09:03:51 2020

@author: Andidar83
"""

import numpy as np
import math
import matplotlib.pyplot as plt

xi= np.linspace(0,4000,20)
yi= np.linspace(0,4000,20)
X, Y = np.meshgrid(xi, yi) 
 


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

print(g1)

Z = np.reshape(gteo, (-1, 20))
plt.contourf(X,Y,Z)
plt.scatter(X, Y, marker = 'd', s = 20, zorder = 10)

#########################################################################

# Observed Data
xstation = np.linspace(0,4000,20)
data = np.array(gteo[9])



ndata = len(data)
obs = []
for val in data:
    obs.append(val + np.random.randint(0,50)/100) # obs + noise

## PARAMETER MODEL ##
    
# Grid Koordinat
xgrid = np.linspace(0,4000,20)
zgrid = np.linspace(0,2000,10)

ncorn = 4 # Jumlah corner rectangle
xblok = len(xgrid)-1 ; zblok = len(zgrid)-1
mmodel = xblok*zblok # Jumlah parameter blok

# Fungsi forward tanpa parameter rho
def gpoly(x0,z0,xcorn,zcorn,ncorn):
    total = 0
    for n1 in range(ncorn):
        if n1 == ncorn-1:
            n2 = 0
        else:
            n2 = n1 + 1
        x1 = xcorn[n1]-x0
        z1 = zcorn[n1]-z0
        x2 = xcorn[n2]-x0
        z2 = zcorn[n2]-z0
        r1sq = x1**2 + z1**2
        r2sq = x2**2 + z2**2
        if r1sq == 0:
            break
        if r2sq == 0:
            break
        denom = z2 - z1
        if denom == 0:
            denom = 1e-7
        alpha = (x2-x1)/denom
        beta = x1 - alpha*z1
        factor = beta/(1+alpha**2)
        term1 = 0.5*(np.log(r2sq)-np.log(r1sq))
        term2 = np.arctan2(z2,x2)-np.arctan2(z1,x1)
        total += factor*(term1-alpha*term2)
    Gij = 2*gamma*total*(10**5)*(10**3)
    return Gij

# Calculate Kernel
gamma = 6.67e-11
G = np.zeros((ndata,mmodel)) # Siapkan matriks kernel  

for i,site in enumerate(xstation):
    for j in range(zblok):
        zcorn = [zgrid[j],zgrid[j],zgrid[j+1],zgrid[j+1]]
        for k in range(xblok):
            xcorn = [xgrid[k],xgrid[k+1],xgrid[k+1],xgrid[k]]
            G[i,(j*xblok)+k] = gpoly(site,0,xcorn,zcorn,ncorn) # z0 = 0
       
## INVERSI ##

# Fungsi Solusi Inversi Under-determined
modelref =  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0.1,0.1,0.1,0.1,0.1,0.1,0,0,0,0,0,0,0,0,0,0,0,0,0,0.1,0.1,0.1,0.1,0.1,0.1,0,0,0,0,0,0,0,0,0,0,0,0,0,0.1,0.1,0.1,0.1,0.1,0.1,0,0,0,0,0,0,0,0,0,0,0,0,0,0.1,0.1,0.1,0.1,0.1,0.1,0,0,0,0,0,0,0,0,0,0,0,0,0,0.1,0.1,0.1,0.1,0.1,0.1,0,0,0,0,0,0,0,0,0,0,0]
modelreft = np.transpose(modelref)

def inverseUD(d,G,damping):
    Gt = G.transpose() # G transpose
    I = np.identity(len(d))
    Gi = np.linalg.pinv(np.dot(G,Gt)+np.dot(damping,I)) # [GGt+dampingI]-1
    m = np.dot(np.dot(Gt,Gi),d) # Gt*[GGt+dampingI]-1* d
    print(len(d))
    return m

rho = inverseUD(obs,G,0.000001)

model = np.reshape(rho,(zblok,xblok))



# Calculated Data
cal = []
for i,site in enumerate(xstation):
    grav = 0
    for j in range(zblok):
        zcorn = np.array([zgrid[j],zgrid[j],zgrid[j+1],zgrid[j+1]])
        for k in range(xblok):
            xcorn = np.array([xgrid[k],xgrid[k+1],xgrid[k+1],xgrid[k]])
            grav += gpoly(site,0,xcorn,zcorn,ncorn)*rho[j*xblok+k]
    cal.append(grav)
    
## PLOTTING ##
    
import matplotlib.pyplot as plt
import matplotlib.cm as cm

plt.figure(figsize=(9,9))
ax1 = plt.subplot(211)
ax1.plot(xstation,obs,'o')
ax1.plot(xstation,cal,'r')
ax2 = plt.subplot(212)
mod = ax2.imshow(model, extent=(xgrid.min(), xgrid.max(), zgrid.max(), zgrid.min()),
           aspect='auto', cmap=cm.gist_rainbow)
plt.colorbar(mod, orientation='horizontal')
plt.show()


print("OBS=",G.shape)



#############################################################

"""
modelref = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0.1,0.1,0.1,0.1,0.1,0.1,0,0,0,0,0,0,0,0,0,0,0],[0,0,0.1,0.1,0.1,0.1,0.1,0.1,0,0,0,0,0,0,0,0,0,0,0],[0,0,0.1,0.1,0.1,0.1,0.1,0.1,0,0,0,0,0,0,0,0,0,0,0],[0,0,0.1,0.1,0.1,0.1,0.1,0.1,0,0,0,0,0,0,0,0,0,0,0],[0,0,0.1,0.1,0.1,0.1,0.1,0.1,0,0,0,0,0,0,0,0,0,0,0]]


 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0]
 
[[ 0.32400062  0.55007978  0.15929724  0.08813189  0.43695185 -0.10651151
   0.66167301 -0.15816901  0.45869543 -0.17511291  0.23511298 -0.02095048
   0.09777238 -0.01196806 -0.11429108  0.34162399 -0.13396122  0.25092921
  -0.87715349]
 [ 0.61931027  0.31371809  0.64743058  1.05253029  1.19332428  1.22383395
   0.79784545  0.6758744   0.34142921  0.29900903  0.09905695  0.08530254
   0.09860263  0.16754529  0.13470174 -0.00645263  0.19266816  0.27411817
   0.22357478]
 [ 0.59624921  0.55346837  0.65781355  0.8125793   0.89605784  0.86915171
   0.72810676  0.57698067  0.42169277  0.30783872  0.2123731   0.16209536
   0.14648278  0.14370718  0.12257914  0.09953126  0.12112923  0.11398706
   0.00230399]
 [ 0.5452825   0.56361229  0.61484675  0.67853999  0.71206127  0.69056889
   0.61735544  0.51909932  0.41709739  0.3270575   0.25469635  0.20437981
   0.17318996  0.15147879  0.13045358  0.11149263  0.09530075  0.06241729
  -0.00387466]
 [ 0.49405757  0.52210979  0.55403426  0.58324354  0.59438941  0.57662775
   0.53053021  0.46609     0.39534434  0.32802764  0.2700779   0.22410463
   0.18911629  0.16134189  0.13694009  0.11374042  0.08871212  0.05610757
   0.01422274]
 [ 0.44701177  0.47252154  0.49431105  0.50927712  0.51144969  0.49639922
   0.46428757  0.4196358   0.36888061  0.3180215   0.27133279  0.23096805
   0.19700037  0.16798598  0.1419587   0.11704093  0.09127721  0.06303553
   0.03305856]
 [ 0.40523585  0.42601977  0.44151414  0.44993545  0.44872905  0.43612221
   0.4123737   0.3798566   0.34215951  0.30294586  0.26511304  0.23038525
   0.19927677  0.17136123  0.1457094   0.12126065  0.09708261  0.07279645
   0.04919629]
 [ 0.36870222  0.38510251  0.39640413  0.40148935  0.39915619  0.38871091
   0.37047418  0.34585075  0.31696199  0.28608958  0.25518539  0.22557837
   0.19790034  0.1721954   0.14813802  0.12527405  0.10325585  0.08209144
   0.06233586]
 [ 0.33693377  0.34978556  0.35818916  0.36141238  0.35883821  0.35020993
   0.33583754  0.31662563  0.29390264  0.26913884  0.24366917  0.21850308
   0.19425257  0.17116937  0.14925674  0.12841546  0.10859485  0.08992396
   0.07277438]]
"""