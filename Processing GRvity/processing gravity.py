# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 21:04:05 2021

@author: Andidar83
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as inter
from scipy.signal import convolve2d


#Read data freeair dan elevasi dari Topex berformat txt
#Format data Untuk saat ini X, Y, elev, FAA 
data = np.loadtxt('Sesar Saddang.txt', skiprows=1)

#Mengambil data hanya nilai freeair dan elevasinya saja
fa = data[:, 3] #mengambil semua data freeair yang ada di kolom 2
elev = data[:, 2] #mengambil semua data elevasi yang ada di kolom 2
ba = np.array(0.04192*elev) #bouger tanparho

#koordinat titik pengukuran
east_g = data[:, 0]
north_g = data[:, 1]

#Estimasi Densitas using Parasnis
def density_parasnis_est(freeair, elevasi):
    freeair = np.transpose(np.array([freeair]))
    elevasi = np.transpose(np.array([elevasi]))
    konstanta = 1/(.04192)
    return float(konstanta * np.transpose(elevasi).dot(np.linalg.pinv(elevasi.dot(np.transpose(elevasi)), hermitian=True)).dot(freeair))

density_parasnis = density_parasnis_est(fa, elev)

#Bouger Correction
bouger_correction = np.array(elev*0.04192*density_parasnis)

#Simple Bouger Anomaly
SBA = fa - bouger_correction

#Gridding Koordinat
east_g_grid= np.linspace(np.min(east_g), np.max(east_g), 100) #grid koordinat
north_g_grid= np.linspace(np.min(north_g), np.max(north_g), 100)
x,y = np.meshgrid(east_g_grid, north_g_grid)

#melakukan interpolasi G dan Elev
interpolasig = inter.Rbf(east_g,north_g,SBA, function='cubic')
interpolasiz = inter.Rbf(east_g,north_g,elev, method='cubic')
g_interp = interpolasig(x,y)

fig = plt.figure(figsize=(12,8))
plt.contourf(x, y, g_interp, cmap='nipy_spectral', levels=50)
plt.colorbar()
plt.contour(x, y, g_interp, colors='black', levels=5, linewidths=1)
plt.contour(x, y, g_interp, levels=15, linewidths=0.5, linestyles='solid')
# plt.scatter(x, y, color='black')
# plt.plot(diagx, diagy, color='black', linewidth=2)
plt.title('Simple Bouger Anomaly (mgal)', fontsize='15', fontweight='bold')
plt.xlabel('Easting', fontsize='15', fontweight='bold')
plt.ylabel('Northing', fontsize='15', fontweight='bold')
plt.show()