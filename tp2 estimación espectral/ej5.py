# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 16:52:10 2020

@author: mnaca
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from pandas import read_csv

# Metrica de Akaike
def AIC(m, sigma):
    muestras = 1000
    return 2 * (m + 1) + 2 * muestras * (1+np.log(2*np.pi*sigma))

def estimadorCoefMV(y, m,):
    
    N = y.size
    
    # y monio es (N-m-1)x1
    yy = np.zeros((N-m,1))
    for i in range(N-m):
        yy[i] = y[i+m]
    yy = yy[::-1]
    
    # para m <= n <= N se define y(n) = [y(n),...,y(n-m+1)] mx1
    YY = np.zeros((m,N-m))
    for i in range(m):
        for j in range(N-m):
            YY[i][j] = y[N-j-i-2]
    # ecuacion normal
    a_mv = np.dot( np.dot( np.linalg.inv( np.dot(YY,YY.T) ) , YY ) , yy )  
    # En el calculo se definieron los coeficientes como negativos 1 - a1*Y(n-1) - a2*Y(n-2) - a3*Y(n-3) - a4*Y(n-4)
    # Entonces como en teoria son positivos, a_mv los devuelve negativos
    sigma_mv = pow(np.linalg.norm( yy - np.dot(YY.T,a_mv) ),2)/(N-m)
    
    return a_mv , sigma_mv

# ================================================================================================ #

# Obtengo las muestras
y = read_csv('Ej4.csv', sep=',' , header = None)
y = y.to_numpy()

# Encuentro el mejor orden m:
aic = 0
aic_min = 0
sigma_aic = 0
for m in range(1,21):
    
    a_mv, sigma = estimadorCoefMV(y, m)
    aic = AIC(m, sigma)

    if m == 1:
        aic_min = aic
    if aic < aic_min:
        aic_min = aic
        orden = m
        a = np.append(1,-a_mv)
        sigma_aic = sigma

f_welch, psd_welch = signal.welch(y ,nperseg = 250, noverlap = 250/2 ,nfft = 5000
                                  , window = 'bartlett',return_onesided = False, axis = 0)

f, psd_aic = signal.freqz(np.array([ 1 ]) , a, whole = True)

plt.figure(figsize = (15,10))

plt.plot(np.linspace(0,2*np.pi,5000), (psd_welch), 
         label = r'$\hat{S}_Y$' + " Welch" , color = 'black')

plt.plot(f,sigma_aic * abs( psd_aic)**2 , color = 'red', linewidth = 3, 
         label = r'$\hat{S}_Y$'+" AR-{}".format(orden))

plt.legend(loc = 'best')
plt.xlabel("Frecuencia normalizada [rad/muestra]")
plt.xticks([ 0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 1.25*np.pi, 1.5*np.pi, 1.75*np.pi , 2* np.pi], 
           ['$0$', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', 
            r'$\pi$',  r'$\frac{5\pi}{4}$', r'$\frac{3\pi}{2}$', r'$\frac{7\pi}{4}$',r'$2\pi$' ])

plt.rc('font', size=18)
plt.grid(True)
plt.savefig("welchyakaike.png")


plt.figure(figsize = (15,10))

plt.plot(np.linspace(0,2*np.pi,5000), (psd_welch), 
         label = r'$\hat{S}_Y$' + " Welch" , color = 'black')

plt.legend(loc = 'best')
plt.xlabel("Frecuencia normalizada [rad/muestra]")
plt.xticks([ 0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 1.25*np.pi, 1.5*np.pi, 1.75*np.pi , 2* np.pi], 
           ['$0$', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', 
            r'$\pi$',  r'$\frac{5\pi}{4}$', r'$\frac{3\pi}{2}$', r'$\frac{7\pi}{4}$',r'$2\pi$' ])

plt.grid(True)
plt.savefig("welch.png")

plt.figure(figsize = (15,10))

plt.plot(f,sigma_aic * abs( psd_aic)**2 , color = 'red', linewidth = 3, 
         label = r'$\hat{S}_Y$'+" AR-{}".format(orden))

plt.legend(loc = 'best')
plt.xlabel("Frecuencia normalizada [rad/muestra]")
plt.xticks([ 0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 1.25*np.pi, 1.5*np.pi, 1.75*np.pi , 2* np.pi], 
           ['$0$', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', 
            r'$\pi$',  r'$\frac{5\pi}{4}$', r'$\frac{3\pi}{2}$', r'$\frac{7\pi}{4}$',r'$2\pi$' ])

plt.grid(True)
plt.savefig("akaike.png")




