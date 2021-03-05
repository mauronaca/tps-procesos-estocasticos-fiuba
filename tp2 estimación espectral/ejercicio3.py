# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 14:32:03 2020

@author: mnaca
"""
import numpy as np
import numpy.fft as fft
from scipy.signal import correlate
from scipy import signal  

def acorrSesgado(y):
    r = correlate(y, y) / len(y)
    l = np.arange(-(len(y)-1), len(y))
    return r,l


def blackmanTukey(y, w, nfft):
    # w : ventana
    N = len(y)
    M = len(w)
    
    # Estimar la correlacion
    ryy, k = acorrSesgado(y)
    
    # Ventaneo la autocorrelacion
    ryy = ryy[np.logical_and(k >= -(M/2), k < M/2)]
    ryw = ryy * w
    
    # Computo la fft:
    Y = fft.fft(ryw, nfft)
    f = np.arange(nfft) / nfft # eje de frecuencias
    
    return (pow(np.abs(Y),1), f)



#############################################################################################
                                     ## Ejercicio 3##
#############################################################################################

import matplotlib.pyplot as plt

hDenominador = np.array([1, 0.3544, 0.3508, 0.1736, 0.2401])
hNumerador = np.array([1, -1.3817, 1.5632, -0.8843, 0.4096])
[angularFrequencyAxis, Syy_teorico] = signal.freqz(hNumerador,   hDenominador, whole = True) # En frecuencias rad/sample * 2pi

nfft = 256
J = 1000
M2 = int(nfft/16)
M3 = int(nfft/4)
ventanaBox = signal.boxcar(nfft)
ventanaTri1 = np.bartlett(M2)
ventanaTri2 = np.bartlett(M3)

X = np.random.normal(0, 1, (J,256)) # Filas: Realizaciones ; Columnas: Muestras de una realizacion de un proceso
Y = signal.lfilter(hNumerador, hDenominador, X, axis = 1)

# Estimador con M = N y ventana rectangular


psd_mean = np.zeros(int(nfft))
for i in range(J):
    (psd, w) = blackmanTukey(Y[i], ventanaBox ,nfft)
    psd_mean += psd
psd_mean /= J

# Varianza : 
sigma_psd = np.zeros(int(nfft))
for i in range(J):
    (psd, w) = blackmanTukey(Y[i], ventanaBox ,nfft)
    sigma_psd += pow((psd - psd_mean),2)
sigma_psd /= (J - 1)


plt.figure(figsize = (15,10))
plt.plot(np.linspace(0,2*np.pi,nfft),psd_mean, linewidth = 4, color = 'red',label = r'$\bar{\hat{S_Y}}$')
plt.plot(np.linspace(0,2*np.pi,nfft),pow(sigma_psd,1/2) - psd_mean, linewidth = 4, color = 'violet',
         label = r'$\hat{\sigma_{S_Y}}-\bar{\hat{S_Y}}$')
plt.plot(np.linspace(0,2*np.pi,512),(pow(np.abs(Syy_teorico), 2)),linestyle = 'dashed' ,color = 'black',label = r'$S_Y$')

plt.title(r'Estimador Blackman Tukey con M = N = 256')
plt.xlabel('Frecuencia Angular [rad/muestra]')
plt.ylabel('Amplitud')
plt.grid(b = True, color = 'black', linestyle = '-', linewidth = 0.4)
plt.xticks([ 0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 1.25*np.pi, 1.5*np.pi, 1.75*np.pi , 2* np.pi], 
           ['$0$', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', r'$\pi$',  r'$\frac{5\pi}{4}$', r'$\frac{3\pi}{2}$', r'$\frac{7\pi}{4}$',r'$2\pi$' ])
plt.rc('font', size=20)         
plt.legend()
plt.savefig("blackmanTukey256.png")


# Para M = N/16

psd_mean = np.zeros(int(nfft))
for i in range(J):
    (psd, w) = blackmanTukey(Y[i], ventanaTri1 ,nfft)
    psd_mean += psd
psd_mean /= J


plt.figure(figsize = (15,10))
plt.plot(np.linspace(0,2*np.pi,nfft),psd_mean, linewidth = 4, color = 'red',label = r'$\bar{\hat{S_Y}}$')
plt.plot(np.linspace(0,2*np.pi,512),(pow(np.abs(Syy_teorico), 2)),linestyle = 'dashed' ,color = 'black',label = r'$S_Y$')
plt.title(r'Estimador Blackman Tukey con M = N/16')
plt.xlabel('Frecuencia Angular [rad/muestra]')
plt.ylabel('Amplitud')
plt.grid(b = True, color = 'black', linestyle = '-', linewidth = 0.4)
plt.rc('font', size=20)     
plt.xticks([ 0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 1.25*np.pi, 1.5*np.pi, 1.75*np.pi , 2* np.pi], 
           ['$0$', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', r'$\pi$',  r'$\frac{5\pi}{4}$', r'$\frac{3\pi}{2}$', r'$\frac{7\pi}{4}$',r'$2\pi$' ])
    
plt.legend()
plt.savefig("blackmanTukey16.png")

### Para M = N/4 ###

psd_mean = np.zeros(int(nfft))
for i in range(J):
    (psd, w) = blackmanTukey(Y[i], ventanaTri2 ,nfft)
    psd_mean += psd
psd_mean /= J


plt.figure(figsize = (15,10))
plt.plot(np.linspace(0,2*np.pi,nfft),psd_mean, linewidth = 4, color = 'red',label = r'$\bar{\hat{S_Y}}$')
plt.plot(np.linspace(0,2*np.pi,512),(pow(np.abs(Syy_teorico), 2)),linestyle = 'dashed' ,color = 'black',label = r'$S_Y$')
plt.title(r'Estimador Blackman Tukey con M = N/4')
plt.xlabel('Frecuencia Angular [rad/muestra]')
plt.ylabel('Amplitud')
plt.grid(b = True, color = 'black', linestyle = '-', linewidth = 0.4)
plt.rc('font', size=20)         
plt.xticks([ 0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 1.25*np.pi, 1.5*np.pi, 1.75*np.pi , 2* np.pi], 
           ['$0$', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', r'$\pi$',  r'$\frac{5\pi}{4}$', r'$\frac{3\pi}{2}$', r'$\frac{7\pi}{4}$',r'$2\pi$' ])

plt.legend()
plt.savefig("blackmanTukey32.png")


plt.show()