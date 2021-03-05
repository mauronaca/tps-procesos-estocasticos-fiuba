# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 16:50:15 2020

@author: mnaca
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def acorrSesgado(y):
    r = signal.correlate(y, y) / len(y)
    l = np.arange(-(len(y)-1), len(y))
    return r,l

#### PROCESO AR-4 ####

N = np.array([1000,5000]) # Camtodad de muestras
m = 4 # Orden del proceso
a = np.array([ 1, 0.3544,0.3508,0.1736,0.2401 ]) # Coeficientes del proceso DENOM
b = np.array([ 1, 0, 0, 0, 0 ])

for i in range(N.size):    
    
    w_axis = np.linspace(0,2*np.pi,256)
    # Genero ruido blanco
    W = np.random.normal(0, 1, N[i])
    # Genero el proceso Y
    Y = signal.lfilter(b, a, W)

    # Periodograma!
    w_periodogram,sy_periodogram = signal.periodogram(Y,window = 'bartlett',return_onesided = False , nfft = 256)
    # Respuesta en frecuencia del filtro , PSD teorica
    w_freq,sy_teorico = signal.freqz(b, a, whole = True)
    # Estimador Welch
    if(i == 0):
        f, y_psd = signal.welch(Y , window = 'bartlett', nperseg = 50, 
        nfft = 5000, noverlap = 50/2,return_onesided = False)
    if(i == 1):
        f, y_psd = signal.welch(Y , window = 'bartlett', nperseg = 250, 
        nfft = 5000, noverlap = 250/2,return_onesided = False)
    
    # Graficos
    plt.figure(figsize = (15,10))
    plt.plot( w_axis[1:], 20 * np.log10(sy_periodogram[1:]),
             linewidth = 1, color = 'green', label = r'$\hat{S}_Y$ Periodogram' )
    plt.plot(np.linspace(0,2*np.pi,5000),20 * np.log10(y_psd), 
             linewidth = 3, color = 'red', linestyle = '-', label = r'$\hat{S}_Y$ Welch' )
    plt.plot( np.linspace(0,2*np.pi,512), 20 * np.log10(pow(abs(sy_teorico),2)) , 
             linewidth = 2, color = 'black', linestyle = '--', label = r'$S_Y$')
    plt.grid(True)
    plt.title(" N = {}".format(N[i]))
    plt.legend(loc = 'best')
    plt.xlim(0,np.pi*2)
    plt.xticks([ 0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 1.25*np.pi, 1.5*np.pi, 1.75*np.pi , 2* np.pi], 
           ['$0$', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', r'$\pi$',  r'$\frac{5\pi}{4}$', r'$\frac{3\pi}{2}$', r'$\frac{7\pi}{4}$',r'$2\pi$' ])
    plt.rc('font', size=18)         
    plt.ylabel("Amplitud [dB]")
    plt.savefig("comparacion{}.png".format(N[i]))
    

    
    



"""
      Para calcular psd real
    w = np.arange(0,2*np.pi,1/100)
    WK = np.zeros((m+1,w.size), dtype = 'complex')

    for k in range(5):
        WK[k] = np.exp(-1j*k*w)
        
    den = np.dot(a,WK)
    
    sy = pow(abs(1/(den)),2)
"""