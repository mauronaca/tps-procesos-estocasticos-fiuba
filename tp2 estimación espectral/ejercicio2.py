# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 14:25:14 2020

@author: mnaca
"""

import numpy as np
from scipy import signal  
import scipy
import matplotlib.pyplot as plt 

hDenominador = np.array([1, 0.3544, 0.3508, 0.1736, 0.2401])
hNumerador = np.array([1, -1.3817, 1.5632, -0.8843, 0.4096])
[angularFrequencyAxis, Syy_teorico] = signal.freqz(hNumerador, hDenominador, whole = True) # En frecuencias rad/sample * 2pi



#############################################################################################
                                     ## Ejercicio 2##
#############################################################################################

# X:Ruido de entrada
# Cantidad de realizaciones del proceso X
J = 100
# Cantidad de muestras de cada realizacion del proceso X
N = np.array([64,512]) # Para el ejercicio va esto!

plt.figure(figsize = (15,10))

for n in range(N.size):
    
    X = np.random.normal(0, 1, (J,N[n])) # Filas: Realizaciones ; Columnas: Muestras de una realizacion de un proceso
    w = np.linspace(0,2*np.pi,N[n]);

    # Filtro X con H(z)
    Y = signal.lfilter(hNumerador, hDenominador, X, axis = 1)

    # Media de las J's estiamaciones del periodograma de Sy
    Syy_promedio = np.zeros(int(N[n]))
    for i in range(J):
        (wyy, syy) = signal.periodogram(Y[i], axis = -1, return_onesided = False, scaling = 'density', nfft = N[n]) # Periodograma de Y en frecuencia rad/sample normalizado ; Con nfft = N
        Syy_promedio += syy
    Syy_promedio /= J

    # Varianza : Iterar de vuelta y restar cada periodograma por la Syy_promedio
    SyySigma = np.zeros(int(N[n]))
    for i in range(J):
        (wyy, syy) = signal.periodogram(Y[i], return_onesided = False , scaling = 'density')
        SyySigma += pow((syy - Syy_promedio),2)
    SyySigma /= (J - 1)

    plt.plot(w, Syy_promedio, label = r'$\bar{\hat{S}}_Y$ ' + " N = {}".format(N[n]))
    plt.plot(w, pow(SyySigma,1/2) - Syy_promedio, label = r'$\sigma_{\hat{S}_Y} - \bar{\hat{S}}_Y$ '+ " N = {}".format(N[n]))
    
plt.plot(w, (pow(np.abs(Syy_teorico), 2)), label = r'$S_Y$') 

plt.title(r'Estimacion de la Densidad Espectral de Potencia $\bar{\hat{S}}_Y$')
plt.xlabel('Frecuencia (0, $\pi$) [rad/samples] ')
plt.ylabel('Amplitud')
plt.rc('font', size=20)         
plt.grid(b = True, color = 'black', linestyle = '-', linewidth = 0.4)
plt.legend()
plt.rc('font', size=20)         
plt.xticks([ 0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 1.25*np.pi, 1.5*np.pi, 1.75*np.pi , 2* np.pi], 
           ['$0$', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', r'$\pi$',  r'$\frac{5\pi}{4}$', r'$\frac{3\pi}{2}$', r'$\frac{7\pi}{4}$',r'$2\pi$' ])

plt.savefig("periodogram64y512.png")

#############################################################################################