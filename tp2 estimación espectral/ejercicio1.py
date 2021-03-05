# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 18:51:58 2020

@author: mnaca
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


### Generacion de muestras de Y(n) ####

N = 1000 # Camtodad de muestras
m = 4 # Orden del proceso
a = np.array([ 1, 0.3544,0.3508,0.1736,0.2401 ]) # Coeficientes del proceso DENOM
b = np.array([ 1, 0, 0, 0, 0 ]) 

# Genero ruido blanco
W = np.random.normal(0, 1, N)
# Genero muestras del proceso Y(n)
Y = signal.lfilter(b, a, W)


# --------------------------------------------------------------------------------------#
#  Para m = 4, se estiman los coeficientes a_MV mediante cuadrados minimos              #
# --------------------------------------------------------------------------------------#

## Estimador de maxima verosimilitud de los coeficientes de a :
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
    
a_mv , sigma_mv = estimadorCoefMV(Y, 4)
print("Coeficientes de orden 4 por MV:")
print(-a_mv)
print("Sigma por MV:",sigma_mv)
a_mv = np.append(1, -a_mv)


# --------------------------------------------------------------------------------------#
# Se busca el orden m usando el criterio de Akaike y luego se estiman los coeficientes #
# --------------------------------------------------------------------------------------#

# Metrica de Akaike
def AIC(m, sigma):
    return 2 * (m + 1) + 2 * N * (1+np.log(2*np.pi*sigma))

aic = np.zeros(10)
# [m=1, m=2, m=3, m=4, m=5, m=6 ,... , m=10]
a_mv_lista = []
sigma_mv_lista = []
for mm in range(1,11):
    coef , sigma_MV = estimadorCoefMV(Y, mm)
    a_mv_lista.append(coef)
    sigma_mv_lista.append(sigma_MV)
    
    #print('\n')
    #print("Para m = {}:".format(m))
    #print("Coeficientes:")
    #print(-a_mv_lista[m-1])
    #print("Varianza = ", sigma_MV)
    
    aic[mm-1] = AIC(mm,sigma_MV)

# Imprime el indice + 1, seria el orden m
print("\nMinimo AIC(m):")
m_aic = np.argmin(aic)+1 # La minima metrica de Akaike
print("m = ", m_aic)
print("AIC({})".format(m_aic), aic[m_aic-1])
a_mv_aic = np.zeros(m_aic)
a_mv_aic = np.append(1, -a_mv_lista[m_aic - 1])
sigma_mv_aic = sigma_mv_lista[m_aic - 1]
print("Sigma = ", sigma_mv_aic)
print("Coeficientes:")
print(a_mv_aic.T)
print('\n')

# Diferencia entre la metrica de Akaike 1...10 y la minima:
for mm in range(1,11):
    delta = aic[mm-1] - aic[m_aic-1]
    print(r'Δ'+"({})".format(mm)+"=AIC({})".format(mm)+"-minAIC({})".format(m_aic))
    print(delta)
    
# Graficos....
plt.figure(figsize = (15,10))

# Estimador de Welch:
w_axis, y_psd = signal.welch(Y , window = 'bartlett', nperseg = 50, noverlap = 50/2 ,nfft = N, return_onesided = False)
plt.plot(np.linspace(0,2*np.pi,1000), 20 * np.log10(pow(abs(y_psd),1)), label = r'$\hat{S}_Y$ - Welch', color = 'black', 
          linewidth = 2)

# PSD verdadera:
f, psd = signal.freqz(b, a, whole = True)
plt.plot(f, 20 * np.log10(pow(abs(psd),2)), label = r'$S_Y$', linestyle = '--', color = 'black', linewidth = 3)

# Por estimador MV de orden 4
f_mv, psd_mv = signal.freqz(b, a_mv, whole = True)
plt.plot(f_mv, 20 * np.log10(pow(abs(psd_mv),2))*sigma_mv, label = r'$\hat{S}_Y$, m=4', color = 'red', linewidth = 3)

# Por estimador MV y akaike
f_mv_aic, psd_mv_aic = signal.freqz(b, a_mv_aic, whole = True)
plt.plot(f_mv_aic, 20 * np.log10(pow(abs(psd_mv_aic),2))*sigma_mv_aic, label = r'$\hat{S}_Y$,'+" Akaike m={}".format(m_aic) ,
         color = 'orange', linewidth = 3)

plt.legend(loc = 'best')
plt.xlabel("Frecuencia normalizada [rad]")
plt.ylabel(r'Amplitud [dB]')
plt.xlim(0,2*np.pi)
plt.ylim(-20,11)
plt.grid(True)
plt.rc('font', size=18)
plt.savefig("estimador.png")


##============================================================================###
## Repetir 2000 veces el experimento:

# Para 2000 el tiempo de carga es demasiado... Lo pongo el 100 y despues lo camibo
orden = np.zeros(10)
for i in range(10): 
    
    ruido = np.random.normal(0, 1, N)
    y = signal.lfilter(b, a, ruido)
    orden[i] = 1
    min_aic = 0
    
    for mm in range(1,11):
        
        coef, sigmas = estimadorCoefMV(y,mm)
        _aic = AIC(mm,sigmas)
        # Busco el menor:
        if mm == 1:
            min_aic = _aic
        if _aic < min_aic:
            min_aic = _aic
            orden[i] = mm # Orden dado por la metrica
            
plt.figure(figsize = (15,10))
hist, bins, patches = plt.hist(orden, bins = 10, density = True)        
plt.xticks(range(1,11))       
colores = iter( ( plt.get_cmap('winter') )( np.linspace(0,1,len(patches)) ) )
for i in range(len(patches)):
    patches[i].set_facecolor(next(colores))
plt.yticks(np.arange(0,1,0.1))        
plt.ylim(0,np.max(hist))        
plt.xlabel("Orden m")        
plt.ylabel("Probabilidad")       
#plt.savefig("hist.png")

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        



        
        
        
        
        