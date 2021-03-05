import numpy as np
from scipy import signal  
import scipy
import matplotlib.pyplot as plt 


#############################################################################################
    ## Grafico teorico de la densidad espectral de potencia S_Y tal que Y = h * W ##
                                ## Ejercicio 1##
#############################################################################################


hDenominador = np.array([1, 0.3544, 0.3508, 0.1736, 0.2401])
hNumerador = np.array([1, -1.3817, 1.5632, -0.8843, 0.4096])
[angularFrequencyAxis, Syy_teorico] = signal.freqz(hNumerador,  hDenominador, whole = True) # En frecuencias rad/sample * 2pi


plt.figure(figsize = (15,10))
plt.plot(angularFrequencyAxis, ((pow(np.abs(Syy_teorico), 2))), linewidth = 2, color = 'red')
plt.xlim(xmin = angularFrequencyAxis[0])
plt.xlim(xmax = angularFrequencyAxis[angularFrequencyAxis.size - 1])
plt.title(r'Densidad Espectral de Potencia $S_Y$')
plt.xlabel('Frecuencia Angular [rad/muestra]')
plt.ylabel('Amplitud')
plt.xticks([ 0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 1.25*np.pi, 1.5*np.pi, 1.75*np.pi , 2* np.pi], 
           ['$0$', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', r'$\pi$',  r'$\frac{5\pi}{4}$', r'$\frac{3\pi}{2}$', r'$\frac{7\pi}{4}$',r'$2\pi$' ])
plt.grid(b = True, color = 'black', linestyle = '-', linewidth = 0.4)
plt.rc('font', size=25)         
plt.savefig("psdteorica.png")

#############################################################################################
