from dateutil.parser import parse 
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import seaborn as sns
import numpy as np
import pandas as pd
import sys
from scipy import *
from scipy import signal


airline = pd.read_csv('D:\Repos\spectralanalysis\src\_airline_passengers.csv',parse_dates=['date'],index_col='date')
numbers = airline["number"]
plt.figure()
plt.title("Number of passengers each month")
plt.ylabel("Number of passangers")
plt.xlabel("Month")
plt.plot(airline)
plt.figure()
plt.title("Result of detrending with least squares")
# the series is not stationary, in order to apply fft we need to make it stationary by detrending
detrended_series = signal.detrend(numbers,type="linear")

plt.plot(detrended_series)
fft = np.fft.fft(detrended_series)
n = len(fft)
# for only real values time series, fft has the conjugate complex simmetry
# it is sufficient to consider only the first half of the values since the rest is the same
# X_real[k] = X_real[N-k]
# X_im[k] = - X_im[N-k]
# with k in [0,N]
one_sided_fft = fft[:int(n/2)+1]
#sampling rate is 1: number of observation is 1 per unit of time (1 obs per month)
Fs = 1
# nyquist theorem: is it sufficient to consider only the frequencies up to the sample rate / 2
nyquist = Fs/2
# magnitude at power of 2 divided by the length
power = np.abs(one_sided_fft)**2 / n
# normalized frequency  with n/2 +1 values 
freq=array(range(n))/n
# frequencies values necessary from 0 to nyquist
freq_to_plot = freq[:int(n/2)+1]
plt.figure()
plt.title("Periodogram (Power Spectrum Density)")
power_max = np.amax(power)
index = np.where(power == power_max)
frequency_max = freq[index]
plt.annotate('Maximum peak\nat frequency\n'+str(round(float(frequency_max),3))+', 12 months', xy=(frequency_max, power_max), xytext=(0.36, 30000),
            arrowprops=dict(facecolor='black', shrink=0.05))
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.plot(freq_to_plot,power)
plt.show()