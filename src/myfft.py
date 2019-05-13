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

df = pd.read_csv('D:\Repos\spectralanalysis\src\_annual_sunspots.csv',parse_dates=['date'],index_col='date',usecols=["date","mean"],delim_whitespace=True)
df_monthly = pd.read_csv('D:\Repos\spectralanalysis\src\_sunspots.csv',parse_dates=['date'],index_col='date',usecols=["date","mean"])
values = df["mean"]
values_monthly = df_monthly["mean"]
Y = np.fft.fft(values)
n = len(Y)
print(n / 2)
power = np.abs(Y)**2
nyquist = 1./2
freq=array(range(n))
period = 1./freq
f,p = signal.periodogram(values)
f_monthly,p_monthly = signal.periodogram(values_monthly)
period = 1./f
# plt.plot(period,power)
# plt.figure(1)
# #plt.show()
plt.figure(1)
plt.plot(f,p)
plt.figure(2)
plt.plot(f_monthly,p_monthly)
airline = pd.read_csv('D:\Repos\spectralanalysis\src\_airline_passengers.csv',parse_dates=['date'],index_col='date')
series = airline["number"]
# series_for_detrending = airline.values
# detrended = []
# for index in range(1,len(series_for_detrending)):
#     diff = series[index]-series[index-1]
#     detrended.append(diff)

f_air,p_air = signal.periodogram(series,detrend="linear")
# f_air_d,p_air_d = signal.periodogram(detrended)
plt.figure(3)
period_air = 1./f_air
plt.plot(f_air,p_air)
# plt.figure(4)
# plt.plot(f_air_d,p_air_d)
plt.show()
