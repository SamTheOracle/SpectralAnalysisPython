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

tempdata = pd.read_csv("D:\Repos\spectralanalysis\src\_air_Traffic_Passenger_Statistics.csv",parse_dates=["date"],index_col="date",usecols=["date","value"])
f,p = signal.periodogram(tempdata["value"])
plt.plot(f,p)
plt.show()


