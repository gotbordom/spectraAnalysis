#!/bin/usr/env/python

import numpy as np
import argparse
import h5py as h5
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
from scipy.fftpack import fft, ifft
from scipy import signal

ap = argparse.ArgumentParser()
ap.add_argument("-fN",
                required=True,
                type = str,
                help = "Path/filename of hdf5 file")
ap.add_argument("-ds",
                required=False,
                default='Spectra/Spectrum  10/Auxiliary/QA',
                type = str,
                help = "Name of specific dataset to load")
args = vars(ap.parse_args())


def model(t,coeffs):
  return coeffs[0]+coeffs[1]*np.exp(-1*((t-coeffs[2])/coeffs[3])**2)

def residuals(coeffs,y,t):
  return y - model(t,coeffs)


# Load the hdf5 datafile (df):
df = h5.File(args['fN'],'r')

# Pull dataset from datafile (ds):
fName = args['ds']+'Amplitude'
tName = args['ds']+'Time'
ds = np.array(df[fName])
t = np.array(df[tName])
dt = np.abs(t[1]-t[0])

# Looking at DFT of ds:
fds = fft(ds)
ft = np.linspace(0.0, 1.0/(2.0*dt), len(ds)//2)

print type(fds),len(fds)

# Load a triangle wave and fft it:
window = signal.triang(301)*.01
fWin = fft(window,len(fds))

print type(fWin),len(fWin)

# Sooo... smoothing ?
convDS = fds*fWin
dsSmooth = ifft(convDS)

print max(window)

# Trying for a quick qaussian fit with least squares:
# Nope didn't work with a gaussian model... no due
#x0 = np.array([3,30,15,1],dtype=float)
#x, flag = leastsq(residuals,x0,args=(ds,t))

#f,ax = plt.subplots(1,1,sharex=True)
#plt.plot(t,ds,t,model(t,x))
#plt.show()

fig, ax = plt.subplots(2,1)
ax[0].plot(window,'g')
ax[1].plot(t,ds,'r',t,dsSmooth.real,'b')
plt.show()

df.close()
