#!/bin/usr/env/python

import numpy as np
import argparse
import h5py as h5
import matplotlib.pyplot as plt
import scipy.interpolate as inter
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
                default='   3',
                type = str,
                help = "Number of specific dataset to load. For example, if I want 'Spectra/Spectrum   3/Auxiliary/QA' I only need enter '   3'. Make note of white space, I haven't corrected this yet.")
ap.add_argument("-t",
                required=False,
                default='??',
                type = str,
                help = "Name of Target for this data - helps organize saved files.")
ap.add_argument("-n",
                required=False,
                default='none',
                type=str,
                help = "What noise to use: none, mean or median. None is default.")
args = vars(ap.parse_args())


def H(x):
  # Heavyside step function:
  return 1*(x>0)

def model2(t,coeffs):
  # coeffs = [amp,noise,t1,tm,t2]
  amp,noise,t1,tm,t2=coeffs
  y1 = amp/(tm-t1)
  y2 = amp/(tm-t2)

  a = y1*(t-t1)*(H(t-t1)-H(t-tm))
  b = y2*(t-t2)*(H(t-tm)-H(t-t2))
  
  return a+b+noise

def residuals(coeffs,y,t):
  return y - model2(t,coeffs)


# Load the hdf5 datafile (df):
df = h5.File(args['fN'],'r')

# Pull dataset from datafile (ds):
sName = 'Spectra/Spectrum'+args['ds']+'/Auxiliary/QA'
fName = sName+'Amplitude'
tName = sName+'Time'
ds = np.array(df[fName])

# For my model I want to make sure time is all positive...
time = np.array(df[tName],dtype=float)
t = time-time[0]
dt = np.abs(t[1]-t[0])

# Smoothing the DS using dft:
fds = fft(ds)
ft = np.linspace(0.0, 1.0/(2.0*dt), len(ds)//2)

# Load triangle wave and fft it to convolute:
window = signal.triang(401)*.01
fWin = fft(window,len(fds))

convDS = fds*fWin
dsSmooth = ifft(convDS)
amp = float(np.min(dsSmooth.real))
tmInd = np.argmin(dsSmooth.real)
tm = t[tmInd]

# Boolian for what noise to use:
if args['n'] is 'mean':
  noise=np.mean(dsSmooth.real)
elif args['n'] is 'median':
  noise=np.median(dsSmooth.real)
else:
  noise=0

print "Amp: ",amp
print "tm: ",tm
print "Noise: ",noise

# Now to fit my data to my model:
step = 50
x0 = np.array([amp,noise,t[tmInd-step],tm,t[tmInd+step]],dtype=float)
x,flag = leastsq(residuals,x0,args=(dsSmooth.real,t))

y_=model2(t,x)

fig,ax = plt.subplots(2,1)
ax[0].plot(window)
ax[0].set_title("Window for smoothing")
ax[1].plot(t,ds,'r',t,dsSmooth.real,'b',t,y_,'k')
ax[1].set_title("Data vs. Smoothed data w/ fitted line")

tmp = args['t']+sName.replace("/","_")
tmp = tmp.replace(" ","0")
print tmp
plt.savefig(tmp+".jpg",format='jpg')

#plt.show()

