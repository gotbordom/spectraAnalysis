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
                default='Spectra/Spectrum   3/Auxiliary/QA',
                type = str,
                help = "Name of specific dataset to load")
args = vars(ap.parse_args())


def H(x):
  # Heavyside step function:
  return 1*(x>0)

def model(t,tm,coefs):
  # coefs = [eta1,eta2,m1,m2,,t1,tm,t2] ideally b&c are gone due to assumptions
  eta1,eta2,m1,m2,t1,t2=coefs
  a = eta1*(H(t)-H(t-t2))
  b = eta2*(H(t-t2))
  c = m1*t*(H(t-t1)-H(t-tm))
  d = (m2*t+(tm*(m1-m2)))*(H(t-tm)-H(t-t2))
  return a+b+c+d

def model1(t,coeffs):
  # coeffs = [q,t1,tm,t2,eta]]
  q,t1,tm,t2,eta=coeffs

  a = eta*(H(t))
  b = (q*t+q*t1)*(H(t-t1)-H(t-tm))
  c = (-q*t+2*q*tm+q*t1)*(H(t-tm)-H(t-t2))
  return b+c

#def model(t,coeffs):
#  return coeffs[0]+coeffs[1]*np.exp(-1*((t-coeffs[2])/coeffs[3])**2)

def residuals(coeffs,y,t,tm):
  return y - model1(t,tm,coeffs)


# Load the hdf5 datafile (df):
df = h5.File(args['fN'],'r')

# Pull dataset from datafile (ds):
fName = args['ds']+'Amplitude'
tName = args['ds']+'Time'
ds = np.array(df[fName])
# For my model I want to make sure time is all positive...
t = np.array(df[tName],dtype=float)
print t[0]
t = t-t[0]
print t[0]

dt = np.abs(t[1]-t[0])

# Looking at DFT of ds:
fds = fft(ds)
ft = np.linspace(0.0, 1.0/(2.0*dt), len(ds)//2)

print type(fds),len(fds)

# Load a triangle wave and fft it:
window = signal.triang(401)*.01
fWin = fft(window,len(fds))

print type(fWin),len(fWin)

# Sooo... smoothing ?
convDS = fds*fWin
dsSmooth = ifft(convDS)
tm = float(np.min(dsSmooth.real))
tmInd = np.argmin(dsSmooth.real)
print tmInd

print max(window)

# Trying for a quick qaussian fit with least squares:
# Nope didn't work with a gaussian model... no due
x0 = np.array([-0.5,1,2,3,1],dtype=float)
#x, flag = leastsq(residuals,x0,args=(dsSmooth.real,t,tm))

# Test the model to see if it plots well:
testTime = np.arange(0,10,0.1)
test = model1(testTime,x0)
plt.plot(testTime,test)

#print "x:" ,x

#y_ = model1(t,tm,x)
# commenting out for now working with splines...
#fig, ax = plt.subplots(2,1)
#ax[0].plot(window,'g')
#ax[1].plot(t,ds,'r',t,dsSmooth.real,'b',t,y_,'g')
#ax[1].axhline(y=np.mean(dsSmooth.real),color='k',linestyle='-')
#plt.savefig('signal.jpg',format='jpg')
#
plt.show()

# Trying for a spline...
# Over fits ... so I can't really find where the t-in and t-out are
#tt = np.arange(0,t[-1],dt*1000)
#print len(t),len(tt)
#s1 = inter.InterpolatedUnivariateSpline(t,dsSmooth.real)
#s1rev = inter.InterpolatedUnivariateSpline(t[::-1],dsSmooth.real[::-1])
#s2 = inter.UnivariateSpline(t[::-1],dsSmooth.real[::-1],s=0.1)

#plt.plot(t,dsSmooth.real,'r',label='Smooth data')
#plt.plot(tt,s1(tt),'b--',label='Spline, wrong order')
#plt.plot(tt,s1rev(tt),'g--',label='Spline, Correct order')
#plt.plot(tt,s2(tt),'k-',label='Spline,fit')

#plt.minorticks_on()
#plt.legend()
#plt.show()

# Working on finding coeficientts for model1:
# Need to find the b & c terms in the medel1 piecewise function:

df.close()
