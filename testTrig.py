#!/bin/usr/env/python

import numpy as np
import matplotlib.pyplot as plt


def H(x):
  # Heavyside step function:
  return 1*(x>0)

def model(coefs,t):
  # coefs = [eta1,eta2,m1,m2,,t1,tm,t2] ideally b&c are gone due to assumptions
  eta1,eta2,m1,m2,t1,tm,t2=coefs
  a = eta1*(H(t)-H(t-t2))
  b = eta2*(H(t-t2))
  c = m1*t*(H(t-t1)-H(t-tm))
  d = (m2*t+(tm*(m1-m2)))*(H(t-tm)-H(t-t2))
  return a+b+c+d

# Test the model... Hope it works...
coefsTEST = [0,0,2,-2,0,1,2]
time = np.linspace(0,3,300)
yTEST = model(coefsTEST,time)

plt.plot(time,yTEST)
plt.show()
