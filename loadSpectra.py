#!/bin/usr/env/python

import numpy as np
import h5py as h5

# This file is just for loading my HDF5 files and making them python readable...

fName = 'ThisIsATest.hdf5'

dF = h5.File(fName,'r')


