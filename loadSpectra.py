#!/bin/usr/env/python

import numpy as np
import h5py as h5
import argparse

# This file is just for loading my HDF5 files and making them python readable...

ap = argparse.ArgumentParser()
ap.add_argument("-fN",
				required=True,
				type = str,
				help = "Path/filename of hdf5 file")
ap.add_argument("-ds",
				required=False,
				default='NA',
				type = str,
				help = "Name of specific dataset to load")
args = vars(ap.parse_args())

# Some helper functinos:
def printname(name):
# This is really just for looking at every directory in the hdf5 file...
  print name

# Get filename && dataset
fName = args(['fN'])
dsName = args(['ds'])

dF = h5.File(fName,'r')

