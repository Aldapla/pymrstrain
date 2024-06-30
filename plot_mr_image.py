import pickle

import matplotlib.pyplot as plt
import numpy as np
import yaml

from PyMRStrain.Filters import Tukey_filter
from PyMRStrain.KSpaceTraj import Cartesian, Radial, Spiral
from PyMRStrain.Math import itok, ktoi
from PyMRStrain.Noise import add_cpx_noise
from PyMRStrain.Plotter import MRIPlotter

if __name__ == '__main__':

  # Import image kspace
  im_file = 'MRImages/FFE_V250.pkl'
  with open(im_file, 'rb') as f:
    data = pickle.load(f)

  # Extract information from data
  K = data['kspace']
  traj = data['traj']

  # Fix the direction of kspace lines measured in the opposite direction
  if isinstance(traj, Cartesian) and traj.lines_per_shot > 1:   
    # Reorder lines depending of their readout direction
    for ph in range(K.shape[1]):
      # The first line in every shot should be ordered from left to right
      if ph % traj.lines_per_shot == 0:
        ro = 1

      # Reverse orientations (only when ro=-1)
      K[::ro,ph,...] = K[::1,ph,...]

      # Reverse readout
      ro = -ro

  # Add noise to the signal
  K = itok(add_cpx_noise(ktoi(K, [0,1,2]), relative_std=0.02, mask=1), [0,1,2])

  # Kspace filtering (as the scanner would do)
  h_meas = Tukey_filter(K.shape[0], width=0.9, lift=0.3)
  h_pha  = Tukey_filter(K.shape[1], width=0.9, lift=0.3)
  h = np.outer(h_meas, h_pha)
  H = np.tile(h[:,:,np.newaxis, np.newaxis, np.newaxis], (1, 1, K.shape[2], K.shape[3], K.shape[4]))
  K_fil = H*K

  # Reconstruct image
  I = ktoi(K_fil,[0,1,2])

  # Chop if needed
  if (K.shape[0] == data['traj'].res[0]):
      I = I
  else:
      ind1 = int(np.floor((K.shape[0] - data['traj'].res[0])/2))
      ind2 = int(np.floor((K.shape[0] - data['traj'].res[0])/2)+data['traj'].res[0])
      I = I[ind1:ind2,...]

  # Show image
  M = np.abs(I[...,2,:])
  P = np.angle(I[...,2,:])
  plotter = MRIPlotter(images=[M, P])
  plotter.show()