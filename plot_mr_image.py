import pickle

import matplotlib.pyplot as plt
import numpy as np
import yaml

from PyMRStrain.Filters import Tukey_filter
from PyMRStrain.KSpaceTraj import Cartesian, Radial, Spiral
from PyMRStrain.Math import itok, ktoi
from PyMRStrain.Noise import add_cpx_noise
from PyMRStrain.Plotter import multi_slice_viewer

if __name__ == '__main__':

  # Import image kspace
  im_file = 'MRImages/FFE_V250.pkl'
  with open(im_file, 'rb') as f:
    data = pickle.load(f)

  # Extract information from data
  K = data['kspace']
  MPS_ori = data['MPS_ori']
  LOC = data['LOC']
  traj = data['traj']
  RES = traj.res
  print(traj.points[0].min(), traj.points[0].max(), print(1.0/(traj.points[0].max()-traj.points[0].min())))
  print(traj.points[1].min(), traj.points[1].max(), print(1.0/(traj.points[1].max()-traj.points[1].min())))
  print(traj.points[2].min(), traj.points[2].max(), print(1.0/(traj.points[2].max()-traj.points[2].min())))

  # # Zero padding in the dimensions with even measurements to avoid shifts in 
  # # the image domain
  # if RES[0] % 2 == 0:
  #   pad_width = ((0, 1), (0, 0), (0, 0), (0, 0), (0, 0))
  #   K = np.pad(K, pad_width, mode='constant')
  #   RES[0] += 1
  # if RES[1] % 2 == 0:
  #   pad_width = ((0, 0), (0, 1), (0, 0), (0, 0), (0, 0))
  #   K = np.pad(K, pad_width, mode='constant')
  #   RES[1] += 1
  # if RES[2] % 2 == 0:
  #   pad_width = ((0, 0), (0, 0), (0, 1), (0, 0), (0, 0))
  #   K = np.pad(K, pad_width, mode='constant')
  #   RES[2] += 1

  # # Add noise to the signal
  # K = itok(add_cpx_noise(ktoi(K, [0,1,2]), relative_std=0.02, mask=1), [0,1,2])

  # Kspace filtering (as the scanner would do)
  h_meas = Tukey_filter(K.shape[0], width=0.9, lift=0.3)
  h_pha  = Tukey_filter(K.shape[1], width=0.9, lift=0.3)
  h = np.outer(h_meas, h_pha)
  H = np.tile(h[:,:,np.newaxis, np.newaxis, np.newaxis], (1, 1, K.shape[2], K.shape[3], K.shape[4]))
  K_fil = H*K

  # Fix the direction of kspace lines measured in the opposite direction
  if isinstance(traj, Cartesian) and traj.lines_per_shot > 1:   
    # Reorder lines depending of their readout direction
    for ph in range(K_fil.shape[1]):
      # The first line in every shot should be ordered from left to right
      if ph % traj.lines_per_shot == 0:
        ro = 1

      # Reverse orientations (only when ro=-1)
      K_fil[::ro,ph,...] = K_fil[::1,ph,...]

      # Reverse readout
      ro = -ro

  # Reconstruct image
  I = ktoi(K_fil[::2,...],[0,1,2])

  # Show figure
  for i in range(I.shape[3]):
    M = np.abs(I[...,i,:])
    P = np.angle(I[...,i,:])
    multi_slice_viewer([M, P])
    plt.show()
