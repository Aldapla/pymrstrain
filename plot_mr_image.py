import pickle

import matplotlib.pyplot as plt
import numpy as np
import yaml
from sigpy.fourier import ifft, nufft_adjoint

from PyMRStrain.Filters import Tukey_filter
from PyMRStrain.Math import Rx, Ry, Rz, itok, ktoi
from PyMRStrain.Noise import add_cpx_noise
from PyMRStrain.Plotter import multi_slice_viewer
from PyMRStrain.KSpaceTraj import Cartesian, Radial, Spiral

def nufft_recon(traj, K):
  # Non-uniform fft
  x0 = traj.points[0].flatten().reshape((-1,1))
  x1 = traj.points[1].flatten().reshape((-1,1))
  x2 = traj.points[2].flatten().reshape((-1,1))
  x0 *= RES[0]//2/x0.max()
  x1 *= RES[1]//2/x1.max()
  x2 *= RES[2]//2/x2.max()
  if isinstance(traj, Cartesian):
    dcf = 1.0
  else:
    dcf = (x0**2 + x1**2 + x2**2)**0.5
  kxkykz = np.concatenate((x0, x1, x2), axis=1)   # flattened kspace coordinates
  y = K.flatten().reshape((-1,1))          # flattened kspace measures
  image = nufft_adjoint(y*dcf, kxkykz, RES)  # inverse nufft
  return image

if __name__ == '__main__':

  # Import imaging parameters
  with open('PARAMETERS.yaml') as file:
    pars = yaml.load(file, Loader=yaml.FullLoader)

  # Imaging parameters
  FOV = np.array(pars["Imaging"]["FOV"])
  RES = np.array(pars["Imaging"]["RES"])
  T2star = pars["Imaging"]["T2star"]/1000.0
  VENC = pars["Imaging"]["VENC"]*100.0
  OFAC = pars["Imaging"]["OVERSAMPLING"]
  dt = pars["Imaging"]["TIMESPACING"]

  # Hardware parameters
  G_sr  = pars["Hardware"]["G_sr"]
  G_max = pars["Hardware"]["G_max"]
  r_BW  = pars["Hardware"]["r_BW"]

  # Imaging orientation parameters
  theta_x = np.deg2rad(pars["Formatting"]["theta_x"])
  theta_y = np.deg2rad(pars["Formatting"]["theta_y"])
  theta_z = np.deg2rad(pars["Formatting"]["theta_z"])
  LOCs = np.array(pars["Formatting"]["LOC"])

  # Build rotation matrix and get slice location
  MPS_ori = Rz(theta_z)@Rx(theta_x)@Ry(theta_y)
  LOC = LOCs

  # Sequence
  seq  = 'EPI'
  lps = pars[seq]['LinesPerShot']
  traj = Cartesian(FOV=FOV, res=RES, oversampling=OFAC, lines_per_shot=lps, VENC=VENC, MPS_ori=MPS_ori, LOC=LOC, receiver_bw=r_BW, Gr_max=G_max, Gr_sr=G_sr, plot_seq=False)

  # Import generated data
  im = 'MRImages/{:s}_V{:.0f}.pkl'.format(seq, VENC)
  with open(im, 'rb') as f:
    data = pickle.load(f)

  # Extract information from data
  K = data['kspace']

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
  if seq in 'EPI':
    I = np.zeros(K_fil[::2,...].shape, dtype=np.complex64)
    for v in range(I.shape[3]):
      for fr in [0,1,2]:#range(I.shape[4]):
        I[...,v,fr] = nufft_recon(traj, K_fil[...,v,fr])
  else:
    I = ktoi(K_fil[::2,...],[0,1,2])

  # Show figure
  for fr in range(K.shape[-1]):
    for i in [2]:
      M = np.abs(I[:,:,:,i,fr])
      P = np.angle(I[:,:,:,i,fr])
      multi_slice_viewer([M, P])
      plt.show()
