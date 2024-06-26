import os
import pickle
import time
from pathlib import Path

import matplotlib.pyplot as plt
import meshio
import numpy as np
import yaml
from Fem import massAssemble
from FlowToImage import FlowImage3D

from PyMRStrain.IO import scale_data
from PyMRStrain.KSpaceTraj import Cartesian
from PyMRStrain.Math import Rx, Ry, Rz, itok, ktoi
from PyMRStrain.MPIUtilities import MPI_comm, MPI_rank, gather_image
from PyMRStrain.MRImaging import SliceProfile
from PyMRStrain.Phantom import femPhantom

if __name__ == '__main__':

  # Preview partial results
  preview = True

  # Import imaging parameters
  with open('PARAMETERS.yaml') as file:
    pars = yaml.load(file, Loader=yaml.FullLoader)

  # Imaging parameters
  FOV = np.array(pars['Imaging']['FOV'])
  RES = np.array(pars['Imaging']['RES'])
  T2star = pars['Imaging']['T2star']/1000.0
  VENC = np.array(pars['Imaging']['VENC'])
  OFAC = pars['Imaging']['OVERSAMPLING']

  # Hardware parameters
  G_sr  = pars['Hardware']['G_sr']
  G_max = pars['Hardware']['G_max']
  r_BW  = pars['Hardware']['r_BW']

  # Imaging orientation paramters
  theta_x = np.deg2rad(pars['Formatting']['theta_x'])
  theta_y = np.deg2rad(pars['Formatting']['theta_y'])
  theta_z = np.deg2rad(pars['Formatting']['theta_z'])
  MPS_ori = Rz(theta_z)@Rx(theta_x)@Ry(theta_y)
  LOC = np.array(pars['Formatting']['LOC'])

  # Imaging sequences for the image generation
  sequences = ['EPI'] #['FFE', 'EPI']

  # Iterate over sequences
  for seq in sequences:

    # Navier-Stokes simulation data to be used
    path_NS = 'phantom/phantom.xdmf'

    # Create FEM phantom object
    phantom = femPhantom(path=path_NS, vel_label='velocity', scale_factor=0.01)

    # Assemble mass matrix for integrals (just once)
    M = massAssemble(phantom.mesh['elems'], phantom.mesh['nodes'])

    # Field inhomogeneity
    x =  phantom.mesh['nodes']
    gammabar = 1.0e+6*42.58 # Hz/T 
    delta_B0 = x[:,0] + x[:,1] + x[:,2]  # spatial distribution
    delta_B0 /= np.abs(delta_B0).max()  # normalization
    delta_B0 *= 1.5*1e-6  # scaling (1 ppm of 1.5T)        
    delta_B0 *= 0.0  # additional scaling (just for testing)
    gamma_x_delta_B0 = 2*np.pi*gammabar*delta_B0

    # Path to export the generated data
    export_path = Path('MRImages/{:s}_V{:.0f}.pkl'.format(seq, 100.0*VENC))

    # Make sure the directory exist
    os.makedirs(str(export_path.parent), exist_ok=True)

    # Generate kspace trajectory
    lps = pars[seq]['LinesPerShot']
    traj = Cartesian(FOV=FOV, res=RES, oversampling=OFAC, lines_per_shot=lps, VENC=VENC, MPS_ori=MPS_ori, LOC=LOC, receiver_bw=r_BW, Gr_max=G_max, Gr_sr=G_sr, plot_seq=False)

    # Translate phantom to obtain the desired slice location
    nodes = (phantom.mesh['nodes'] - traj.LOC)@traj.MPS_ori

    # Slice profile
    # gammabar = 1.0e+6*42.58 # Hz/T 
    # G_z = 1.0e-3*30.0       # [T/m]
    delta_z = FOV[2]        # [m]
    # delta_g = gammabar*G_z*delta_z
    profile = SliceProfile(z=nodes[:,2], delta_z=delta_z).profile
    # profile = np.ones_like(nodes[:,2])

    # Print echo time
    if MPI_rank==0: print('Echo time = {:.1f} ms'.format(1000.0*traj.echo_time))

    # kspace array
    ro_samples = traj.ro_samples
    ph_samples = traj.ph_samples
    slices = traj.slices
    K = np.zeros([ro_samples, ph_samples, slices, 3, phantom.Nfr], dtype=np.complex64)

    # List to store how much is taking to generate one volume
    times = []

    # Iterate over cardiac phases
    for fr in range(phantom.Nfr):

      # Read velocity data in frame fr
      phantom.read_data(fr)
      velocity = phantom.velocity@traj.MPS_ori

      # Generate 4D flow image
      if MPI_rank == 0: print('Generating frame {:d}'.format(fr))
      t0 = time.time()
      K[traj.local_idx,:,:,:,fr] = FlowImage3D(MPI_rank, M, traj.local_points, traj.local_times, velocity, nodes, gamma_x_delta_B0, T2star, VENC, profile)
      t1 = time.time()
      times.append(t1-t0)

      # Save kspace for debugging purposes
      if preview:
        K_copy = np.copy(K)
        K_copy = gather_image(K_copy)
        if MPI_rank==0:
          with open(str(export_path), 'wb') as f:
            pickle.dump({'kspace': K_copy, 'MPS_ori': MPS_ori, 'LOC': LOC, 'traj': traj}, f)

      # Synchronize MPI processes
      print(np.array(times).mean())
      MPI_comm.Barrier()

    # Show mean time that takes to generate each 3D volume
    print(np.array(times).mean())

    # Gather results
    K = gather_image(K)

    # Export generated data
    if MPI_rank==0:
      with open(str(export_path), 'wb') as f:
        pickle.dump({'kspace': K, 'MPS_ori': MPS_ori, 'LOC': LOC, 'traj': traj}, f)