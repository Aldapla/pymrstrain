import os
import pickle
import time

import cupy as cp
import matplotlib.pyplot as plt
import meshio
import numpy as np
import yaml
from cupyx.scipy.sparse import csr_matrix as cp_csr_matrix
from Fem import massAssemble

from PyMRStrain.IO import scale_data
from PyMRStrain.KSpaceTraj import Cartesian
from PyMRStrain.Math import Rx, Ry, Rz, itok, ktoi
from PyMRStrain.MPIUtilities import MPI_comm, MPI_rank, gather_image
from PyMRStrain.MRImaging import SliceProfile
from PyMRStrain.Phantom import femPhantom

def FlowImage3D(M, kxyz, t ,v, r0, gamma_x_delta_B0, T2, VENC, profile):
  # Number of kspace lines/spokes/interleaves
  nb_lines = kxyz[0].shape[1] # kxy[0].cols()
  
  # Number of measurements in the readout direction
  nb_meas = kxyz[0].shape[0] # kxy[0].rows()

  # Number of measurements in the kz direction
  nb_kz = kxyz[0].shape[2]

  # Number of spins
  nb_spins = r0.shape[0]

  # Get the equivalent gradient needed to go from the center of the kspace
  # to each location
  kx = 2.0 * np.pi * kxyz[0]
  ky = 2.0 * np.pi * kxyz[1]
  kz = 2.0 * np.pi * kxyz[2]

  # Copy blood position to estimate the current position using the approximation r(t0+dt) = r0 + v0*dt
  r = cp.asarray( np.zeros([nb_spins, 3], dtype=np.float32) , dtype=cp.float32)

  # Kspace and Fourier exponential
  Mxy = 1.0e+3 * nb_spins * cp.exp(1j * np.pi / VENC * v) * profile
  fe_xy = cp.asarray( np.zeros([nb_spins, 1], dtype=np.float32), dtype=cp.float32)
  fe = cp.asarray( np.zeros([nb_spins, 1], dtype=np.complex64), dtype=cp.complex64)
  phi_off = cp.asarray( np.zeros([nb_spins, 1], dtype=np.float32), dtype=cp.float32)

  # kspace
  kspace = cp.asarray( np.zeros([nb_meas, nb_lines, nb_kz, 3], dtype=np.complex64) , dtype=cp.complex64)

  # T2* decay
  T2_decay = cp.exp(-t / T2)

  # Iterate over kspace measurements/kspace points
  for j in range(nb_lines):

    # Debugging
    print("  ky location ", j)

    # Iterate over slice kspace measurements
    for i in range(nb_meas):

      # Update blood position at time t(i,j)
      r[:,:] = r0 + v*t[i,j]

      # Update off-resonance phase
      phi_off[:,0] = gamma_x_delta_B0*t[i,j]

      for k in range(nb_kz):

        # Update Fourier exponential
        fe[:,0] = cp.exp(1j * (-r[:,0] * kx[i,j,k] - r[:,1] * ky[i,j,k] - r[:,2] * kz[i,j,k] - phi_off[:,0]))

        # Calculate k-space values, add T2* decay, and assign value to output array
        for l in range(3):
          kspace[i,j,k,l] = M.dot(Mxy[:,l]).dot(fe[:,0]) * T2_decay[i,j]

  return kspace

if __name__ == '__main__':

  stream = cp.cuda.stream.Stream(non_blocking=True)
  cp.show_config()

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
  sequences = ['FFE', 'EPI']

  # Iterate over sequences
  for seq in sequences:

    # Navier-Stokes simulation data to be used
    path_NS = 'phantom/phantom.xdmf'

    # Create FEM phantom object
    phantom = femPhantom(path=path_NS, vel_label='velocity', scale_factor=0.01)

    # Assemble mass matrix for integrals (just once)
    M = cp_csr_matrix(massAssemble(phantom.mesh['elems'], phantom.mesh['nodes']))

    # Field inhomogeneity
    x =  phantom.mesh['nodes']
    gammabar = 1.0e+6*42.58 # Hz/T 
    delta_B0 = (x[:,0] + x[:,1] + x[:,2]) # spatial distribution
    delta_B0 /= np.abs(delta_B0).max()    # normalization
    delta_B0 *= 1.5*1e-6                  # scaling (1 ppm of 1.5T)        
    delta_B0 *= 0.0  # additional scaling (just for testing)
    gamma_x_delta_B0 = cp.asarray(2*np.pi*gammabar*delta_B0, dtype=cp.float32)

    # Path to export the generated data
    im_folder = 'MRImages/'

    # Make sure the directory exist
    if MPI_rank==0:
      os.makedirs(im_folder, exist_ok=True)

    # Generate kspace trajectory
    lps = pars[seq]['LinesPerShot']
    traj = Cartesian(FOV=FOV, res=RES, oversampling=OFAC, lines_per_shot=lps, VENC=VENC, MPS_ori=MPS_ori, LOC=LOC, receiver_bw=r_BW, Gr_max=G_max, Gr_sr=G_sr, plot_seq=False)

    # Convert trajectory numpy arrays to cupy arrays
    traj.points = (cp.asarray(traj.points[0], dtype=cp.float32), 
                  cp.asarray(traj.points[1], dtype=cp.float32), 
                  cp.asarray(traj.points[2], dtype=cp.float32))
    traj.times  = cp.asarray(traj.times, dtype=cp.float32)

    # Translate phantom to obtain the desired slice location
    nodes = cp.asarray((phantom.mesh['nodes'] - traj.LOC)@traj.MPS_ori, dtype=cp.float32)

    # Slice profile
    # gammabar = 1.0e+6*42.58 # Hz/T 
    # G_z = 1.0e-3*30.0       # [T/m]
    delta_z = FOV[2]        # [m]
    # delta_g = gammabar*G_z*delta_z
    profile = cp.asarray(np.tile(SliceProfile(z=nodes[:,2], delta_z=delta_z).profile[:, np.newaxis], (1, 3)), dtype=cp.float32)
    # profile = np.ones_like(nodes[:,2])

    # Print echo time
    if MPI_rank==0: print('Echo time = {:.1f} ms'.format(1000.0*traj.echo_time))

    # kspace array
    ro_samples = traj.ro_samples
    ph_samples = traj.ph_samples
    slices = traj.slices
    K = cp.asarray(np.zeros([ro_samples, ph_samples, slices, 3, phantom.Nfr], dtype=np.complex64), dtype=cp.complex64)

    # List to store how much is taking to generate one volume
    times = []

    # Iterate over cardiac phases
    for fr in range(phantom.Nfr):

      # Read velocity data in frame fr
      phantom.read_data(fr)
      velocity = cp.asarray(phantom.velocity@traj.MPS_ori, dtype=cp.float32)

      # Generate 4D flow image
      if MPI_rank == 0: print('Generating frame {:d}'.format(fr))
      t0 = time.time()
      K[traj.local_idx,:,:,:,fr] = FlowImage3D(M, traj.points, traj.times, velocity, nodes, gamma_x_delta_B0, T2star, VENC, profile)
      t1 = time.time()
      times.append(t1-t0)

      # Save kspace for debugging purposes
      if preview:
        K_copy = np.copy(K)
        K_copy = gather_image(K_copy)
        if MPI_rank==0:
          np.save(export_path, K_copy)

      # Synchronize MPI processes
      print(np.array(times).mean())
      MPI_comm.Barrier()

    # Show mean time that takes to generate each 3D volume
    print(np.array(times).mean())

    # Gather results
    K = gather_image(K)

    # Export generated data
    if MPI_rank==0:
      # K_scaled = scale_data(K, mag=False, real=True, imag=True, dtype=np.uint64)
      np.save(export_path, K)

  stream.synchronize()
