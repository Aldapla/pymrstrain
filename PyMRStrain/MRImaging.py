import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

from PyMRStrain.Filters import Hamming_filter, Riesz_filter, Tukey_filter
from PyMRStrain.Helpers import build_idx, order
from PyMRStrain.Math import itok, ktoi
from PyMRStrain.MPIUtilities import MPI_print, MPI_rank


# EPI class for the generation of EPI-like artifacts
class EPI:
  def __init__(self, receiver_bw = 64*1000,
                echo_train_length = 1,
                off_resonance = 100,
                acq_matrix = [128,64],
                spatial_shift = 'top-down'):
    self.receiver_bw = receiver_bw
    self.echo_train_length = echo_train_length
    self.off_resonance = off_resonance
    self.acq_matrix = acq_matrix
    self.spatial_shift = spatial_shift

  # Get kspace with EPI-like artifacts
  def kspace(self,k,delta,dir,T2star=0.02):
    # kspace bandwith
    k_max = 0.5/delta

    # kspace of the input image
    m_profiles = self.acq_matrix[dir[0]]
    ph_profiles = self.acq_matrix[dir[1]]
    grid = np.meshgrid(np.linspace(-k_max,k_max,m_profiles),
                        np.linspace(-k_max,k_max,ph_profiles),
                        indexing='ij')
    ky = grid[dir[1]].flatten(order[dir[0]])
    # ky = np.sqrt(np.power(grid[dir[1]].flatten(order[dir[0]]), 2)
    #    + np.power(grid[dir[0]].flatten(order[dir[0]]), 2))

    # Parameters
    df_off = self.off_resonance                 # off-resonance frequency
    dt_esp = self.acq_matrix[1]*1.0/(2.0*self.receiver_bw) # temporal echo spacing

    # Spatial shifts
    if self.spatial_shift == 'top-down':
        dy_off = df_off*dt_esp*self.echo_train_length*delta
    elif self.spatial_shift == 'center-out':
        dy_off = 2*df_off*dt_esp*self.echo_train_length*delta

    # Acquisition times
    t = self.time_map(k,dir,dt_esp)

    # Truncation
    MTF_off = np.exp(1j*2*np.pi*dy_off*np.abs(ky),order=order[dir[0]])
    # MTF_off = np.exp(1j*2*np.pi*dy_off*np.abs(ky)*t,order=order[dir[0]])
    # MTF_off = np.exp(1j*2*np.pi*dy_off*np.abs(ky)*t/T2star,order=order[dir[0]])
    MTF = MTF_off

    return MTF*k

  # Time maps of the EPI acquisition
  def time_map(self, k, dir, temporal_echo_spacing):

    # kspace of the input image
    m_profiles = self.acq_matrix[dir[0]]
    ph_profiles = self.acq_matrix[dir[1]]

    # Acquisition times for different cartesian techniques:
    # EPI
    t_train = np.linspace(0,temporal_echo_spacing*self.echo_train_length,
                        m_profiles*self.echo_train_length)
    for i in range(self.echo_train_length):
        if (i % 2 != 0):
            a, b = i*m_profiles, (i+1)*m_profiles
            t_train[a:b] = np.flip(t_train[a:b])

    t = np.copy(t_train)
    for i in range(int(ph_profiles/self.echo_train_length)-1):
        t = np.append(t, t_train, axis=0)

    # Simpler GRE
    # delta = 0.002
    # t = np.linspace(0,dt_esp*ph_profiles,m_profiles*ph_profiles)
    # t = t.reshape(k.shape,order='F')
    # for i in range(t.shape[1]):
    #     t[:,i] += i*dt_esp + delta

    return t


class SliceProfile:
  def __init__(self, z=None, z0=0.0, delta_z=0.008, gammabar=42.58, Gz=30.0, RFShape='sinc', NbLobes=2, flip_angle=np.deg2rad(15.0)):
    self.z = z
    self.z0 = z0
    self.delta_z = delta_z
    self.gammabar = gammabar                # [MHz/T]
    self.gammabar_ = 1e+6*gammabar          # [Hz/T]
    self.Gz = Gz # dummy gradient (not really necessary to estimate the profile [yet])
    self.Gz_ = 1e-3*Gz
    self.RFShape = RFShape
    self.NbLobes = NbLobes
    self.flip_angle = flip_angle
    self.profile = self.calculate().astype(np.float32)

  def calculate(self):

    # Pulse frequency needed for the desired slice thickness
    delta_f = self.gammabar_*self.Gz_*self.delta_z

    # Angular frequency needed to excitate around z0
    omega_rf = 2.0*np.pi*self.gammabar_*self.Gz_*self.z0

    if self.RFShape == 'sinc':
      # Pulse duration and apodization
      tau = (self.NbLobes+1)*2/delta_f
      t = np.linspace(-4*tau/2, 4*tau/2, 10000)

      # RF pulse definition
      dt = t[1] - t[0]
      B1 = np.sinc(delta_f*t)*np.exp(1j*omega_rf*t)*(np.abs(t) <= tau/2)
      B1 *= self.flip_angle/(2.0*np.pi*self.gammabar_*np.abs(B1.sum())*dt)

      # Slice profile
      N = len(t)
      bandwith = 1.0/(t[1] - t[0])
      k = np.linspace(0, bandwith, N) - 0.5*bandwith
      z = k/(self.gammabar_*self.Gz_)
      p = np.abs(np.fft.fftshift(np.fft.fft((B1))))

      fig, ax = plt.subplots(1, 2, figsize=(12, 4))
      ax[0].plot(t, np.real(B1))
      ax[0].plot(t, np.imag(B1))
      ax[0].plot(t, np.abs(B1))
      ax[0].set_xlim([-tau, tau])
      ax[0].legend(['Real','Imag','Abs'])

      ax[1].plot(z, np.real(np.fft.fftshift(np.fft.fft((B1)))))
      ax[1].plot(z, np.imag(np.fft.fftshift(np.fft.fft((B1)))))
      ax[1].plot(z, np.abs(np.fft.fftshift(np.fft.fft((B1)))))
      ax[1].set_xlim([self.z0 - 2*self.delta_z, self.z0 + 2*self.delta_z])
      ax[1].legend(['Real','Imag','Abs'])
      plt.show()

      # Interpolator
      f = interp1d(z, p, kind='linear', bounds_error=False, fill_value=0.0)

    elif self.RFShape == 'hard':
      # Pulse duration and apodization
      tau = 1.0/delta_f
      t = np.linspace(-6*tau/2, 6*tau/2, 10000)

      # RF pulse definition
      dt = t[1] - t[0]
      B1 = np.exp(1j*omega_rf*t)*(np.abs(t) <= tau/2)
      B1 *= self.flip_angle/(2.0*np.pi*self.gammabar_*B1.sum()*dt)

      # Slice profile
      N = len(t)
      bandwith = 1.0/(t[1] - t[0])
      k = np.linspace(0, bandwith, N) - 0.5*bandwith
      z = k/(self.gammabar_*self.Gz_)
      p = np.abs(np.fft.fftshift(np.fft.fft((B1))))

      fig, ax = plt.subplots(1, 2, figsize=(12, 4))
      ax[0].plot(t, np.real(B1))
      ax[0].plot(t, np.imag(B1))
      ax[0].plot(t, np.abs(B1))
      ax[0].set_xlim([-tau, tau])
      ax[0].legend(['Real','Imag','Abs'])

      ax[1].plot(z, np.real(np.fft.fftshift(np.fft.fft((B1)))))
      ax[1].plot(z, np.imag(np.fft.fftshift(np.fft.fft((B1)))))
      ax[1].plot(z, np.abs(np.fft.fftshift(np.fft.fft((B1)))))
      ax[1].set_xlim([self.z0 - 4*self.delta_z, self.z0 + 4*self.delta_z])
      ax[1].legend(['Real','Imag','Abs'])
      plt.show()

      # Interpolator
      f = interp1d(z, p, kind='linear', bounds_error=False, fill_value=0.0)

    return f(self.z)