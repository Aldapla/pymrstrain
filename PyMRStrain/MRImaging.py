import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import DOP853, RK23, RK45
from scipy.interpolate import interp1d

from PyMRStrain.Filters import Hamming_filter, Riesz_filter, Tukey_filter
from PyMRStrain.Helpers import build_idx, order
from PyMRStrain.KSpaceTraj import Gradient
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


# class SliceProfileBase:
#   def __init__(self):


class SliceProfile:
  def __init__(self, z0=0.0, delta_z=0.008, gammabar=42.58, Gz=30.0, RFShape='sinc', NbLeftLobes=2, NbRightLobes=2, alpha=0.46, flip_angle=np.deg2rad(15.0), NbPoints=1000, plot=True):
    self.z0 = z0
    self.delta_z = delta_z
    self.gammabar = gammabar                # [MHz/T]
    self.gammabar_ = 1e+6*gammabar          # [Hz/T]
    self.Gz = Gz # dummy gradient (not really necessary to estimate the profile [yet])
    self.Gz_ = 1e-3*Gz
    self.RFShape = RFShape
    self.NbLeftLobes = NbLeftLobes
    self.NbRightLobes = NbRightLobes
    self.alpha = alpha # 0.46 for Hamming and 0.5 for Hanning
    self.flip_angle = flip_angle
    self.NbPoints = NbPoints
    self.plot = plot
    self.interp_profile = self.calculate()

  def calculate(self):

    # Pulse frequency needed for the desired slice thickness
    delta_f = self.gammabar_*self.Gz_*self.delta_z

    # Angular frequency needed to excitate around z0
    omega_rf = 2.0*np.pi*self.gammabar_*self.Gz_*self.z0

    if self.RFShape == 'sinc':
      # Pulse duration and time window
      tau_l = (self.NbLeftLobes+1)*2/delta_f
      tau_r = (self.NbRightLobes+1)*2/delta_f
      tau = np.max([tau_l, tau_r])
      t = np.linspace(-4*tau/2, 4*tau/2, self.NbPoints)

      # RF pulse definition
      dt = t[1] - t[0]
      B1e = np.sinc(delta_f*t)*(t >= -tau_l/2)*(t <= tau_r/2)
      B1  = B1e*np.exp(1j*omega_rf*t)
      B1 *= self.flip_angle/(2.0*np.pi*self.gammabar_*B1e.sum()*dt)

      # Slice profile
      N = len(t)
      bandwith = 1.0/(t[1] - t[0])
      k = np.linspace(0, bandwith, N) - 0.5*bandwith
      z = k/(self.gammabar_*self.Gz_)
      p = np.abs(np.fft.fftshift(np.fft.fft((B1))))

    elif self.RFShape == 'hard':
      # Pulse duration and window
      tau = 1.0/delta_f
      t = np.linspace(-6*tau/2, 6*tau/2, self.NbPoints)

      # RF pulse definition
      dt = t[1] - t[0]
      B1e = 1.0*(np.abs(t) <= tau/2)
      B1  = B1e*np.exp(1j*omega_rf*t)
      B1 *= self.flip_angle/(2.0*np.pi*self.gammabar_*B1e.sum()*dt)

      # Slice profile
      N = len(t)
      bandwith = 1.0/(t[1] - t[0])
      k = np.linspace(0, bandwith, N) - 0.5*bandwith
      z = k/(self.gammabar_*self.Gz_)
      p = np.abs(np.fft.fftshift(np.fft.fft((B1))))

    elif self.RFShape == 'apodized_sinc':
      # Maximun number of lobes
      N = np.max([self.NbLeftLobes, self.NbRightLobes])

      # Pulse duration and time window
      tau_l = (self.NbLeftLobes+1)*2/delta_f
      tau_r = (self.NbRightLobes+1)*2/delta_f
      tau = np.max([tau_l, tau_r])
      t = np.linspace(-4*tau/2, 4*tau/2, self.NbPoints)

      # RF pulse definition
      dt = t[1] - t[0]
      B1e = (1/delta_f)*((1-self.alpha) + self.alpha*np.cos(np.pi*delta_f*t/N))*np.sin(np.pi*delta_f*t)/(np.pi*t)*(t >= -tau_l/2)*(t <= tau_r/2)
      B1  = B1e*np.exp(1j*omega_rf*t)
      B1 *= self.flip_angle/(2.0*np.pi*self.gammabar_*B1e.sum()*dt)

      # Slice profile
      N = len(t)
      bandwith = 1.0/(t[1] - t[0])
      k = np.linspace(0, bandwith, N) - 0.5*bandwith
      z = k/(self.gammabar_*self.Gz_)
      p = np.abs(np.fft.fftshift(np.fft.fft((B1))))

    if self.plot:
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
    interp_profile = interp1d(z, p, kind='linear', bounds_error=False, fill_value=0.0)

    return interp_profile


class BlochSliceProfile:
  def __init__(self, z0=0.0, delta_z=0.008, gammabar=42.58, Gz=1.0, RFShape='sinc', NbLeftLobes=2, NbRightLobes=2, alpha=0.46, flip_angle=np.deg2rad(10.0), dt=1e-4, NbPoints=150, plot=False, small_angle=False, refocusing_area_frac=0.5):
    self.z0 = z0
    self.delta_z = delta_z
    self.gammabar = gammabar                # [MHz/T]
    self.gammabar_ = 1e+6*gammabar          # [Hz/T]
    self.Gz = Gz # dummy gradient (not really necessary to estimate the profile [yet])
    self.Gz_ = 1.0e-3*Gz
    self.RFShape = RFShape
    self.NbLeftLobes = NbLeftLobes
    self.NbRightLobes = NbRightLobes
    self.alpha = alpha # 0.46 for Hamming and 0.5 for Hanning
    self.flip_angle = flip_angle
    self.NbPoints = NbPoints
    self.plot = plot
    self.z_rk = self.z0 # temporary assignment
    self.dt = dt
    self.small_angle = small_angle
    self.refocusing_area_frac = refocusing_area_frac
    if self.RFShape == 'sinc':
      self.RFPulse = self._rf_sinc
    elif self.RFShape == 'apodized_sinc':
      self.RFPulse = self._rf_apodized_sinc
    elif self.RFShape == 'hard':
      self.RFPulse = self._rf_hard
    self.interp_profile = self.calculate()

  def bloch(self, t, M):
    # Gyromagnetic constant
    gamma = 2.0*np.pi*self.gammabar_

    # Frequency offset
    dw = gamma*1e-03*self.ss_gradient(1000.0*t)*(self.z_rk - self.z0)
    # dw = gamma*self.Gz_*(self.z_rk - self.z0)

    # Bloch equations
    if self.small_angle:
      dMxdt = dw*M[1]
      dMydt = gamma*self.B1e_norm(t)*M[2] - dw*M[0]
      dMzdt = -gamma*self.B1e_norm(t)*M[1]*0.0
    else:
      dMxdt = dw*M[1]
      dMydt = gamma*self.B1e_norm(t)*M[2] - dw*M[0]
      dMzdt = -gamma*self.B1e_norm(t)*M[1]

    return np.array([dMxdt, dMydt, dMzdt]).reshape((3,))

  def ss_gradient(self, t):
    return self.g1.evaluate(t) - self.g2.evaluate(t)

  def _rf_sinc(self, t):
    # Pulse frequency needed for the desired slice thickness
    delta_f = self.gammabar_*self.Gz_*self.delta_z

    # Pulse duration and time window
    tau_l = (self.NbLeftLobes+1)*2/delta_f
    tau_r = (self.NbRightLobes+1)*2/delta_f

    # RF pulse definition
    B1e = np.sinc(delta_f*t)*(t >= -tau_l/2)*(t <= tau_r/2)

    return B1e

  def _rf_apodized_sinc(self, t):
    # Pulse frequency needed for the desired slice thickness
    delta_f = self.gammabar_*self.Gz_*self.delta_z

    # Pulse duration and time window
    tau_l = (self.NbLeftLobes+1)*2/delta_f
    tau_r = (self.NbRightLobes+1)*2/delta_f

    # Maximun number of lobes
    N = np.max([self.NbLeftLobes, self.NbRightLobes])

    # RF pulse definition
    B1e = (1/delta_f)*((1-self.alpha) + self.alpha*np.cos(np.pi*delta_f*t/N))*np.sinc(delta_f*t)*(t >= -tau_l/2)*(t <= tau_r/2)

    return B1e

  def _rf_hard(self, t):
    # Pulse frequency needed for the desired slice thickness
    delta_f = self.gammabar_*self.Gz_*self.delta_z

    # Pulse duration and time window
    tau_l = (self.NbLeftLobes+1)*2/delta_f
    tau_r = (self.NbRightLobes+1)*2/delta_f

    # RF pulse definition
    B1e = 1.0*(t >= -tau_l/2)*(t <= tau_r/2)

    return B1e

  def B1e_norm(self, t):
    return self.RFPulse(t)*self.flip_angle_factor

  def calculate(self, y0=np.array([0,0,1]).reshape((3,))):

    # Pulse frequency needed for the desired slice thickness
    delta_f = self.gammabar_*self.Gz_*self.delta_z

    # RF pulse bounds
    tau_l = (self.NbLeftLobes+1)*2/delta_f
    tau_r = (self.NbRightLobes+1)*2/delta_f

    # Create slice selection gradient objects
    g1 = Gradient(G=self.Gz, slope=0.0, lenc=0.5*(tau_l + tau_r)*1000.0, t_ref=-1000*tau_l/2)
    g2 = Gradient(G=self.Gz, slope=0.0, t_ref=g1.timings[-1])
    # g2.calculate_area(0.5*g1.G_*g1.lenc_ + 0.5*g1.G_*g1.slope_)
    g2.calculate_area(self.refocusing_area_frac*(g1.G_*g1.lenc_ + g1.G_*g1.slope_))

    # Fix timings
    g1.t_ref -= g1.slope
    g1.timings, g1.amplitudes, _ = g1.group_timings()
    g2.t_ref -= g1.slope
    g2.timings, g2.amplitudes, _ = g2.group_timings()
    self.g1 = g1
    self.g2 = g2

    # Integration bounds
    t0 = g1.timings[0]/1000.0
    t_bound = g2.timings[-1]/1000.0   

    # Calculate factor to accoundt for flip angle
    t = np.linspace(t0, t_bound, int((t_bound - t0)/self.dt))
    self.flip_angle_factor = self.flip_angle/(2.0*np.pi*self.gammabar_*self.RFPulse(t).sum()*self.dt)

    # Slice positions
    z_min = self.z0 - 2*self.delta_z
    z_max = self.z0 + 2*self.delta_z

    z_arr = np.linspace(z_min, z_max, self.NbPoints)
    M = np.zeros([3, len(z_arr)])
    for (i, z) in enumerate(z_arr):

      # Solve
      self.z_rk = z
      solver = RK45(self.bloch, t0, y0, t_bound, vectorized=True, first_step=self.dt, max_step=100.0*self.dt)

      # collect data
      t  = []
      B1 = []
      while solver.status not in ['finished','failed']:
        # get solution step state
        solver.step()
        t.append(solver.t)
        B1.append(self.B1e_norm(t[-1]))

      t  = np.array(t)
      B1 = np.array(B1)
      M[:,i] = solver.y

    if self.plot:
      fig, ax = plt.subplots(1, 3, figsize=(12, 4))
      ax[0].plot(t, B1)
      ax[0].legend(['B1'])
      axt = ax[0].twinx()
      axt.plot(g1.timings/1000, g1.amplitudes)
      axt.plot(g2.timings/1000, -g2.amplitudes)
      axt.set_xlim([t0, t_bound])
      axt.legend(['g1','g2'])

      ax[1].plot(z_arr, M[0,:])
      ax[1].plot(z_arr, M[1,:])
      ax[1].plot(z_arr, np.abs(M[0,:] + 1j*M[1,:]))
      ax[1].legend(['Mx','My','Mxy'])

      ax[2].plot(z_arr, M[2,:])
      ax[2].legend(['Mz'])
      plt.tight_layout()
      plt.show()

    # Interpolator
    p = np.abs(M[0,:] + 1j*M[1,:])
    interp_profile = interp1d(z_arr, p, kind='linear', bounds_error=False, fill_value=0.0)

    return interp_profile