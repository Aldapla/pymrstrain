import matplotlib.pyplot as plt
import numpy as np


class MRIPlotter:
  def __init__(self, images=[], caxis=None, cmap=plt.get_cmap('Greys_r'), title='Slice', swap_axes=None, shape=None, next_frame_key='d', previous_frame_key='a', next_slice_key='w', previous_slice_key='s'):
    self.images = images
    self.caxis = caxis
    self.cmap = cmap
    self.title = title
    self.swap_axes = swap_axes
    self.shape = shape
    self._images = [im.swapaxes(0, 1) for im in self.images]
    if self.swap_axes is not None:
      self._images = [im.swapaxes(self.swap_axes[0], self.swap_axes[1]) for im in self._images]
    self.check_dimensions()
    self.next_frame_key     = next_frame_key
    self.previous_frame_key = previous_frame_key
    self.next_slice_key     = next_slice_key
    self.previous_slice_key = previous_slice_key
    # Create figures
    if self.shape is not None:
      self.fig, self.ax = plt.subplots(self.shape[0], self.shape[1], layout='constrained')
    else:
      if len(self.images) < 4:
        self.fig, self.ax = plt.subplots(1, len(self.images), layout='constrained')
        if len(self.images) == 1:
          self.ax = [self.ax, None]
      else:
        cols = np.ceil(np.sqrt(len(self.images))).astype(int)
        rows = np.floor(len(self.images)/cols).astype(int)
        self.fig, self.ax = plt.subplots(rows, cols, layout='tight')

  def check_dimensions(self):
    # Make all the images have 4 dimensions (to make all functions below generic)
    while len(self._images[0].shape) < 4:
      self._images = [im[...,np.newaxis] for im in self._images]  

  def show(self):
    # Remove key conflicts with defined keys for slices and frames
    self.remove_keymap_conflicts({self.next_slice_key, self.previous_slice_key, self.next_frame_key, self.previous_slice_key})
    
    # Plot volumes
    flat_ax = self.ax.flatten() # flattened axes
    for (i, (im, ax)) in enumerate(zip(self._images, flat_ax)):
      # Add image, index, and frame to each axis (ax)
      ax.im = im
      ax.slice = 0
      ax.frame = 0

      if self.caxis is None:
        # Use minimun and maximun image values to define caxis
        image = ax.imshow(im[..., ax.slice, ax.frame], cmap=self.cmap, vmin=im.min(), vmax=im.max())
      else:
        if isinstance(self.caxis[0], list):
          # Use each list contained in caxis to define minimun and maximun values
          image = ax.imshow(im[..., ax.slice, ax.frame], cmap=self.cmap, vmin=self.caxis[i][0], vmax=self.caxis[i][1])
        else:
          # Use caxis for all images contained in _images
          image = ax.imshow(im[..., ax.slice, ax.frame], cmap=self.cmap, vmin=self.caxis[0], vmax=self.caxis[1])

      # Invert axis to have y=0 at the bottom
      ax.invert_yaxis()

      # Set title to each axis
      ax.set_title('Slice {:d}, frame{:d}'.format(ax.slice, ax.frame))

      # Add colorbar
      cbar = self.fig.colorbar(image, location='bottom', orientation='horizontal', pad=0.06)
      cbar.minorticks_on()

      # self.fig.tight_layout()
      self.fig.canvas.mpl_connect('key_press_event', self.process_key)
    
    plt.show()

  def previous_slice(self, ax):
    """Go to the previous slice."""
    im = ax.im
    ax.slice = (ax.slice - 1) % im.shape[2]  # wrap around using %
    ax.images[0].set_array(im[...,ax.slice,ax.frame])
    ax.set_title('Slice {:d}, frame {:d}'.format(ax.slice, ax.frame))

  def next_slice(self, ax):
    """Go to the next slice."""
    im = ax.im
    ax.slice = (ax.slice + 1) % im.shape[2]
    ax.images[0].set_array(im[...,ax.slice,ax.frame])
    ax.set_title('Slice {:d}, frame {:d}'.format(ax.slice, ax.frame))

  def previous_frame(self, ax):
    """Go to the previous frame."""
    im = ax.im
    ax.frame = (ax.frame - 1) % im.shape[3]  # wrap around using %
    ax.images[0].set_array(im[...,ax.slice,ax.frame])
    ax.set_title('Slice {:d}, frame {:d}'.format(ax.slice, ax.frame))

  def next_frame(self, ax):
    """Go to the next frame."""
    im = ax.im
    ax.frame = (ax.frame + 1) % im.shape[3]
    ax.images[0].set_array(im[...,ax.slice,ax.frame])
    ax.set_title('Slice {:d}, frame {:d}'.format(ax.slice, ax.frame))

  def remove_keymap_conflicts(self, new_keys_set):
    for prop in plt.rcParams:
      if prop.startswith('keymap.'):
        keys = plt.rcParams[prop]
        remove_list = set(keys) & new_keys_set
        for key in remove_list:
          keys.remove(key)

  def process_key(self,event):
    fig = event.canvas.figure
    for ax in fig.axes[:len(self._images)]:
      if event.key == 's':
          self.previous_slice(ax)
      elif event.key == 'w':
          self.next_slice(ax)
      elif event.key == 'a':
          self.previous_frame(ax)
      elif event.key == 'd':
          self.next_frame(ax)
    fig.canvas.draw()