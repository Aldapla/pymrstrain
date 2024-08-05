import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


class MRIPlotter:
  def __init__(self, images=[], FOV=[1, 1, 1], caxis=None, cmap=plt.get_cmap('Greys_r'), title=[], swap_axes=None, shape=None, next_frame_key='d', previous_frame_key='a', next_slice_key='w', previous_slice_key='s'):
    self.images = images
    self.FOV = np.array(FOV)
    self.caxis = caxis
    self.cmap = cmap
    if title == []:
      title = ['',] * len(self.images)
    else:
      self.title = title    
    self.title = title
    self.swap_axes = swap_axes
    self.shape = shape
    self._images = [im.swapaxes(0, 1) for im in self.images]
    self._FOV = self.FOV[[1, 0, 2]]
    if self.swap_axes is not None:
      self._images = [im.swapaxes(self.swap_axes[0], self.swap_axes[1]) for im in self._images]
      self._FOV[self.swap_axes[::-1]] = self._FOV[self.swap_axes]
    self.check_dimensions()
    self.next_frame_key     = next_frame_key
    self.previous_frame_key = previous_frame_key
    self.next_slice_key     = next_slice_key
    self.previous_slice_key = previous_slice_key
    # Create figures
    if self.shape is not None:
      self.fig, self.ax = plt.subplots(self.shape[0], self.shape[1])
    else:
      if len(self.images) < 4:
        self.fig, self.ax = plt.subplots(1, len(self.images))
        if len(self.images) == 1:
          self.ax = [self.ax, None]
      else:
        cols = np.ceil(np.sqrt(len(self.images))).astype(int)
        rows = np.floor(len(self.images)/cols).astype(int)
        self.fig, self.ax = plt.subplots(rows, cols)

  def check_dimensions(self):
    # Make all the images have 4 dimensions (to make all functions below generic)
    while len(self._images[0].shape) < 4:
      self._images = [im[...,np.newaxis] for im in self._images]  

  def show(self):
    # Calculate extent based on FOV
    extent = [0, self._FOV[1], 0, self._FOV[0]]

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
        image = ax.imshow(im[..., ax.slice, ax.frame], cmap=self.cmap, vmin=im.min(), vmax=im.max(), extent=extent)
      else:
        if isinstance(self.caxis[0], list):
          # Use each list contained in caxis to define minimun and maximun values
          image = ax.imshow(im[..., ax.slice, ax.frame], cmap=self.cmap, vmin=self.caxis[i][0], vmax=self.caxis[i][1], extent=extent)
        else:
          # Use caxis for all images contained in _images
          image = ax.imshow(im[..., ax.slice, ax.frame], cmap=self.cmap, vmin=self.caxis[0], vmax=self.caxis[1], extent=extent)

      # Invert axis to have y=0 at the bottom
      ax.invert_yaxis()

      # Set title to each axis
      ax.set_title(self.title[i])

      # Locating current axes 
      divider = make_axes_locatable(ax) 
        
      # creating new axes on the right 
      # side of current axes(ax). 
      # The width of cax will be 5% of ax 
      # and the padding between cax and ax 
      # will be fixed at 0.05 inch. 
      colorbar_axes = divider.append_axes("right", 
                                          size="10%", 
                                          pad=0.1) 

      # Add colorbar
      cbar = self.fig.colorbar(image, cax=colorbar_axes)
      cbar.minorticks_on()

      # self.fig.tight_layout()
      self.fig.canvas.mpl_connect('key_press_event', self.process_key)

    # Add title and show figure
    self.fig.suptitle('Slice {:d}, frame {:d}'.format(ax.slice, ax.frame))
    self.fig.tight_layout()
    plt.show()

  def previous_slice(self, ax):
    """Go to the previous slice."""
    im = ax.im
    ax.slice = (ax.slice - 1) % im.shape[2]  # wrap around using %
    ax.images[0].set_array(im[...,ax.slice,ax.frame])
    self.fig.suptitle('Slice {:d}, frame {:d}'.format(ax.slice, ax.frame))

  def next_slice(self, ax):
    """Go to the next slice."""
    im = ax.im
    ax.slice = (ax.slice + 1) % im.shape[2]
    ax.images[0].set_array(im[...,ax.slice,ax.frame])
    self.fig.suptitle('Slice {:d}, frame {:d}'.format(ax.slice, ax.frame))

  def previous_frame(self, ax):
    """Go to the previous frame."""
    im = ax.im
    ax.frame = (ax.frame - 1) % im.shape[3]  # wrap around using %
    ax.images[0].set_array(im[...,ax.slice,ax.frame])
    self.fig.suptitle('Slice {:d}, frame {:d}'.format(ax.slice, ax.frame))

  def next_frame(self, ax):
    """Go to the next frame."""
    im = ax.im
    ax.frame = (ax.frame + 1) % im.shape[3]
    ax.images[0].set_array(im[...,ax.slice,ax.frame])
    self.fig.suptitle('Slice {:d}, frame {:d}'.format(ax.slice, ax.frame))

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