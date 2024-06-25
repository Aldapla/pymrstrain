import matplotlib.pyplot as plt
import numpy as np

def remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:
        if prop.startswith('keymap.'):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)

def multi_slice_viewer(volumes, caxis=None):
    remove_keymap_conflicts({'w', 's', 'a', 'd'})
    volumes = [np.transpose(volume, (1,0,2,3)) for volume in volumes]
    fig, ax = plt.subplots(1, len(volumes))
    if len(volumes) == 1:
      ax = [ax, None]
    
    # Plot volumes
    for (i, volume) in enumerate(volumes):
      ax[i].volume = volume
      ax[i].index = 0
      ax[i].frame = 0
      if caxis==None:
        ax[i].imshow(volume[...,ax[i].index,ax[i].frame], cmap=plt.get_cmap('Greys_r'), vmin=volume.min(), vmax=volume.max())
      else:
        if isinstance(caxis[0], list):
          ax[i].imshow(volume[...,ax[i].index,ax[i].frame], cmap=plt.get_cmap('Greys_r'), vmin=caxis[i][0], vmax=caxis[i][1])
        else:
          ax[i].imshow(volume[...,ax[i].index,ax[i].frame], cmap=plt.get_cmap('Greys_r'), vmin=caxis[0], vmax=caxis[1])          
      ax[i].invert_yaxis()
      ax[i].set_title('Slice {:d}, frame{:d}'.format(ax[i].index,ax[i].frame))

    fig.canvas.mpl_connect('key_press_event', process_key)
    plt.show()

def process_key(event):
    fig = event.canvas.figure
    for ax in fig.axes:
      if event.key == 's':
          previous_slice(ax)
      elif event.key == 'w':
          next_slice(ax)
      elif event.key == 'a':
          previous_frame(ax)
      elif event.key == 'd':
          next_frame(ax)
    fig.canvas.draw()

def previous_slice(ax):
    """Go to the previous slice."""
    volume = ax.volume
    ax.index = (ax.index - 1) % volume.shape[2]  # wrap around using %
    ax.images[0].set_array(volume[...,ax.index,ax.frame])
    ax.set_title('Slice {:d}, frame {:d}'.format(ax.index, ax.frame))

def next_slice(ax):
    """Go to the next slice."""
    volume = ax.volume
    ax.index = (ax.index + 1) % volume.shape[2]
    ax.images[0].set_array(volume[...,ax.index,ax.frame])
    ax.set_title('Slice {:d}, frame {:d}'.format(ax.index, ax.frame))

def previous_frame(ax):
    """Go to the previous frame."""
    volume = ax.volume
    ax.frame = (ax.frame - 1) % volume.shape[3]  # wrap around using %
    ax.images[0].set_array(volume[...,ax.index,ax.frame])
    ax.set_title('Slice {:d}, frame {:d}'.format(ax.index, ax.frame))

def next_frame(ax):
    """Go to the next frame."""
    volume = ax.volume
    ax.frame = (ax.frame + 1) % volume.shape[3]
    ax.images[0].set_array(volume[...,ax.index,ax.frame])
    ax.set_title('Slice {:d}, frame {:d}'.format(ax.index, ax.frame))
