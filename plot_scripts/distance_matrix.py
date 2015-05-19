import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os

#####################
# Load data
#####################

percentage = ''
percentage = 0.1
folder = '../data/'
name = 'information_distances'
format = '.npy'
file_name = folder + name + str(percentage) + format
distances = np.load(file_name)


#####################
# Plotting
#####################

to_plot = distances

# Plot parameters
fontsize = 20
figsize = (16, 12)
axes_position = [0.1, 0.1, 0.8, 0.8]
colorbar = False
title = 'Distance Matrix'
xlabel = 'Sensor'
ylabel = 'Sensor'

# Plot format
inter = 'nearest'
cmap = 'hot'
origin = 'lower'

# Save directory
folder = '../results/'
extensions = '.pdf'
name = 'distance_matrix'
filename = folder + name + extensions

# Plot here

fig = plt.figure(figsize=figsize)
ax = fig.add_axes(axes_position)
im = ax.imshow(to_plot, interpolation=inter, cmap=cmap, origin=origin)
ax.set_xlabel(xlabel)
ax.set_ylabel(ylabel)
ax.set_title(title)

# Change the font size
axes = fig.get_axes()
for ax in axes:
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(fontsize)

# Colorbar
if colorbar:
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    cbar.solids.set_edgecolor('face')


# Save the figure
plt.savefig(filename)
os.system('pdfcrop %s %s' % (filename, filename))
plt.show()
