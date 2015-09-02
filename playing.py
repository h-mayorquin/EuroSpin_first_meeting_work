import numpy as np
import matplotlib.pyplot as plt
import os

#####################
# Load data
#####################

percentage = ''
percentage = 0.1
folder = './data/'
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
remove_axis = True
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

# Save the figure
plt.savefig(filename)
os.system('pdfcrop %s %s' % (filename, filename))
plt.show()
