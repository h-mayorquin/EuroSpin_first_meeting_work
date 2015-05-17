import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
import os


########################
# Load data
########################

percentage = ''
percentage = 0.1
folder = '../data/'
name = 'information_distances'
format = '.npy'
file_name = folder + name + str(percentage) + format
distances = np.load(file_name)

########################
# Do the MDS for different dimensions
########################

dimensions = np.arange(5, 30, 1)
stress_vector = np.zeros_like(dimensions)


for i, dim in enumerate(dimensions):

    # Define classifier
    n_comp = dim
    max_iter = 1000
    eps = 1e-9
    mds = MDS(n_components=n_comp, max_iter=max_iter, eps=eps,
              n_jobs=2, dissimilarity='precomputed')

    x = mds.fit(distances)
    stress = x.stress_ / distances.shape[0]
    print 'Dimension', dim
    print 'The stress is', stress
    stress_vector[i] = stress

########################
# Plot Here
########################

# Plot parameters
fontsize = 20
figsize = (16, 12)
axes_position = [0.1, 0.1, 0.8, 0.8]
title = 'Stress vs size of embedding space'
xlabel = 'Dimension'
ylabel = 'Stress'

# Plot format
width = 7
markersize = 14

# Save directory
folder = '../results/'
extensions = '.pdf'
name = 'stress_vs_dimensions'
filename = folder + name + extensions

# Plot
fig = plt.figure(figsize=figsize)
ax = fig.add_axes(axes_position)
ax.plot(dimensions, stress_vector, '*-', linewidth=width,
        markersize=markersize, markerfacecolor='r')

ax.set_xlabel(xlabel)
ax.set_ylabel(ylabel)
ax.set_title(title)
ax.set_ylim(bottom=0)

# Change the font size
axes = fig.get_axes()
for ax in axes:
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(fontsize)

# Save the figure
plt.savefig(filename)
os.system('pdfcrop %s %s' % (filename, filename))
plt.show()
