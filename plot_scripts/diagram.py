import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
from sklearn.cluster import KMeans
import os
import sys
sys.path.append('../')
from mnist import MNIST

#####################
# Load data
#####################


mndata = MNIST('../data')
# Load a list with training images and training labels
training_ims, training_labels = mndata.load_training()
testing_ims, testing_labels = mndata.load_testing()

# Transform everything into array
training_ims = np.array(training_ims)
training_labels = np.array(training_labels)

# Make them binary
training_ims[training_ims > 0] = 1


# Select quantity of data to use
N_data_total = len(training_labels)
percentage = 0.1
N_to_use = int(percentage * N_data_total)

# Decide how much data to use
X = training_ims[0:N_to_use]
Y = training_labels[0:N_to_use]

# Load the distances
percentage = ''
percentage = 0.1
folder = '../data/'
name = 'mds_embedd'
format = '.npy'
file_name = folder + name + str(percentage) + format
embedd = np.load(file_name)


#####################
# Do the sensor space clustering
#####################

# K Means parameters
max_iter = 1000
jobs = 2
eps = 1e-9
n_init = 3
space_clusters = 9

# Do the clustering
kmeans = KMeans(n_clusters=space_clusters, n_init=n_init,
                max_iter=max_iter, n_jobs=jobs)

labels = kmeans.fit_predict(embedd)

patch_to_plot = 5
indexes = np.where(labels == patch_to_plot)[0]

#####################
# Do the data space clustering
#####################

data_clusters = 9
code_vectors = [0, 2, 4, 6, 8]
npatches = len(code_vectors)

# Get the data for the patch
data = X[..., indexes]

kmeans = KMeans(n_clusters=data_clusters, n_jobs=3)
fit = kmeans.fit(data)
centers = fit.cluster_centers_

##########################
# Ploting
##########################

# Plot parameters
fontsize = 20
figsize = (16, 12)
axes_position = [0.1, 0.1, 0.8, 0.8]
remove_axis = True
title = 'Diagram'
xlabel = ''
ylabel = ''

# Plot format
inter = 'nearest'
cmap1 = 'jet'
cmap2 = 'coolwarm'
cmap2 = 'bwr'
cmap2 = 'seismic'

# Save directory
folder = '../results/'
extensions = '.pdf'
name = 'diagram'
filename = folder + name + extensions

# Plot here
gs = plt.GridSpec(5, npatches)
fig = plt.figure(figsize=figsize)
axes = []

# First we plot the whoe image
ax = fig.add_subplot(gs[0, 0])
map = labels.reshape((28, 28))
im = ax.imshow(map, interpolation=inter, cmap=cmap1)
axes.append(ax)

for index, code_vector in enumerate(code_vectors):
    ax = fig.add_subplot(gs[1, index])
    matrix = np.zeros(X.shape[1])
    matrix[indexes] = centers[code_vector, ...]
    map = matrix.reshape((28, 28))
    aux = np.max(map)
    im = ax.imshow(map, vmin=-aux, vmax=aux, interpolation=inter, cmap=cmap2)
    axes.append(ax)


# Remove the axis
if remove_axis:
    for axi in axes:
        axi.get_xaxis().set_visible(False)
        axi.get_yaxis().set_visible(False)


plt.subplots_adjust(left=0.25, right=0.75, wspace=0.0, hspace=0)

# Save the figure
plt.savefig(filename)
os.system('pdfcrop %s %s' % (filename, filename))
plt.show()
