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

# Load the patches
folder = '../data/'
name1 = 'pixel_to_hyper'
name2 = 'label_to_patch'
format1 = '.npy'
format2 = '.cpickle'

filename1 = folder + name1 + format1
filename2 = folder + name2 + format2


with open(filename2, 'rb') as fp:
    partition = pickle.load(fp)

space_labels = np.load(filename1)


#####################
# Do the data space clustering
#####################

# Get the data for the sensor patch
patches = [0, 2, 4, 8]
centers = [1, 3, 7, 9]

centers_set = []

n_cluster = 10
n_int = 4

for patch in patches:
    print 'patch ', patch
    indexes = partition[patch][0]
    data = X[:, indexes]
    # Now we do the clustering
    kmeans = KMeans(n_clusters=n_cluster, n_jobs=3)
    fit = kmeans.fit(data)
    centers_set.append(fit.cluster_centers_)


# Put the data in a form to plot
matrix = np.zeros(X.shape[1])

##########################
# Ploting
##########################

npatches = len(patches)
ncenters = len(centers)
Nplots = npatches * ncenters

# to_plot = [label.reshape((28, 28)) for label in labels_list]
# to_plot = to_plot[0]

# Plot parameters
fontsize = 20
figsize = (16, 12)
axes_position = [0.1, 0.1, 0.8, 0.8]
remove_axis = True
title = 'Stress vs size of embedding space'
xlabel = 'Dimension'
ylabel = 'Stress'

# Plot format
inter = 'nearest'
cmap = 'binary'
cmap = 'jet'

# Save directory
folder = '../results/'
extensions = '.pdf'
name = 'data_clustering'
filename = folder + name + extensions

# Plot here
gs = plt.GridSpec(ncenters, npatches)
fig = plt.figure(figsize=figsize)
axes = []


for center_index in xrange(ncenters):
    for patch_index in xrange(npatches):

        ax = fig.add_subplot(gs[center_index, patch_index])
        matrix = np.zeros(X.shape[1])
        # Get the indexes of that particular patch
        map_indexes = partition[patches[patch_index]][0]
        # Extract the centers of this particular patch
        centers_of_patch = centers_set[patch_index]
        # Get the center from the center set
        center = centers_of_patch[centers[center_index]]
        # Construct and remap the matrix
        matrix[map_indexes] = center
        map = matrix.reshape((28, 28))
        im = ax.imshow(map, interpolation=inter, cmap=cmap)
        title = 'Patch=' + str(patches[patch_index]) + ' - '
        title += 'Code vector=' + str(centers[center_index])
        ax.set_title(title)
        axes.append(ax)


# Remove the axis
if remove_axis:
    for index in range(Nplots):
        axes[index].get_xaxis().set_visible(False)
        axes[index].get_yaxis().set_visible(False)

# fig.tight_layout(pad=0, w_pad=0, h_pad=0)
# plt.subplots_adjust(left=0.25, right=0.75, wspace=0.0, hspace=0)

# Save the figure
plt.savefig(filename)
os.system('pdfcrop %s %s' % (filename, filename))
plt.show()
