import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os

#####################
# Load data
#####################

percentage = ''
percentage = 0.1
folder = '../data/'
name = 'mds_embedd'
format = '.npy'
file_name = folder + name + str(percentage) + format
embedd = np.load(file_name)

##########################
# Cluster the Space
##########################

labels_list = []
Npartitions = 16
partition_set = np.arange(2, 18, 1)

# K Means parameters
max_iter = 1000
jobs = 2
eps = 1e-9
n_init = 3

for index in xrange(Npartitions):

    # Clustering parameters
    clusters = partition_set[index]
    kmeans = KMeans(n_clusters=clusters, n_init=n_init,
                    max_iter=max_iter, n_jobs=jobs)

    labels = kmeans.fit_predict(embedd)
    labels_list.append(labels)

##########################
# Ploting
##########################

to_plot = [label.reshape((28, 28)) for label in labels_list]
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
cmap = 'jet'

# Save directory
folder = '../results/'
extensions = '.pdf'
name = 'space_clustering'
filename = folder + name + extensions

# Plot here
nplots = int(np.sqrt(Npartitions))
gs = plt.GridSpec(nplots, nplots)
fig = plt.figure(figsize=figsize)
axes = []

for index1 in xrange(nplots):
    for index2 in xrange(nplots):

        ax = fig.add_subplot(gs[index1, index2])
        index = index1 * nplots + index2
        map = to_plot[index]
        im = ax.imshow(map, interpolation=inter, cmap=cmap)
        ax.set_title(str(partition_set[index]) + ' clusters')
        # ax.set_aspect(1)
        axes.append(ax)


# Remove the axis
if remove_axis:
    for index in range(Npartitions):
        axes[index].get_xaxis().set_visible(False)
        axes[index].get_yaxis().set_visible(False)

# fig.tight_layout(pad=0, w_pad=0, h_pad=0)

plt.subplots_adjust(left=0.25, right=0.75, wspace=0.0, hspace=0)

# Save the figure
plt.savefig(filename)
os.system('pdfcrop %s %s' % (filename, filename))
plt.show()
