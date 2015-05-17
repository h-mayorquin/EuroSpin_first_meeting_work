import numpy as np
import matplotlib.pyplot as plt
from load_data.load_binary_files import training_ims, training_labels
import cPickle as pickle
from sklearn.cluster import KMeans


#####################
# Load data
#####################

# Select quantity of data to use
N_data_total = len(training_labels)
percentage = 0.1
N_to_use = int(percentage * N_data_total)

# Decide how much data to use
X = training_ims[0:N_to_use]
Y = training_labels[0:N_to_use]

# Load the patches
folder = './data/'
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
patch = 2
indexes = partition[patch][0]
data = X[:, indexes]

# Now we do K means
# Now we do the clustering
n_cluster = 10
n_int = 4
kmeans = KMeans(n_clusters=n_cluster, n_jobs=3)
fit = kmeans.fit(data)
centers = fit.cluster_centers_


#####################
# Plot
#####################

matrix = np.zeros(X.shape[1])
matrix[indexes] = centers[0]
plt.imshow(matrix.reshape((28, 28)))
plt.show()
