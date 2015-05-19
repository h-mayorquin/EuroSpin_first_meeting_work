import numpy as np
from sklearn.manifold import MDS

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
# Do the MDS
#####################

# Define classifier
dim = 10
max_iter = 3000
eps = 1e-9

mds = MDS(n_components=dim, max_iter=max_iter, eps=eps,
          n_jobs=1, dissimilarity='precomputed')

embedd = mds.fit_transform(distances)

########################
# Here we save the data
########################

folder = './data/'
name = 'mds_embedd'
filename = folder + name + str(percentage)
np.save(filename, embedd)
