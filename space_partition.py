import numpy as np
from sklearn.cluster import KMeans
import cPickle as pickle

############################
# First we load the data
############################

percentage = ''
percentage = 0.1
folder = './data/'
name = 'mds_embedd'
format = '.npy'
file_name = folder + name + str(percentage) + format
embedd = np.load(file_name)

############################
# Now we do the clustering
############################

# Parameters
n_cluster = 10
n_int = 4
k_means = KMeans(n_clusters=n_cluster, n_jobs=3)
labels = k_means.fit_predict(embedd)

# Initialize a dictionary
partition = {}

# Create a dictionary
for label in range(n_cluster):
    partition[label] = np.where(labels == label)

print 'Done with file creation now start saving'

# Store the files for further use
folder = './data/'
name1 = 'pixel_to_hyper'
name2 = 'label_to_patch'
format1 = '.npy'
format2 = '.cpickle'

filename1 = folder + name1 + format1
filename2 = folder + name2 + format2

np.save(filename1, labels)
print 'Saved the first file'

with open(filename2, 'wb') as fp:
    pickle.dump(partition, fp)

print 'Done'
