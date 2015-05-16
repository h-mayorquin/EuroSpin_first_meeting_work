import numpy as np
import matplotlib.pyplot as plt
from load_data.load_binary_files import training_ims, training_labels
from prob_functions import joint, p_independent
from information_functions import mutual_information, joint_entropy

# Select quantity of data to use
N_data_total = len(training_labels)
percentage = 0.001
N_to_use = int(percentage * N_data_total)

# Decide how much data to use
X = training_ims[0:N_to_use]
Y = training_labels[0:N_to_use]
N_hypercolumns = X.shape[1]
units_per_hypercolumn = 2
distribution = {0: 0, 1: 1, 'no': 3}

# Normalize
normalize = True
low_noise = 10e-10
N1 = 28 * 1
N2 = N1 + 28 * 3
N1 = 0
N2 = N_hypercolumns

# Values fo the map

d_matrix = np.zeros((N_hypercolumns, N_hypercolumns))
domain = range(N1, N2)

for i in domain:
    print 'i', i
    for j in domain:
        p = p_independent(N_hypercolumns, units_per_hypercolumn, X)
        p_i = p[i, :]
        p_j = p[j, :]
        p_joint = joint(i, j, X, distribution, units_per_hypercolumn)

        # If there is any value in the joint distribution that is equal to 0
        if np.any(p_joint == 0):
            # Joint
            p_joint[p_joint < low_noise] = low_noise
            p_joint = p_joint / p_joint.sum()
            p_i = p_joint.sum(axis=1)
            p_j = p_joint.sum(axis=0)

        x3 = joint_entropy(p_joint)

        MI = mutual_information(p_i, p_j, p_joint)
        distance = x3 - MI

        d_matrix[i, j] = distance


########################
# Plot here
########################

plt.imshow(d_matrix, interpolation='nearest')
plt.colorbar()
plt.show()

########################
# Here we save the data
########################

folder = './data/'
name = 'information_distances'
file_name = folder + name + str(percentage)
np.save(file_name, d_matrix)
