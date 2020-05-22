#%% Import stuff
import pandas as pd
import sys
import time
import os
from os import listdir as listdir
from os.path import join
from os.path import isdir
from os.path import isfile
import numpy as np
sys.path.insert(0, '/imaging/ai05/RED/RED_MEG/resting/analysis/RED_Rest')
from REDTools import preprocess
from REDTools import study_info
from REDTools import sourcespace_setup
from REDTools import connectivity
import mne
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from nilearn import datasets
from nilearn.input_data import NiftiLabelsMasker
from nilearn import plotting
sys.path.insert(0, '/home/ai05/Downloads/glm')
import glmtools
import scipy.stats as ss
from copy import deepcopy
import scipy
import joblib
import networkx as nx
from scipy.io import loadmat
from numpy.linalg import svd
from scipy.spatial.distance import euclidean
from scipy.optimize import linear_sum_assignment
#%% Paths and meta-deta
root_dir = '/imaging/ai05/RED/RED_MEG'
envcor_path = join(root_dir, 'aec_combined')
meta_dat_path = join(root_dir, 'Combined2.csv')
meta = pd.read_csv(meta_dat_path)
MAINDIR = join(root_dir, 'resting', 'STRUCTURALS')

#%% Calculate corellation matrix for each brain

tmp_id = meta.Alex_ID.to_list()# get IDS
tmp_id = [str(f).replace('_','-') for f in tmp_id] # replace to match filenames

betahigh =[f for f in listdir(envcor_path) if 'Upper Beta' in f] # find upper betas
nMEG_id = [i for i in tmp_id if i in [ii.split('_')[0] for ii in betahigh]]
betahigh = [betahigh[[i for i, s in enumerate(betahigh) if i_d in s][0]] for i_d in nMEG_id] # get them in order

# Read into one big matrix nparcels x nparcels x participants
mat_3d = np.zeros(shape=(len(betahigh), 68, 68, ))
for i, f in enumerate(betahigh):
    mat_3d[i,:,:] = np.load(join(envcor_path, f))

#%% load in stuctural connectomes
# read in matlab objects
str_path = join('/imaging', 'ai05', 'RED', 'RED_MEG', 'resting', 'Structural_connectomes')
red_struct = loadmat(join(str_path, 'red_data68.mat'))['stackdata68']
ace_struct = loadmat(join(str_path, 'amy_data68.mat'))['stackdata68']
# get labels for each column in the above
f= open(join(str_path, 'info.txt'), "r")
stack_labels =f.read().split("\n")[0:-1]
f.close()

#%%reorder to match participants above
# first values with summary (i.e. one value per parcel, per participant) [part,measure, parcel]
struct_sum = np.zeros((len(nMEG_id), (len([f for f in red_struct[0] if len(f.shape)<3])), 68))

# then whole connectomes per participant [part, measure, parcel_dim1, parcel_dim2]
struct_main = np.zeros((len(nMEG_id), len([f.shape for f in red_struct[0] if len(f.shape)>2]),68,68))

#isnan counter for participants!
is_data_nan = []
# loop through participants
#for i, row in meta.iterrows():
for i, megid in enumerate([f.replace('-', '_') for f in nMEG_id]):
    row = meta[meta.Alex_ID == megid]
    # what struct are we using
    if row.Study.array[0] == 1:
        this_str = ace_struct
    elif row.Study.array[0] == 2:
        this_str = red_struct
    # loop through cells and construct row for each sub structure
    tmp_sum = []
    tmp_main = []
    dan_ind = row.Dan_ID.array[0]
    if ~np.isnan(dan_ind):
        dan_ind = int(dan_ind)-1
        is_data_nan.append(True)
    else:
        is_data_nan.append(False)
    nan3 = np.empty((68,68))
    nan3[:] = np.nan
    struct_sum_labs = []
    struct_main_labs = []
    for cell, lab in zip(this_str[0], stack_labels):
        # parcel summary?
        if len(cell.shape) <3:
            if np.isnan(dan_ind):
                tmp_sum.append(np.array([np.nan]*68))
            else:
                tmp_sum.append(cell[dan_ind,:])
            struct_sum_labs.append(lab)
        # full connectomes
        elif len(cell.shape) >2:
            if np.isnan(dan_ind):
                tmp_main.append(nan3)
            else:
                tmp_main.append(cell[dan_ind,:,:])
            struct_main_labs.append(lab)
    # add these onto the main arrays
    struct_sum[i,:,:] = np.array(tmp_sum)
    struct_main[i,:,:] = np.array(tmp_main)



#%% Use inexact graph matching algorithms to calculate node-to-node similarity (Osmanlioglu et al 2019)

#first only use participants with both full connectomes

# get inputs
in_struct = struct_main[is_data_nan, 0, :,:]
in_func = mat_3d[is_data_nan,:,:]
#in_func = struct_main[is_data_nan, 4, :,:]
# functional MEG connectomes are pretty dense, so do some thresholding
# let's take the top X percentage of connections
percentile = 90

# or we can use density base thresholding
# i.e. number of nonzero connections divided by all possible connections
# first choose a threshold for removal and remove


def calc_density(matrix):
    rem_self =  matrix- np.diag(np.diag(matrix))
    return np.count_nonzero(rem_self)/np.prod(rem_self.shape)

def permute_connections(m):

    """Permutes the connection weights in a sparse 2D matrix. Keeps the
    location of connections in place, but permutes their strengths.

    Arguments

    m       -   numpy.ndarray with a numerical dtype and shape (N,N) where N
                is the number of nodes. Lack of connection should be denoted
                by 0, and connections should be positive or negative values.

    Returns

    perm_m  -   numpy.ndarray of shape (N,N) with permuted connection weights.
    """

    # Verify the input.
    if len(m.shape) != 2:
        raise Exception("Connection matrix `m` should be 2D")
    if m.shape[0] != m.shape[1]:
        raise Exception("Connection matrix `m` should be symmetric")

    # Copy the original matrix to prevent modifying it in-place.
    perm_m = np.copy(m)

    # Get the indices to the lower triangle.
    i_low = np.tril_indices(perm_m.shape[0], -1)

    # Create a flattened copy of the connections.
    flat_m = perm_m[i_low]

    # Shuffle all non-zero connections.
    nonzero = flat_m[flat_m != 0]
    np.random.shuffle(nonzero)
    flat_m[flat_m != 0] = nonzero

    # Place the shuffled connections over the original connections in the
    # copied matrix.
    perm_m[i_low] = flat_m

    # Copy lower triangle to upper triangle to make the matrix symmertical.
    perm_m.T[i_low] = perm_m[i_low]

    return perm_m

density_S = [calc_density(f) for f in in_struct]

# get copy
in_c = in_func.copy()


start_t = 0.07 # start threshold
step = 0.0001 # start step

# target density to match input
target_density = np.mean(density_S)

for i in range(1000): # iterations
    tmp_c = in_c.copy() # make copy
    for ii in range(in_c.shape[0]): # participants
        tmp_c[ii][tmp_c[ii,:,:] < start_t] = 0
    # calc density
    mean_density = np.mean([calc_density(f) for f in tmp_c])

    if mean_density > target_density:
        start_t += step
    else:
        start_t -= step

    print(mean_density)

#use tmp_c as the real
in_func = tmp_c
#%%


graph_1 = in_struct
graph_2 = match_density(in_func, graph_1, 0.7, 0.0001,1000)

matching = match_graphs(graph_1, graph_2, 20)

null = generate_null_dist(graph_1, graph_2, 10, 10)

#%%
#initialise empty arrays
binary_matched = np.full((in_func.shape[0:2]), False) # an array for just the node accuracy
euc_distances = np.zeros(in_func.shape) # for all the euclidean distances
binary_matched_matrix = np.zeros(in_func.shape, dtype=int)


for p in range(in_func.shape[0]):
    print(p)
    # Here we just use one participant
    this_structural = in_struct[p, :,:]
    this_functional = in_func[p,:,:]


    # Make functional pretty sparse

    # # Simple percentage thresholding
    # thresh = np.percentile(this_functional, percentile)
    # this_functional[this_functional<thresh] = 0


    # z score both
    this_structural = ss.zscore(this_structural)
    this_functional = ss.zscore(this_functional)

    # deal with NaN and infinte values
    this_structural = np.nan_to_num(this_structural)
    this_functional = np.nan_to_num(this_functional)
    # Step 1, feature vector for each node, containing connectivity for each over parcellation in graph
    # For each graph func and struct

    # Note: this is essentially the connectome matrix (each row is a feature vector), so no need to do anything here

    # Step 2, cost function for each node in structural to it's corresponding node in  functional graph
    # Cost funcion is the eucledian distance between the two feature vectors of these nodes, do this for all nodes
    # structural
    cost_euc = np.empty(this_structural.shape)
    for i in range(cost_euc.shape[0]):
        cost_euc[i] = [euclidean(this_structural[i], f) for f in this_functional]

    # Step 3, Hungarian algorithm for matching nodes of structural graphs with that of functional graphs,
    # Matching matrix where each structural node has it's most similiar equivalent node in functional
    match_inds = linear_sum_assignment(cost_euc) # 2nd index gives least cost cost node for each node

    # Step 4, Create a binary matching accuracy graph. A node being matched with it's equivalent in the other graph
    # is considered accurate (i.e. a 1) anything else is a 0
    binary_nodes = np.array([i == j for i, j in zip(match_inds[0], match_inds[1])])

    # Make this in 2D as well
    binary_match_mat = np.zeros(this_structural.shape, dtype=int) # intialise empty
    for i, row in enumerate(binary_match_mat): # loop through rows/nodes
        row[match_inds[1][i]] = 1 # place a 1 in the corresponding column where it matched

    # add to group level matrices
    binary_matched[p,:] = binary_nodes
    print(binary_nodes.sum())
    binary_matched_matrix[p,:,:] = binary_match_mat
    euc_distances[p,:,:] = cost_euc

# Step 5, repeat for all participants and average to give a probability for each node

nodes_freq = binary_matched.mean(axis=0) # percentage of users with correctly matched nodes
nodes_freq_matrix = binary_matched_matrix.mean(axis=0) # all node combination matching weights

# plotting results

#%% setup brain glassbrain bits
labels = mne.read_labels_from_annot('fsaverage_1', parc='aparc',
                                    subjects_dir=join(MAINDIR,'FS_SUBDIR'))
labels = labels[0:-1]
label_names = [label.name for label in labels]
coords = []
# TODO: Find a better way to get centre of parcel
#get coords of centre of mass
for i in range(len(labels)):
    if 'lh' in label_names[i]:
        hem = 1
    else:
        hem = 0
    coord = mne.vertex_to_mni(labels[i].center_of_mass(subjects_dir=join(MAINDIR,'FS_SUBDIR')), subject='fsaverage_1',hemis=hem,subjects_dir=join(MAINDIR,'FS_SUBDIR'))
    coords.append(coord[0])

#%%
plt.close('all')
fig, axs = plt.subplots(nrows=1, ncols=3) # two plots

# plot the matrix
pos = axs[0].imshow(nodes_freq_matrix, cmap= plt.cm.YlOrRd)
fig.colorbar(pos, ax=axs[0])
axs[0].set_xlabel('MEG Nodes')
axs[0].set_ylabel('MRI Nodes')
axs[0].set_title('Node Matching between MRI and MEG Graphs')
# plot each nodes frequency for matching

#we need to fake a connectome to use niilearns built in plotting functions
#first make a fake connectome from the vector, with each node having one connection with strength f its frequency
fake_connect = np.zeros((len(nodes_freq), len(nodes_freq))) # intialise empty
for i, row in enumerate(fake_connect): # loop through rows/nodes
    row[i] = nodes_freq[i]
plotting.plot_connectome_strength(fake_connect,
                         node_coords=coords,
                         title='Parcellation Matching Accuracy',
                         figure=fig,
                         axes=axs[1],
                         cmap=plt.cm.YlOrRd)

# Calculate label names for 10% most matching ROIs
top_percentile = np.percentile(nodes_freq, 90)
top_labels = np.array(label_names)[nodes_freq > top_percentile]
top_vals = nodes_freq[nodes_freq > top_percentile]
top_vals = top_vals.round(3).astype(str)
to_table = [top_vals.tolist(), top_labels.tolist()]
rotated = list(zip(*to_table[::-1]))
table = axs[2].table(rotated, loc=9)
table.auto_set_font_size(False)
table.set_fontsize(12)


fig.set_size_inches(22.0, 6)
fig.savefig('/imaging/ai05/images/net_matching.png')
