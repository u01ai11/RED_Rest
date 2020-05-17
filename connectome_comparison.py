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


#%% for each subject calculate difference from structural to functional

#%% options for this are complicated:
# - Singular Value Decomposition, the absolute SVD for two identical matrices will be the same
#   , so the difference between them gives a coarse idea of differences
differences_weighted = []
differences_binarized = []
bin_thresh = 95

struct_ind = 0

bfunc= np.array([mat > np.percentile(mat, bin_thresh) for mat in mat_3d], dtype=int)
bstruc= np.array([mat > np.percentile(mat, bin_thresh) for mat in struct_main[:, struct_ind, :,:]], dtype=int)

for i, sub in enumerate(nMEG_id):
    if is_data_nan[i] == False:
        differences_weighted.append(np.nan)
        differences_binarized.append(np.nan)
    else:
        difference = np.abs(svd(bstruc[i, :,:])[1].sum() - svd(bfunc[i, :,:])[1].sum())
        differences_binarized.append(difference)

#%% Use inexact graph matching algorithms to calculate node-to-node similarity (Osmanlioglu et al 2019)

#first only use participants with both full connectomes

# get inputs
in_struct = struct_main[is_data_nan, 4, :,:]
in_func = mat_3d[is_data_nan,:,:]

# functional MEG connectomes are pretty dense, so do some thresholding

binary_matched = np.full((in_func.shape[0:2]), False)
euc_distances = np.zeros(in_func.shape)

for p in range(in_func.shape[0]):
    print(p)
    # Here we just use  participant
    this_structural = in_struct[p, :,:]
    #this_functional = in_struct[p, 7, :,:]
    this_functional = in_func[p,:,:]

    # deal with NaN and infinte values
    this_structural = np.nan_to_num(this_structural)

    # z score both
    this_structural = ss.zscore(this_structural)
    this_functional = ss.zscore(this_functional)

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

    # add to group level matrices
    binary_matched[i,:] = binary_nodes
    print(binary_nodes.sum())
    euc_distances[i,:,:] = cost_euc

# Step 5, repeat for all participants and average to give a probability for each node


