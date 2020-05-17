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
#%% Paths and meta-deta
root_dir = '/imaging/ai05/RED/RED_MEG'
envcor_path = join(root_dir, 'aec_combined')
meta_dat_path = join(root_dir, 'Combined2.csv')
meta = pd.read_csv(meta_dat_path)
MAINDIR = join(root_dir, 'resting', 'STRUCTURALS')
#%% Make list of AEC files to read in, and check they exist
#%% Calculate average corellation matrix for each brain

tmp_id = meta.Alex_ID.to_list()# get IDS
tmp_id = [str(f).replace('_','-') for f in tmp_id] # replace to match filenames

betahigh =[f for f in listdir(envcor_path) if 'Upper Beta' in f] # find upper betas
nMEG_id = [i for i in tmp_id if i in [ii.split('_')[0] for ii in betahigh]]
betahigh = [betahigh[[i for i, s in enumerate(betahigh) if i_d in s][0]] for i_d in nMEG_id] # get them in order

# Read into one big matrix nparcels x nparcels x participants
mat_3d = np.zeros(shape=(len(betahigh), 68, 68, ))
for i, f in enumerate(betahigh):
    mat_3d[i,:,:] = np.load(join(envcor_path, f))

#%% Create new dimensions with different graph theory metrics
# Strength
# For each node calculate sum of weights
strength = mat_3d.sum(axis=1)

# Degree
# get rid of the uneccesary triangle
# Binarize using a proportional threshold
binarized = np.array([mat > np.percentile(mat, 70) for mat in mat_3d], dtype=int)
degree = binarized.sum(axis=1)

#import into networkx, one graph for each participant
G_list = [nx.from_numpy_matrix(g) for g in binarized]
G_weighted_list = [nx.from_numpy_matrix(g) for g in mat_3d]

# clustering coefficient
clustering = np.array([list(nx.clustering(g).values()) for g in G_list])

# node conneectivity for each graph
connectivity_n = [nx.node_connectivity(g) for g in G_list]

stacked = np.dstack((strength, degree, clustering))
measure_names = ['Node Stength', 'Node Degree', 'Cluster Coefficient']

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
for i, megid in enumerate(nMEG_id):
    row = meta[meta.Alex_ID == megid.replace('-','_')].iloc[0]
    # what struct are we using
    if row.Study == 1:
        this_str = ace_struct
    elif row.Study == 2:
        this_str = red_struct
    # loop through cells and construct row for each sub structure
    tmp_sum = []
    tmp_main = []
    dan_ind = row.Dan_ID
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

# make no-Nan versions
struct_sum_nanfree = struct_sum[is_data_nan,:,:]
struct_main_nanfree = struct_main[is_data_nan,:,:]
ids_nanfree = list(np.array(nMEG_id)[is_data_nan])
#%% Tests in a random graph
G = G_list[3]
plt.subplot()
nx.draw_shell(G, with_labels=True, font_weight='bold')
plt.savefig('/imaging/ai05/images/net.png')
plt.close('all')
#%%choose behavioural predictors
predictors = ['Age', 'Equivalised_Income', 'Sub_SES', 'WASI_Mat', 'WASI_Voc', 'SDQ_emotional', 'SDQ_prosocial',
              'SDQ_conduct', 'SDQ_peer', 'SDQ_hyperactivity']
nMEG_id = ids_nanfree
#Z-score the behavioural predictors
preds = np.zeros(shape=(len(nMEG_id), len(predictors)))
for i, megid in enumerate(nMEG_id):
    df_id = megid.replace('-','_')
    for ii, p in enumerate(predictors):
        preds[i, ii] = meta.loc[meta.Alex_ID == df_id][p].values[0]

#%%transpose to regressor matrix
regmat = np.transpose(preds)

# linear interpolate nans
# TODO: DO NOT DO THIS, IT JUST ADDS NOISE. WE DO THIS HERE TO AVOID REMOVING MORE PARTICIPANTS
def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]
for i in range(regmat.shape[0]):
    nans, x= nan_helper(regmat[i,:])
    regmat[i,:][nans]= np.interp(x(nans), x(~nans), regmat[i,:][~nans])
# z score
regmat = np.array([scipy.stats.zscore(i) for i in regmat])

# load in data to GLM data struture
# Functional connectomes
#dat = glmtools.data.TrialGLMData(data=mat_3d, dim_labels=['participants', 'parcels_1', 'parcels_2'])

# Functional connectomes stats
#dat = glmtools.data.TrialGLMData(data=stacked, dim_labels=['participants', 'parcels_1', 'measure'])

# Structural connetomes - use CSA non-normalised connectome
#dat = glmtools.data.TrialGLMData(data=struct_main_nanfree[:,0,:,:], dim_labels=['participants', 'parcels_1', 'parcels_2'])

# Structural connectomes stats
dat = glmtools.data.TrialGLMData(data=struct_sum_nanfree, dim_labels=['participants', 'measure', 'parcel'])


# add regressor for intercept
regs = list()
regs.append(glmtools.regressors.ConstantRegressor(num_observations=dat.info['num_observations']))
# make contrasts and add intercept
contrasts = [glmtools.design.Contrast(name='Intercept',values=[1] + [0]*regmat.shape[0])]
#loop through continous regreessors and add (also to the info and contrasts)
for i in range(regmat.shape[0]):
    regs.append(glmtools.regressors.ParametricRegressor(values=regmat[i],
                                                        name=predictors[i],
                                                        preproc='z',
                                                        num_observations=dat.info['num_observations']))
    # add covariate to info
    dat.info[predictors[i]] = regmat[i]
    values = [0] * (regmat.shape[0] +1)
    values[i+1] = 1
    contrasts.append(glmtools.design.Contrast(name=predictors[i],values=values))

# contrasts
des = glmtools.design.GLMDesign.initialise(regs,contrasts)

model = glmtools.fit.OLSModel( des, dat )

#%% permute
nperms = 5000
nulls = np.zeros(shape=(nperms, model.get_tstats().shape[0], model.get_tstats().shape[1], model.get_tstats().shape[2]))
modes = ['shuffle']*nulls.shape[1]
modes[0] = 'sign'

def perm_func(i, chunk, modes):

    for ii in chunk:
        # shuffle design matrix
        this_desmat = des.design_matrix.copy() # copy matrix
        this_des = deepcopy(des) # deep copy object
        if modes[i] == 'shuffle':
            I = np.random.permutation(this_desmat.shape[0]) # indices shuffled
            this_desmat[:,i] = this_desmat[I,i] # do the shuffling
        elif modes[i] == 'sign':
            I = np.random.permutation(np.tile( [1,-1], int(this_desmat.shape[0]/2)))
            if len(I) != this_desmat.shape[0]:
                index = np.random.permutation(I)[0]
                #print(index)
                I = np.append(I, I[index])
            this_desmat[:,i] = this_desmat[:,i] * I

        # replace with shuffled version
        this_des.design_matrix = this_desmat
        #run glm
        this_model = glmtools.fit.OLSModel(this_des, dat)
        #pipe stats into nulls
        nulls[ii,i,:] = this_model.get_tstats()[i]

threads = 1

for i in range(nulls.shape[1]):

    print(f'\r {i}th contrast', end="")
    # divide the perms by threads
    per_thread = round(nperms/threads)
    ii_list = []
    for splits in range(threads):
        ii_list.append(range(per_thread*splits, per_thread*(splits+1)))
    joblib.Parallel(n_jobs =threads)(
        joblib.delayed(perm_func)(i, chunk, modes) for chunk in ii_list)

#%%

np.save('/imaging/ai05/RED/RED_MEG/struct_stat_nulls.npy',nulls)

nulls = np.load('/imaging/ai05/RED/RED_MEG/struct_stat_nulls.npy')
#%% calculate the centiles
thresh = 1
threshs = np.zeros(shape=(2,nulls.shape[1]))
for i in range(nulls.shape[1]):
    threshs[0,i] = np.percentile(nulls[:,i,:,:].flatten(),100-thresh)
    threshs[1,i] = np.percentile(nulls[:,i,:,:].flatten(),thresh)


#%% look at results
stats = model.get_tstats()
sig_mask = np.empty(shape=(stats.shape))
for i in range(stats.shape[0]):
    sig_mask[i] = (stats[i] > threshs[0,i]) | (stats[i] < threshs[1,i])


#%%
#% setup stuff for glass plots
#read in labels from fsaverage
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

#get fsaverage
fsaverage = datasets.fetch_surf_fsaverage()

# derive hemisphere index 0 = left and 1 = right
hemi_ind = np.array([0 if 'lh' in l.name else 1 for l in labels])

#%% do a basic matrix plot for
import matplotlib.ticker as plticker
loc = plticker.MultipleLocator(base=1.0) # this locator puts ticks at regular intervals
loc2 = plticker.MultipleLocator(base=1.0) # this locator puts ticks at regular intervals

for i in range(stats.shape[0]):
    fig, ax = plt.subplots(figsize=(18,7))
    ax.imshow(stats[i,:,:]) # plot the matrix
    ax.imshow(sig_mask[i,:,:],alpha=0.6, cmap='gray')#plot the significane overlay
    ax.set_title(model.contrast_names[i])
    #xaxis
    ax.set_xticklabels(label_names)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
         rotation_mode="anchor")
    ax.xaxis.set_major_locator(loc)

    #yaxis
    ax.set_yticklabels(struct_sum_labs)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_yticklabels(), rotation=0, ha="right",
             rotation_mode="anchor")
    ax.yaxis.set_major_locator(loc2)

    fig.tight_layout()
    plt.savefig(f'/imaging/ai05/images/struct_stats_{i}.png', dpi=250)
    plt.close('all')
#%% load in the surface files from freesurfer
fs_ave_dir = join(MAINDIR,'FS_SUBDIR', 'fsaverage_1')
surf_lh = mne.read_surface(join(MAINDIR,'FS_SUBDIR', 'fsaverage_1', 'surf', 'lh.pial'), return_dict=True)
surf_rh = mne.read_surface(join(MAINDIR,'FS_SUBDIR', 'fsaverage_1', 'surf', 'rh.pial'), return_dict=True)

# use the labels above and work out where each vertice in the surface files lie
# Number of vertices in labels do not total vertices in surface files, as some fall outside any parcellations
# these parcel-less vertices will be 0, everything else will be 1-len(parcellations)
labs_lh = [l for l in labels if 'lh' in l.name]
labs_rh = [l for l in labels if 'rh' in l.name]

# get list of rois with unique non-zero int ID for each
roi_lh = np.zeros(surf_lh[2]['np'])
for i, lab in enumerate(labs_lh):
    for v in lab.vertices:
        roi_lh[v] = i+1

roi_rh = np.zeros(surf_lh[2]['np'])
for i, lab in enumerate(labs_rh):
    for v in lab.vertices:
        roi_rh[v] = i+1
#%% test by plotting all parcelations
plotting.plot_surf_roi(surf_rh[0:2], roi_map=roi_lh)
plt.savefig('/imaging/ai05/images/roi_test.png')
plt.close('all')
#%% Plot the outputs
fig, axs = plt.subplots(nrows=(stats.shape[2]*2)+1, ncols=stats.shape[0],
                       gridspec_kw = {'height_ratios':[1.5] +[1]*(stats.shape[2]*2)})



ims = []
for i in range(stats.shape[0]):
    name = model.contrast_names[i]
    ims.append(axs[0,i].imshow(stats[i])) # plot the matrix
    axs[0,i].imshow(sig_mask[i],alpha=0.6, cmap='gray')#plot the significane overlay
    axs[0,i].set_title(name)
    axs[0,i].set_xticklabels(['']+ measure_names)
    # Rotate the tick labels and set their alignment.
    plt.setp(axs[0,i].get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

    for ii in range(stats.shape[2]):

        # make parcellation mask, ints for each significant parcel
        lh_mask = sig_mask[i, np.array(hemi_ind) == 0, ii]
        rh_mask = sig_mask[i, np.array(hemi_ind) == 1, ii]

        #list of areas to zero out
        lh_zero_out_indices = [i[0] for i in np.argwhere(lh_mask == 0)]
        rh_zero_out_indices = [i[0] for i in np.argwhere(rh_mask == 0)]

        tmp_lh = deepcopy(roi_lh)
        tmp_rh = deepcopy(roi_rh)
        for x in range(len(tmp_lh)):
            if tmp_lh[x] in lh_zero_out_indices:
                tmp_lh[x] = 0
        for x in range(len(roi_rh)):
            if tmp_rh[x] in rh_zero_out_indices:
                tmp_rh[x] = 0
        lhind = ((ii+1)*2)-1
        axs[lhind,i].remove()
        axs[lhind,i] = fig.add_subplot((stats.shape[2]*2)+1, stats.shape[0], (lhind)*stats.shape[0]+(i+1), projection='3d')

        joined = (np.concatenate((surf_rh[0], surf_lh[0])), np.concatenate((surf_rh[1], surf_lh[1])))
        plotting.plot_surf_roi(surf_lh[0:2],
                                roi_map=tmp_lh,
                                figure=fig,
                                axes=axs[((ii+1)*2)-1,i])
        rhind = ((ii+1)*2)
        axs[rhind,i].remove()
        axs[rhind,i] = fig.add_subplot((stats.shape[2]*2)+1, stats.shape[0], (rhind)*stats.shape[0]+(i+1), projection='3d')
        plotting.plot_surf_roi(surf_rh[0:2],
                               roi_map=tmp_rh,
                               figure=fig,
                               axes=axs[rhind,i])

    # plotting.plot_connectome(sig_mask[i], coords,
    #                          edge_threshold=0.9,
    #                          title=f'{name}',
    #                          figure=fig,
    #                          axes=axs[1,i],
    #                          node_size=2)

for i, im in enumerate(ims):
    fig.colorbar(im,ax=axs[0,i])
    axs[0,i].set_aspect('auto')


# rownames = ['', 'Strength LH', 'Strength RH',
#             'Degree LH', 'Degree RH',
#             'Clustering LH', 'Clustering RH',]

rownames = struct_sum_labs
for ax, row in zip(axs[:,0], rownames):
    ax.set_ylabel(row, rotation=0, size='large')

fig.set_size_inches(35.0, 15)
plt.tight_layout()
fig.savefig('/imaging/ai05/images/graph_glm_thresh_01.png')
plt.close('all')