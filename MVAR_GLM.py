import sys
from os import listdir as listdir
from os.path import join
from os.path import isfile
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, '/imaging/ai05/RED/RED_MEG/resting/analysis/RED_Rest')
from REDTools import dynamics

import mne
import glmtools
import joblib
import sails
import copy
import pandas as pd
import os
from nilearn import plotting

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colorbar as cb
from mpl_toolkits.axes_grid1 import Grid
from matplotlib import patches

import holoviews as hv
hv.extension('bokeh')
from holoviews import opts, dim
from bokeh.io import output_file, save, show
from bokeh.io import curdoc
from bokeh.layouts import layout
from bokeh.models import Slider, Button
#%%
root_dir = '/imaging/ai05/RED/RED_MEG' # the root directory for the project
figdir = '/imaging/ai05/images'
parcel_dir = join(root_dir, 'resting', 'parcel_timecourses') # get parcel time courses
parcel_files = listdir(parcel_dir) # list them

node_names = np.load('/imaging/ai05/RED/RED_MEG/resting/analysis/RED_Rest/age_node_names.npy')

meta_dat_path = join(root_dir, 'Combined2.csv')
meta = pd.read_csv(meta_dat_path)
tmp_id = meta.Alex_ID.to_list()# get IDS
tmp_id = [str(f).replace('_','-') for f in tmp_id] # replace to match filenames
meta.Alex_ID = tmp_id

# define some information about our data
sample_rate = 150 # sample rate
freq_vect = np.linspace(0, sample_rate/2, 36) #frequency vector representing the model metrics

outdir = join(root_dir,'resting', 'MVAR')
#%% now we are going to run a GLM to see how consistent connectivity is accross participants
# The data will be in the form 136x12x12x36 (participant by parcels sender by parcel reciever by frequency power)
# accross participants we would like to know which of these connections and at what frequencies are consistent


#first we need to get age info ready

age = []
IQ = []

ids = [i.split('_')[0] for i in parcel_files]

for id_ in ids:
    #if id is not in meta, nan it
    if id_ not in meta.Alex_ID.to_list():
        age.append(np.nan)
        IQ.append(np.nan)
    else:
        age.append(float(meta[meta.Alex_ID == id_].Age))
        IQ.append(float(meta[meta.Alex_ID == id_].WASI_Mat))

age = np.array(age)
IQ = np.array(IQ)

age[np.isnan(age)] = np.nanmean(age)
IQ[np.isnan(IQ)] = np.nanmean(IQ)
#%%

# then we need to load all of the participants modelled data
glm_data = np.empty((len(parcel_files), 68,68,36))
for i in range(len(parcel_files)):
    id_ = parcel_files[i].split('_')[0]
    glm_data[i,:,:,:] = np.load(join(root_dir,'resting', 'MVAR', f'mvar_OLS_{id_}.npy'))[:,:,:,0]



#%% feed into GLM

dat = glmtools.data.TrialGLMData(data=glm_data, dim_labels=['participants', 'source_nodes', 'target_nodes', 'frequency'])

regs = list()
contrasts = list()
regs.append(glmtools.regressors.ConstantRegressor(num_observations=dat.info['num_observations']))
contrasts.append(glmtools.design.Contrast(name='Intercept',values=[1,0,0])) # for regressor

# add regressor for Age
regs.append(glmtools.regressors.ParametricRegressor(values=age,
                                                    name='Age',
                                                    preproc='z',
                                                    num_observations=dat.info['num_observations']))
contrasts.append(glmtools.design.Contrast(name='Age',values=[0,1,0]))


# add regressor for IQ
regs.append(glmtools.regressors.ParametricRegressor(values=IQ,
                                                    name='IQ',
                                                    preproc='z',
                                                    num_observations=dat.info['num_observations']))
contrasts.append(glmtools.design.Contrast(name='IQ',values=[0,0,1]))



des = glmtools.design.GLMDesign.initialise(regs,contrasts)
model = glmtools.fit.OLSModel( des, dat )

np.save(join(outdir, 'MVAR_GLM_MODEL.npy'), model, allow_pickle=True)

#%% For establishing stastical significance we need to use two types of permutation test
# For the intercept we generate surrogate data, then permute for a null distribution
# for the predictors/regressors, we keep the data the same, and shuffle rows of the matrix independentally for each predictor

# These are computationally expensive, so we use the CBU cluster for the surrogate data, and parallelisation for the shuffle

#%% use dynamics module to create surrogate permutations, this is illustrative

null = dynamics.single_perm(type='OLS', modes=25, filter='notch', outdir=join(root_dir,'resting', 'MVAR'),
                            parcel_dir=parcel_dir, parcel_files= parcel_files, sample_rate=150,
                            glm_regs=[], perm=27, metric='partial_directed_coherence')

#%% send to cluster for multiple permutations of the surrogate data
perms = 1000

pythonpath = '/home/ai05/.conda/envs/mne_2/bin/python'
for i in range(perms):
    pycom = f"""
import sys
sys.path.insert(0, '/imaging/ai05/RED/RED_MEG/resting/analysis/RED_Rest')
from REDTools import dynamics
import numpy as np


from os import listdir as listdir
from os.path import join
import pandas as pd 

outdir = '{outdir}'
parcel_dir = '{parcel_dir}'
root_dir = '{root_dir}'
parcel_files = listdir(parcel_dir) # list them

meta_dat_path = join(root_dir, 'Combined2.csv')
meta = pd.read_csv(meta_dat_path)
tmp_id = meta.Alex_ID.to_list()# get IDS
tmp_id = [str(f).replace('_','-') for f in tmp_id] # replace to match filenames
meta.Alex_ID = tmp_id

age = []
IQ = []
ids = [i.split('_')[0] for i in parcel_files]

for id_ in ids:
    #if id is not in meta, nan it
    if id_ not in meta.Alex_ID.to_list():
        age.append(np.nan)
        IQ.append(np.nan)
    else:
        age.append(float(meta[meta.Alex_ID == id_].Age))
        IQ.append(float(meta[meta.Alex_ID == id_].WASI_Mat))

age = np.array(age)
IQ = np.array(IQ)

age[np.isnan(age)] = np.nanmean(age)
IQ[np.isnan(IQ)] = np.nanmean(IQ)

null = dynamics.single_perm(type='OLS', modes=25, filter='notch', outdir=outdir,
                            parcel_dir=parcel_dir, parcel_files= parcel_files, 
                            sample_rate=150, glm_regs=[age,IQ], perm={i}, metric='partial_directed_coherence')
                            
np.save(join(outdir, 'perm_{i}.npy'), null)

    """
    # save to file
    print(pycom, file=open(join(outdir, 'cluster_scripts', f'{i}_permscript.py'), 'w'))

    # construct csh file
    tcshf = f"""#!/bin/tcsh
            {pythonpath} {join(outdir, 'cluster_scripts', f'{i}_permscript.py')}
                    """
    # save to directory
    print(tcshf, file=open(join(outdir, 'cluster_scripts', f'{i}_permscript.csh'), 'w'))

    # execute this on the cluster
    os.system(f"sbatch --job-name=SURR_MVAR_{i} --mincpus=12 -t 0-2:00 {join(outdir, 'cluster_scripts', f'{i}_permscript.csh')}")

#%% We need to do the shuffle permutations for the regressors seperately to the surrogate data

def row_shuffle_perm(des, dat, type, dimension, perm, outdir):
    print(perm)

    if isfile(join(outdir, f'shuffleperm_{dimension}_{perm}.npy')):
        return join(outdir, f'shuffleperm_{dimension}_{perm}.npy')
    this_desmat = des.design_matrix.copy() # copy matrix
    this_des = copy.deepcopy(des) # deep copy object
    if type == 'shuffle':
        I = np.random.permutation(this_desmat.shape[0]) # indices shuffled
        this_desmat[:,dimension] = this_desmat[I,dimension] # do the shuffling
    elif type == 'sign':
        I = np.random.permutation(np.tile( [1,-1], int(this_desmat.shape[0]/2)))
        if len(I) != this_desmat.shape[0]:
            index = np.random.permutation(I)[0]
            print(index)
            I = np.append(I, I[index])
        this_desmat[:,dimension] = this_desmat[:,dimension] * I

    # replace with shuffled version
    this_des.design_matrix = this_desmat
    #run glm
    this_model = glmtools.fit.OLSModel(this_des, dat)
    #get nulls
    permuted = this_model.get_tstats()[dimension]

    np.save(join(outdir, f'shuffleperm_{dimension}_{perm}.npy'), permuted)
    return join(outdir, f'shuffleperm_{dimension}_{perm}.npy')

#%% use parellel to split up shuffle permutations
perms = 5000
for cont in [1,2]:
    saved_files = joblib.Parallel(n_jobs =10)(
        joblib.delayed(row_shuffle_perm)(des, dat, 'shuffle', cont, perm, outdir) for perm in range(perms))

#%% check permutations are there for all of them
perms = 5000
missing_surr = []
for i in range(perms):
    if isfile(join(outdir, f'perm_{i}.npy')) == False:
        missing_surr.append(i)

missing_shuffle_age = []
for i in range(perms):
    if isfile(join(outdir, f'shuffleperm_1_{i}.npy')) == False:
        missing_shuffle_age.append(i)

missing_shuffle_IQ = []
for i in range(perms):
    if isfile(join(outdir, f'shuffleperm_2_{i}.npy')) == False:
        missing_shuffle_age.append(i)
#%% mop up failed permutations for the surrogate data
# Very rarely something causes an error and some permutations don't save, so you have to manually re-run them
failed_ind = 8
null = dynamics.single_perm(type='OLS', modes=25, filter='notch', outdir=join(root_dir,'resting', 'MVAR'),
                            parcel_dir=parcel_dir, parcel_files= parcel_files, sample_rate=150, glm_regs=[age,IQ],
                            perm=failed_ind, metric='partial_directed_coherence')
np.save(join(outdir, f'perm_{failed_ind}.npy'), null)
#%% after this has completed, read the permutation data
nshape_surr = [1] + list(model.get_tstats().shape[1:4]) + [1000]
null_surr = np.zeros(nshape_surr)
for i in range(1000):
    null_surr[0,:,:,:,i] = np.load(join(outdir, f'perm_{i}.npy'))[0]

nshape_shuff = [2] + list(model.get_tstats().shape[1:4]) + [5000]
null_shuff = np.zeros(nshape_shuff)
for i in range(5000):
    null_shuff[0,:,:,:,i] = np.load(join(outdir, f'shuffleperm_1_{i}.npy'))
    null_shuff[1,:,:,:,i] = np.load(join(outdir, f'shuffleperm_2_{i}.npy'))

np.save(join(outdir, 'MVAR_GLM_NULLS_surr.npy'), null_surr, allow_pickle=True)
np.save(join(outdir, 'MVAR_GLM_NULLS_shuff.npy'), null_shuff, allow_pickle=True)
#%% get threshold
thresh = np.percentile(nulls, 95, axis=4)
sig_= model.tstats >= thresh # mask for data

# make copy with zero-out
stats = copy.deepcopy(model.tstats)
stats[~sig_] = 0

#%% get labels and coordinates for plotting in brains
# use our plotting function to look at the results
# get labels and coordinates
MAINDIR = join('/imaging/ai05/RED/RED_MEG/resting/STRUCTURALS')
labels = mne.read_labels_from_annot('fsaverage_1', parc='aparc', subjects_dir=join('/imaging/ai05/RED/RED_MEG/resting/STRUCTURALS','FS_SUBDIR'))
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
    coord = mne.vertex_to_mni(labels[i].center_of_mass(subjects_dir=join('/imaging/ai05/RED/RED_MEG/resting/STRUCTURALS','FS_SUBDIR')),
                              subject='fsaverage_1',hemis=hem,subjects_dir=join(MAINDIR,'FS_SUBDIR'))
    coords.append(coord[0])


# First, we reorder the labels based on their location in the left hemi
lh_labels = [name for name in label_names if name.endswith('lh')]
rh_labels = [name for name in label_names if name.endswith('rh')]

# Get the y-location of the label
label_ypos = list()
for name in lh_labels:
    idx = label_names.index(name)
    ypos = np.mean(labels[idx].pos[:, 1])
    label_ypos.append(ypos)

lh_labels = [label for (yp, label) in sorted(zip(label_ypos, lh_labels))]

# For the right hemi
rh_labels = [label[:-2] + 'rh' for label in lh_labels]

# make a list with circular plot order
node_order = list()
node_order.extend(lh_labels[::-1])  # reverse the order
node_order.extend(rh_labels)

node_order = node_order[::-1] # reverse the whole thing

# get a mapping to the original list
re_order_ind = [label_names.index(x) for x in node_order]
#%% Look at simple heatmaps
contrast = 0
frequency = 10

for i in range(len(freq_vect)):
    square = stats[contrast,:,:,frequency]
    fig, ax = plt.subplots()
    plotting.plot_matrix(sums, labels=label_names, colorbar=True,  axes=ax)
    plt.tight_layout()
    plt.savefig(join(figdir, f'MVAR_GLM_{contrast}_{frequency}.png'))
    plt.close(fig)

#%% plot chord diagram
# We need to get data from it's parcelxparcelxfreq form into a dataframe
# the dataframe needs to be of the form Source, target, value

contrast = 1
frequency = 30
threshold = 90

def hv_chord(contrast, frequency, threshold):
    dtypes = np.dtype([
        ('source', int),
        ('target', int),
        ('value', int),
    ])

    data = np.empty(0, dtype=dtypes)
    links = pd.DataFrame(data)

    square = stats[contrast,:,:,frequency]


    #square = stats[contrast,:,:,:].sum(axis=2)
    thresh_mask = square > np.percentile(square, threshold)
    square[~thresh_mask] = 0

    #reorder
    X_sorted = np.copy(square)
    # Sort along first dim
    X_sorted = X_sorted[re_order_ind,:]
    # Sort along second dim
    X_sorted = X_sorted[:,re_order_ind]

    labels_sorted = np.array(label_names)[re_order_ind]
    # loop through Y axis of matrix
    counter = 0
    for i in range(X_sorted.shape[0]):
        for ii in range(X_sorted.shape[1]):
            links.loc[counter] = [i, ii, int(X_sorted[i,ii])]
            counter +=1

    # make label index
    dtypes = np.dtype([
        ('name', int),
        ('group', int),
    ])

    data = np.empty(0, dtype=dtypes)
    nodes = pd.DataFrame(data)

    for i in range(X_sorted.shape[0]):
        nodes.loc[i] = [labels_sorted[i], 1]

    graph = hv.Chord((links, hv.Dataset(nodes, 'index')))
    graph.opts(
        opts.Chord(cmap='Category20', edge_cmap='Category20', edge_color=dim('source').str(),
                   labels='name', node_color=dim('index').str())
    )
    graph.relabel('Directed Graph').opts(directed=True)
    graph.opts(title=f'{des.contrast_names[contrast]} Partial Directed Coherence @ {int(freq_vect[frequency])}Hz')
    return graph

#%%
renderer = hv.renderer('bokeh')
hv.output(size=250)
#%%  try one
threshold = 90
contrast = 1
frequency = 20

graph = hv_chord(contrast, frequency, threshold).opts(directed=True,
                                                      title=f'MVAR {des.contrast_names[contrast]} {int(freq_vect[frequency])}Hz')
renderer.save(graph, join(figdir, 'chord_MVAR_test'))

#%% animated graph to represent frequency
freq_range = [0, 5, 10, 15, 20, 25, 30, 35]
start = 0
end = 36
hmap = hv.HoloMap({i: hv_chord(2, i, 90) for i in freq_range})

#%%
plot = renderer.get_plot(hmap)

#%% save
renderer.save(plot, join(figdir, 'chord_MVAR_IQ_animated'))