# GLM to predict AEC network connections
# Alex Anwyl-Irvine 2020
# Alexander.Irvine@mrc-cbu.cam.ac.uk

#%% Import stuff
import pandas as pd
import sys
from os import listdir as listdir
from os.path import join
import numpy as np
sys.path.insert(0, '/imaging/ai05/RED/RED_MEG/resting/analysis/RED_Rest')
from REDTools import study_info
import mne
import matplotlib.pyplot as plt
from nilearn import plotting
sys.path.insert(0, '/home/ai05/Downloads/glm')
import glmtools
import scipy.stats as ss
from copy import deepcopy
import scipy

#%% Start with ACE networks, as we have matched behavioural measures for this data
# We will only use the high beta, as this has the most accurate AEC networks


# sum of all possible paths between connections in strucutral connectome is related to functional connectome

"""
https://en.wikipedia.org/wiki/Wilson–Cowan_model 
From Kanad Mandke to Everyone: (2:11 pm)
 Tewarie, P. K., Bright, M. G., Hillebrand, A., Robson, S. E., Gascoyne, L. E., Morris, P. G., … Brookes, M. J. (2016). Predicting haemodynamic networks using electrophysiology: The role of non-linear and cross-frequency interactions. NeuroImage, 130, 273–292. https://doi.org/10.1016/j.neuroimage.2016.01.053 Hunt, B. A. E., Tewarie, P. K., Mougin, O. E., Geades, N., Jones, D. K., Singh, K. D., … Brookes, M. J. (2016). Relationships between cortical myeloarchitecture and electrophysiological networks. Proceedings of the National Academy of Sciences, 113(47), 13510–13515. https://doi.org/10.1073/pnas.1608587113 
From Edwin Dalmaijer to Everyone: (2:22 pm)
 https://doi.org/10.1038/ncomms10340 
"""

# Get info
# Behavioural data
path_amyfile = '/imaging/ai05/phono_oddball/complete_amy.csv'
df = pd.read_csv(path_amyfile);
# paths to stuff
MAINDIR = '/imaging/ai05/RED/RED_MEG/ace_resting'

# Load in information so they match up
MEG_id, MR_id, MEG_fname = study_info.get_info_ACE()

aec_dir = join(MAINDIR, 'envelope_cors') # directory of matrices
betahigh =[f for f in listdir(aec_dir) if 'Upper Beta' in f] # find upper betas
nMEG_id = [i for i in MEG_id if i in [ii.split('_')[0] for ii in betahigh]] # only choose ids with files
betahigh = [betahigh[[i for i, s in enumerate(betahigh) if i_d in s][0]] for i_d in nMEG_id] # get them in order

# Read into one big matrix nparcels x nparcels x participants
mat_3d = np.zeros(shape=(len(betahigh), 68, 68, ))
for i, f in enumerate(betahigh):
    mat_3d[i,:,:] = np.load(join(aec_dir, f))

#%%
#choose behavioural predictors
predictors = ['Age', 'WM_exec', 'Classic_IQ', 'verbalstm_wm', 'education']
#Z-score the behavioural predictors
preds = np.zeros(shape=(len(nMEG_id), len(predictors)))
for i, megid in enumerate(nMEG_id):
    df_id = 'meg'+ megid.split('-b')[0].replace('-','_')
    for ii, p in enumerate(predictors):
        preds[i, ii] = df.loc[df['MEG ID'] == df_id][p].values[0]

#%%
# transpose to regressor matrix
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
dat = glmtools.data.TrialGLMData(data=mat_3d, dim_labels=['participants', 'parcels_1', 'parcels_2'])

# add regressor for intercept
regs = list()
regs.append(glmtools.regressors.ConstantRegressor(num_observations=mat_3d.shape[0]))
# make contrasts and add intercept
contrasts = [glmtools.design.Contrast(name='Intercept',values=[1] + [0]*regmat.shape[0])]
#loop through continous regreessors and add (also to the info and contrasts)
for i in range(regmat.shape[0]):
    regs.append(glmtools.regressors.ParametricRegressor(values=regmat[i],
                                                        name=predictors[i],
                                                        preproc='z',
                                                        num_observations=mat_3d.shape[0]))
    # add covariate to info
    dat.info[predictors[i]] = regmat[i]
    values = [0] * (regmat.shape[0] +1)
    values[i+1] = 1
    contrasts.append(glmtools.design.Contrast(name=predictors[i],values=values))

# contrasts
des = glmtools.design.GLMDesign.initialise(regs,contrasts)


model = glmtools.fit.OLSModel( des, dat )

#%%
nperms = 5000
nulls = np.zeros(shape=(nperms, model.get_tstats().shape[0], model.get_tstats().shape[1], model.get_tstats().shape[2]))
modes = ['shuffle']*nulls.shape[1]
modes[0] = 'sign'
for i in range(nulls.shape[1]):
    for ii in range(nperms):
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
                print(index)
                I = np.append(I, I[index])
            this_desmat[:,i] = this_desmat[:,i] * I

        # replace with shuffled version
        this_des.design_matrix = this_desmat
        #run glm
        this_model = glmtools.fit.OLSModel(this_des, dat)
        #pipe stats into nulls
        nulls[ii,i,:] = this_model.get_tstats()[i]

#%% calculate the centiles
thresh = 5
threshs = np.zeros(shape=(2,nulls.shape[1]))
for i in range(nulls.shape[1]):
    threshs[0,i] = np.percentile(nulls[:,i,:,:].flatten(),100-thresh)
    threshs[1,i] = np.percentile(nulls[:,i,:,:].flatten(),thresh)

#%% look at results
stats = model.get_tstats()
sig_mask = np.empty(shape=(stats.shape))
for i in range(stats.shape[0]):
    sig_mask[i] = (stats[i] > threshs[0,i]) | (stats[i] < threshs[1,i])

#%% setup stuff for glass plots
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


#%% Plot the outputs
fig, axs = plt.subplots(nrows=2, ncols=stats.shape[0],
                        gridspec_kw = {'height_ratios':[1.5,1]})
ims = []
for i in range(stats.shape[0]):
    name = model.contrast_names[i]
    ims.append(axs[0,i].imshow(stats[i])) # plot the matrix
    axs[0,i].imshow(sig_mask[i],alpha=0.6, cmap='gray')#plot the significane overlay
    axs[0,i].set_title(name)

    plotting.plot_connectome(sig_mask[i], coords,
                             edge_threshold=0.9,
                             title=f'{name}',
                             figure=fig,
                             axes=axs[1,i],
                             node_size=2)

for i, im in enumerate(ims):
    fig.colorbar(im,ax=axs[0,i])
    axs[0,i].set_aspect('auto')

fig.set_size_inches(35.0, 12)
#plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
fig.savefig('/imaging/ai05/images/net_glm_thresh_01.png')