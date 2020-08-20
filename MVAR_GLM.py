import sys
from os import listdir as listdir
from os.path import join
from os.path import isfile
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, '/imaging/ai05/RED/RED_MEG/resting/analysis/RED_Rest')
from REDTools import dynamics
import mne
import sails
import glmtools
import joblib
import scipy
import random
import copy
import pandas as pd
import os

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
glm_data = np.empty((len(parcel_files), 12,12,36))
for i in range(len(parcel_files)):
    id_ = parcel_files[i].split('_')[0]
    glm_data[i,:,:,:] = np.load(join(root_dir,'resting', 'MVAR', f'mvar_OLS_{id_}.npy'))[:,:,:,0]



#%% feed into GLM

dat = glmtools.data.TrialGLMData(data=glm_data, dim_labels=['participants', 'parcel_drivers', 'parcel_recievers', 'frequency'])

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
contrasts.append(glmtools.design.Contrast(name='Age',values=[0,0,1]))



des = glmtools.design.GLMDesign.initialise(regs,contrasts)
model = glmtools.fit.OLSModel( des, dat )



#%% use dynamics module to create surrogate permutations

null = dynamics.single_perm(type='OLS', modes=25, filter='notch', outdir=join(root_dir,'resting', 'MVAR'),
                            parcel_dir=parcel_dir, parcel_files= parcel_files, sample_rate=150, glm_regs=[age,IQ],perm=27)

#%% send to cluster for multiple permutations
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
                            parcel_dir=parcel_dir, parcel_files= parcel_files, sample_rate=150, glm_regs=[age,IQ], perm={i})
                            
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
    os.system(f"sbatch --job-name=SURR_MVAR_{i} --mincpus=4 -t 0-3:00 {join(outdir, 'cluster_scripts', f'{i}_permscript.csh')}")

#%% after this has completed, read the permutation data
perms =1000
nshape = list(model.get_tstats().shape)
nshape.append(perms)
nulls = np.zeros(nshape)
for i in range(perms):
    nulls[:,:,:,:,i] = np.load(join(outdir, f'perm_{i}.npy'))

#%% mop up failed permutations
# Very rarely something causes an error and some permutations don't save, so you have to manually re-run them
failed_ind = 500
null = dynamics.single_perm(type='OLS', modes=25, filter='notch', outdir=join(root_dir,'resting', 'MVAR'),
                            parcel_dir=parcel_dir, parcel_files= parcel_files, sample_rate=150, glm_regs=[age,IQ],perm=failed_ind)
np.save(join(outdir, f'perm_{failed_ind}.npy'), null)

#%% get threshold
thresh = np.percentile(nulls, 95, axis=4)
sig_= model.tstats >= thresh # mask for data

# make copy with zero-out
stats = copy.deepcopy(model.tstats)
stats[~sig_] = 0


# Get sums for each area (outflow and inflow)
sums = stats[1,:,:,:].sum(axis=2)

plt.
#%% plot thresholded results

#fig = sails.plotting.plot_vector(stats.transpose([1,2,3,0]), freq_vect, diag=True)

#%% Intercept
fig = sails.plotting.plot_vector(stats.transpose([1,2,3,0])[:,:,:,0], freq_vect, diag=True)

ax = plt.gca()

plt.savefig(join(figdir, f'glm_premuted{perms}_group_intercept.png'), bbox_inches='tight')
plt.close(fig)

#%% Age
fig = sails.plotting.plot_vector(stats.transpose([1,2,3,0])[:,:,:,1], freq_vect, diag=True)

ax = plt.gca()

plt.savefig(join(figdir, f'glm_premuted{perms}_group_age.png'), bbox_inches='tight')
plt.close(fig)

#%% IQ
fig = sails.plotting.plot_vector(stats.transpose([1,2,3,0])[:,:,:,2], freq_vect, diag=True)

ax = plt.gca()

plt.savefig(join(figdir, f'glm_premuted{perms}_group_IQ.png'), bbox_inches='tight')
plt.close(fig)

#%% thresh intercept for reference
fig = sails.plotting.plot_vector(thresh.transpose([1,2,3,0])[:,:,:,0], freq_vect, diag=True)

ax = plt.gca()
ax.set_ylim(0,20)
fig.tight_layout()
plt.savefig(join(figdir, f'glm_nullthresh_group_intercept.png'), bbox_inches='tight')
plt.close(fig)