import sys
import os
from os import listdir as listdir
from os.path import join
import numpy as np
sys.path.insert(0, '/imaging/ai05/RED/RED_MEG/resting/analysis/RED_Rest')
from REDTools import study_info
from REDTools import connectivity
import joblib
import pandas as pd
import mne
#%%
root_dir = '/imaging/ai05/RED/RED_MEG'
MP_name = 'Series005_CBU_MPRAGE_32chn'
dicoms = '/imaging/rs04/RED/DICOMs'
STRUCTDIR = '/imaging/ai05/RED/RED_MEG/resting/STRUCTURALS'
MAINDIR = '/imaging/ai05/RED/RED_MEG/resting'
ACEDIR = '/imaging/ai05/RED/RED_MEG/ace_resting'
folders = [f for f in os.listdir(dicoms) if os.path.isdir(os.path.join(dicoms,f))]
ids = [f.split('_')[0] for f in folders]

#get ids etc
RED_id, MR_id, MEG_fname = study_info.get_info()
#change fsaverage so they can find

# Also get these IDs from Amy's study

#%%
"""
Looking at the dynamics of different nodes in our network. 

Corellation of time courses accross regions
Coupling
Coherence analysis, what frequencies?
Circular-linear corellatons 
Nested rhythms/phase-amplitude coupling
Cross frequency coupling 

"""

#%% Calculate parcel timecourses for RED MEG data
invdir = join(MAINDIR, 'inverse_ops')
rawdir = join(MAINDIR, 'preprocessed')
outdir = join(MAINDIR, 'parcel_timecourses')
fsdir = join(MAINDIR, 'STRUCTURALS','FS_SUBDIR')
MEG_fname = MEG_fname
RED_id = RED_id
MR_id = MR_id
method = 'MNE'
combine = 'mean'
lambda2 = 1. / 9.
pick_ori = 'normal'
parc = 'aparc'
time_course_mode = 'mean'
#age_node_names = np.load('/imaging/ai05/RED/RED_MEG/resting/analysis/RED_Rest/age_node_names.npy')

labels = mne.read_labels_from_annot('fsaverage_1', parc='aparc',
                                    subjects_dir=join(STRUCTDIR,'FS_SUBDIR'))
labels = labels[0:-1]
label_names = [label.name for label in labels]


kw = {
    'invdir': invdir,
    'rawdir': rawdir,
    'outdir': outdir,
    'fsdir': fsdir,
    'MEG_fname': MEG_fname,
    'RED_id': RED_id,
    'MR_id': MR_id,
    'method': method,
    'combine': combine,
    'lambda2': lambda2,
    'pick_ori': pick_ori,
    'parc': parc,
    'time_course_mode': time_course_mode,
    'parcel_names': label_names
}

# connectivity.cluster_parcel_timecourses('/home/ai05/.conda/envs/mne_2/bin/python',
#                                         join(MAINDIR,'dump'),
#                                         kw)
# saved_files = joblib.Parallel(n_jobs =njobs)(
#     joblib.delayed(__preprocess_individual)(os.path.join(indir, thisF), outdir, overwrite) for thisF in flist)

joblib.Parallel(n_jobs =10)(
    joblib.delayed(connectivity.get_parcel_timecourses)(i, kw) for i in range(len(RED_id)))

#%% Calculate parcel timecourses for ACE MEG data

# set folders to find ACE data
invdir = join(ACEDIR, 'invs')
rawdir = join(ACEDIR, 'preprocessed')
fsdir = join(ACEDIR,'FS_SUBDIR')

# read metadata file to get participant info
meta_dat_path = join(root_dir, 'Combined2.csv')
meta = pd.read_csv(meta_dat_path)
amymeta = meta[meta.Study == 1]


ACE_id = amymeta.Alex_ID.to_list()
ACE_id = [f for f in ACE_id if ~np.isnan(float(f))]
ACE_megfname = [[f for f in listdir(rawdir) if str(x).replace('_', '-') in f] for x in ACE_id]
ACE_mri_id = [[f for f in listdir(fsdir) if str(x).split('_')[1] in f] for x in ACE_id]

in_id = []
in_megfname = []
in_mri_id = []
for i in range(len(ACE_id)):
    if len(ACE_megfname[i]) == 0:
        continue
    else:
        in_id.append(ACE_id[i].replace('_','-'))
        in_megfname.append(ACE_megfname[i][0])

        if len(ACE_mri_id[i]) == 0:
            in_mri_id.append('fsaverage_1')
        else:
            in_mri_id.append(ACE_mri_id[i][0])


kw = {
    'invdir': invdir,
    'rawdir': rawdir,
    'outdir': outdir,
    'fsdir': fsdir,
    'MEG_fname': in_megfname,
    'RED_id': in_id,
    'MR_id': in_mri_id,
    'method': method,
    'combine': combine,
    'lambda2': lambda2,
    'pick_ori': pick_ori,
    'parc': parc,
    'time_course_mode': time_course_mode,
    'parcel_names': label_names
}

joblib.Parallel(n_jobs =10)(
    joblib.delayed(connectivity.get_parcel_timecourses)(i, kw) for i in range(len(in_id)))



