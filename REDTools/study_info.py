import numpy as np
import mne
import joblib # for mne multithreading
import os
from os import listdir
from os.path import isfile, join
import pandas as pd
"""
This is a script for coregistration manually 
"""
#%%
# pointers to our directories

def get_info():
    RAWDIR = '/imaging/ai05/RED/RED_MEG/resting/preprocessed'  # raw fifs to input
    STRUCTDIR = '/imaging/ai05/RED/RED_MEG/resting/STRUCTURALS'
    FS_SUBDIR = join(STRUCTDIR, 'FS_SUBDIR')
    #get the mri 2 RED ids
    mri2red = np.genfromtxt(join(STRUCTDIR, 'mr2red.csv'),delimiter=',',dtype=None)
    # get list of files in FS dur
    All_MR_models = [f for f in listdir(FS_SUBDIR) if 'CBU' in f]
    All_resting_scans = [f for f in listdir(RAWDIR) if 'clean' in f]
    # loop through this and find the matching red ids
    # also check that a raw file exists for us to coreg on
    MR_id = []
    RED_id = []
    MEG_fname = []
    for fname in All_resting_scans:
        #extract RED_id
        #find matching RED id
        red_id = fname.split('_')[0]
        if len(red_id) != 5:
            print(f'{red_id} does not seem right, skipping' )
            continue
        tmp_row = mri2red[np.where(mri2red[:, 0] == red_id)]
        try:
            # find matching MRI ID
            scans = [f for f in All_MR_models if tmp_row[0][1] in f]
            if len(scans) > 0:
                RED_id.append(red_id)
                MR_id.append(scans[0])
                MEG_fname.append(fname)
            else:
                print(f'{red_id} no structurals found found {scans}')
                RED_id.append(red_id)
                MR_id.append('FSAVERAGE')
                MEG_fname.append(fname)
        except:
            print(tmp_row)
    return RED_id, MR_id, MEG_fname

def get_info_ACE():
    RAWDIR = '/imaging/ai05/RED/RED_MEG/ace_resting/preprocessed'  # raw fifs to input
    FS_SUBDIR = '/imaging/ai05/RED/RED_MEG/ace_resting/FS_SUBDIR'
    #get the mri 2 RED ids
    path_amyfile = '/imaging/ai05/phono_oddball/complete_amy.csv'
    df = pd.read_csv(path_amyfile);
    # get list of files in FS dur
    All_MR_models = [f for f in listdir(FS_SUBDIR) if '0' in f]
    All_resting_scans = [f for f in listdir(RAWDIR) if 'clean' in f]
    # loop through this and find the matching red ids
    # also check that a raw file exists for us to coreg on
    MR_id = []
    MEG_id = []
    MEG_fname = []
    for fname in All_resting_scans:
        #extract RED_id
        #find matching RED id
        meg_id = fname.split('_')[0]
        try:
            # find matching MRI ID
            scans = [f for f in All_MR_models if meg_id.split('-')[1] in f]
            if len(scans) > 0:
                MEG_id.append(meg_id)
                MR_id.append(scans[0])
                MEG_fname.append(fname)
            else:
                print(f'{meg_id} no structurals found found {scans}')
                MEG_id.append(meg_id)
                MR_id.append('fsaverage_1')
                MEG_fname.append(fname)
        except:
            print('error')
    return MEG_id, MR_id, MEG_fname