import numpy as np
import mne
import joblib # for mne multithreading
import os
from os import listdir
from os.path import isfile, join

"""
This is a script for coregistration manually 
"""
#%%
# pointers to our directories
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
#%%
#sigh, we have to do this because of poor openGL on cluster nodes
os.system("tcsh -c 'setenv MESA_GL_VERSION_OVERRIDE 3.3'")
coreg_list = []
for REDid, MEGf, MRid in zip(RED_id, MEG_fname, MR_id):
    #check if coreg already exists
    #get filename
    trans_f_name = join(STRUCTDIR, 'FS_SUBDIR', MRid, 'bem', f'{MR_id}-trans.fif')
    if os.path.isfile(trans_f_name): # if already existing
        print(f'{RED_id} already coregistered')
        coreg_list.append(trans_f_name)
        continue  # skip this one

    if MR_id != 'FSAVERAGE': # if participant has own MRI
        mne.gui.coregistration(inst=join(RAWDIR, MEGf), subject=MRid, subjects_dir=FS_SUBDIR, advanced_rendering=False)
    else: # do fs average
        mne.gui.coregistration(inst=join(RAWDIR, MEGf), subjects_dir=FS_SUBDIR,advanced_rendering=False)
    
    coreg_list.append(trans_f_name)