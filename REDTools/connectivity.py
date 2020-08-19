import mne
import os
import numpy as np
import joblib
from os import listdir as listdir
from os.path import join
from os.path import isdir

def get_parcel_timecourses(i,*kw):
    kw = kw[0]
    invdir = kw.get('invdir')
    rawdir = kw.get('rawdir')
    outdir = kw.get('outdir')
    fsdir = kw.get('fsdir')
    MEG_fname = kw.get('MEG_fname')
    RED_id = kw.get('RED_id')
    MR_id = kw.get('MR_id')
    method = kw.get('method')
    lambda2 =kw.get('lambda2')
    pick_ori = kw.get('pick_ori')
    parc = kw.get('parc')
    time_course_mode = kw.get('time_course_mode')
    parcel_names = kw.get('parcel_names')

    try:
        if os.path.isfile(f'{RED_id[i]}_parcel_timecourse.npy'):
            print('file exists, skip')
            return
        inv_op = mne.minimum_norm.read_inverse_operator(join(invdir, f'{RED_id[i]}-inv.fif'))
        # get raw file
        raw = mne.io.read_raw_fif(join(rawdir, MEG_fname[i]), preload=True)
        #raw.filter(freq[0],freq[1]) # beta based on Astle WM paper
        events = mne.make_fixed_length_events(raw, duration=raw.times[-1])
        epochs = mne.Epochs(raw, events=events, tmin=0, tmax=raw.times[-1],
                            baseline=None, preload=True)
        del raw
        epochs = epochs.resample(150,npad='auto')
        #hilbert of sensor level data
        #epochs.apply_hilbert()

        # parcellation, inversion and envelope corellation
        #load labels for parcellations
        labels = mne.read_labels_from_annot(MR_id[i], parc=parc,
                                            subjects_dir=fsdir)
        # invert the eopchs
        stcs = mne.minimum_norm.apply_inverse_epochs(epochs, inv_op,
                                                     lambda2=lambda2,
                                                     pick_ori=pick_ori,
                                                     method=method,
                                                     return_generator=True)
        #remove 'unknown' labels
        labels = [lab for lab in labels if 'unknown' not in lab.name]

        #filter into only our age-related anatomical labels
        labels = [lab for lab in labels if lab.name in parcel_names]
        #extract timecourses from each of those labels
        label_ts = mne.extract_label_time_course(
            stcs, labels, inv_op['src'], mode=time_course_mode)

        np.save(join(outdir, f'{RED_id[i]}_parcel_timecourse.npy'),label_ts)
    except:
        print(f'error with {RED_id[i]}')

def envelope_corellation(*kw):
    kw = kw[0]
    i = kw.get('i')
    invdir = kw.get('invdir')
    rawdir = kw.get('rawdir')
    outdir = kw.get('outdir')
    fsdir = kw.get('fsdir')
    MEG_fname = kw.get('MEG_fname')
    RED_id = kw.get('RED_id')
    MR_id = kw.get('MR_id')
    method = kw.get('method')
    combine = kw.get('combine')
    lambda2 =kw.get('lambda2')
    pick_ori = kw.get('pick_ori')
    parc = kw.get('parc')
    time_course_mode = kw.get('time_course_mode')
    freq = kw.get('freq')
    freqname = kw.get('freqname')

    if os.path.isfile(join(outdir, f'{RED_id[i]}_{freqname}_aec.npy')):
        print(f'{RED_id[i]}_{freqname}_aec.npy already exists - skipping')
        return None
    inv_op = mne.minimum_norm.read_inverse_operator(join(invdir, f'{RED_id[i]}-inv.fif'))
    # get raw file
    raw = mne.io.read_raw_fif(join(rawdir, MEG_fname[i]), preload=True)
    raw.filter(freq[0],freq[1]) # beta based on Astle WM paper
    events = mne.make_fixed_length_events(raw, duration=10)
    epochs = mne.Epochs(raw, events=events, tmin=0, tmax=10,
                        baseline=None, preload=True)
    del raw
    epochs = epochs.resample(150,npad='auto')
    #hilbert of sensor level data
    epochs.apply_hilbert()

    # parcellation, inversion and envelope corellation
    #load labels for parcellations
    labels = mne.read_labels_from_annot(MR_id[i], parc=parc,
                                        subjects_dir=fsdir)
    # invert the eopchs
    stcs = mne.minimum_norm.apply_inverse_epochs(epochs, inv_op,
                                                 lambda2=lambda2,
                                                 pick_ori=pick_ori,
                                                 method=method,
                                                 return_generator=True)
    #remove 'unknown' labels
    labels = [lab for lab in labels if 'unknown' not in lab.name]
    #extract timecourses from each of those labels
    label_ts = mne.extract_label_time_course(
        stcs, labels, inv_op['src'], return_generator=True, mode=time_course_mode)

    corr = mne.connectivity.envelope_correlation(label_ts, verbose=True, orthogonalize='pairwise', combine=combine)
    np.save(join(outdir, f'{RED_id[i]}_{freqname}_aec.npy'),corr)

def cluster_parcel_timecourses(pythonpath,logdir,*kw):
    kw = kw[0]
    invdir = kw.get('invdir')
    rawdir = kw.get('rawdir')
    outdir = kw.get('outdir')
    fsdir = kw.get('fsdir')
    MEG_fname = kw.get('MEG_fname')
    RED_id = kw.get('RED_id')
    MR_id = kw.get('MR_id')
    method = kw.get('method')
    lambda2 =kw.get('lambda2')
    pick_ori = kw.get('pick_ori')
    parc = kw.get('parc')
    time_course_mode = kw.get('time_course_mode')
    parcel_names = kw.get('parcel_names')

    for i in range(len(RED_id)):
        pycom = f"""
import mne
import os
import numpy as np
import joblib
from os.path import join

invdir = "{invdir}"
rawdir = "{rawdir}"
outdir = "{outdir}"
fsdir = "{fsdir}"
MEG_fname = {MEG_fname}
RED_id = {RED_id}
MR_id = {MR_id}
method = "{method}"
lambda2 = "{lambda2}"
pick_ori = "{pick_ori}"
parc = "{parc}"
time_course_mode = "{time_course_mode}"
parcel_names = {parcel_names}
inv_op = mne.minimum_norm.read_inverse_operator(join(invdir, f'{RED_id[i]}-inv.fif'))
# get raw file
raw = mne.io.read_raw_fif(join(rawdir, MEG_fname[{i}]), preload=True)
#raw.filter(freq[0],freq[1]) # beta based on Astle WM paper
events = mne.make_fixed_length_events(raw, duration=raw.times[-1])
epochs = mne.Epochs(raw, events=events, tmin=0, tmax=raw.times[-1],
                    baseline=None, preload=True)
del raw
epochs = epochs.resample(150,npad='auto')
#hilbert of sensor level data
#epochs.apply_hilbert()

# parcellation, inversion and envelope corellation
#load labels for parcellations
labels = mne.read_labels_from_annot(MR_id[{i}], parc=parc,
                                    subjects_dir=fsdir)
# invert the eopchs
stcs = mne.minimum_norm.apply_inverse_epochs(epochs, inv_op,
                                             lambda2=lambda2,
                                             pick_ori=pick_ori,
                                             method=method,
                                             return_generator=True)
#remove 'unknown' labels
labels = [lab for lab in labels if 'unknown' not in lab.name]

#filter into only our age-related anatomical labels
labels = [lab for lab in labels if lab.name in parcel_names]
#extract timecourses from each of those labels
label_ts = mne.extract_label_time_course(
    stcs, labels, inv_op['src'], mode=time_course_mode)

np.save(join(outdir, f'{RED_id[i]}_parcel_timecourse.npy'),label_ts)"""

        # save to file
        print(pycom, file=open(join(logdir, f'{i}_par_t.py'), 'w'))

        # construct csh file
        tcshf = f"""#!/bin/tcsh
            {pythonpath} {join(logdir, f'{i}_par_t.py')}
                    """
        # save to directory
        print(tcshf, file=open(join(logdir, f'{i}_par_t.csh'), 'w'))

        # execute this on the cluster
        os.system(f"sbatch --job-name=AmpEnvConnect_{i} --mincpus=4 -t 0-3:00 {join(logdir, f'{i}_par_t.csh')}")

def cluster_envelope_corellation(MR_id, MAINDIR):
    pythonpath = '/home/ai05/.conda/envs/mne_2/bin/python'
    for i in range(len(MR_id)):
        pycom = f"""
import sys
import os
from os import listdir as listdir
from os.path import join
from os.path import isdir
import numpy as np
import dicom2nifti
import mne
import matplotlib.pyplot as plt
sys.path.insert(0, '/imaging/ai05/RED/RED_MEG/resting/analysis/RED_Rest')
from REDTools import sourcespace_setup
from REDTools import sourcespace_command_line
from REDTools import study_info
import joblib
#%%

#located in imaging/rs04/RED/DICOMs

# The T1s DICOMS are located in a folder in a folder for each participants
# The name of this folder is the same for each participant
MP_name = 'Series005_CBU_MPRAGE_32chn'
dicoms = '/imaging/rs04/RED/DICOMs'
STRUCTDIR = '/imaging/ai05/RED/RED_MEG/resting/STRUCTURALS'
MAINDIR = '/imaging/ai05/RED/RED_MEG/resting'
folders = [f for f in os.listdir(dicoms) if os.path.isdir(os.path.join(dicoms,f))]
ids = [f.split('_')[0] for f in folders]
#get ids etc
RED_id, MR_id, MEG_fname = study_info.get_info()
#%% define function for envelope corellations
def envelope_corellation(*kw):
    kw = kw[0]
    i = kw.get('i')
    invdir = kw.get('invdir')
    rawdir = kw.get('rawdir')
    outdir = kw.get('outdir')
    fsdir = kw.get('fsdir')
    MEG_fname = kw.get('MEG_fname')
    RED_id = kw.get('RED_id')
    MR_id = kw.get('MR_id')
    method = kw.get('method')
    combine = kw.get('combine')
    lambda2 =kw.get('lambda2')
    pick_ori = kw.get('pick_ori')
    parc = kw.get('parc')
    time_course_mode = kw.get('time_course_mode')
    freq = kw.get('freq')
    freqname = kw.get('freqname')
    
    if os.path.isfile(join(outdir, f'{{RED_id[i]}}_{{freqname}}_aec.npy')):
        print(f'{{RED_id[i]}}_{{freqname}}_aec.npy already exists - skipping')
        return None
    inv_op = mne.minimum_norm.read_inverse_operator(join(invdir, f'{{RED_id[i]}}-inv.fif'))
    # get raw file
    raw = mne.io.read_raw_fif(join(rawdir, MEG_fname[i]), preload=True)
    raw.filter(freq[0],freq[1]) # beta based on Astle WM paper
    raw = raw.resample(150, npad='auto')
    events = mne.make_fixed_length_events(raw, duration=10)
    epochs = mne.Epochs(raw, events=events, tmin=0, tmax=10,
                        baseline=None, preload=True)
    del raw
    
    #hilbert of sensor level data
    epochs.apply_hilbert()

    # parcellation, inversion and envelope corellation
    #load labels for parcellations
    labels = mne.read_labels_from_annot(MR_id[i], parc=parc,
                                        subjects_dir=fsdir)
    #remove 'unknown' labels
    labels = [lab for lab in labels if 'unknown' not in lab.name]
    
    # invert the eopchs
    stcs = mne.minimum_norm.apply_inverse_epochs(epochs, inv_op,
                                                 lambda2=lambda2,
                                                 pick_ori=pick_ori,
                                                 method=method,
                                                 return_generator=True)
    #extract timecourses from each of those labels
    label_ts = mne.extract_label_time_course(
        stcs, labels, inv_op['src'], return_generator=True, mode=time_course_mode)

    corr = mne.connectivity.envelope_correlation(label_ts, verbose=True, orthogonalize='pairwise', combine=combine)
    np.save(join(outdir, f'{{RED_id[i]}}_{{freqname}}_aec.npy'),corr)
    
# input settings for this function
i = {i}
invdir = join(MAINDIR, 'inverse_ops')
rawdir = join(MAINDIR, 'preprocessed')
outdir = join(MAINDIR, 'envelope_cors')
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

#frequencies we are gonna do
freqs = {{'Theta_c':(4,7),
     'Alpha_c':(8,12),
     'Lower Beta_c': (13, 20),
     'Upper Beta_c':(21, 30)}}

# make list of dicts for input (we're gonna use job lib for this)
dictlist = []
for f in freqs:
    freq = freqs[f]
    freqname = f
    dictlist.append(dict(
        i = i,
        invdir = invdir,
        rawdir = rawdir,
        outdir = outdir,
        fsdir = fsdir,
        MEG_fname = MEG_fname,
        RED_id = RED_id,
        MR_id = MR_id,
        method = method,
        combine = combine,
        lambda2 =lambda2,
        pick_ori = pick_ori,
        parc = parc,
        time_course_mode = time_course_mode,
        freq = freq,
        freqname = freqname,
    ))

joblib.Parallel(n_jobs=len(dictlist))(
joblib.delayed(envelope_corellation)(indict) for indict in dictlist)
        """
        # save to file
        print(pycom, file=open(join(MAINDIR, 'cluster_scripts', f'{i}_aec.py'), 'w'))

        # construct csh file
        tcshf = f"""#!/bin/tcsh
            {pythonpath} {join(MAINDIR, 'cluster_scripts', f'{i}_aec.py')}
                    """
        # save to directory
        print(tcshf, file=open(join(MAINDIR, 'cluster_scripts', f'{i}_aec.csh'), 'w'))

        # execute this on the cluster
        os.system(f"sbatch --job-name=AmpEnvConnect_{i} --mincpus=4 -t 0-3:00 {join(MAINDIR, 'cluster_scripts', f'{i}_aec.csh')}")

def ACE_cluster_envelope_corellation(MR_id, MAINDIR):
    pythonpath = '/home/ai05/.conda/envs/mne_2/bin/python'
    for i in range(len(MR_id)):
        pycom = f"""
import sys
import os
from os import listdir as listdir
from os.path import join
from os.path import isdir
import numpy as np
import dicom2nifti
import mne
import matplotlib.pyplot as plt
sys.path.insert(0, '/imaging/ai05/RED/RED_MEG/resting/analysis/RED_Rest')
from REDTools import sourcespace_setup
from REDTools import sourcespace_command_line
from REDTools import study_info
import joblib
#%%

#located in imaging/rs04/RED/DICOMs

# The T1s DICOMS are located in a folder in a folder for each participants
# The name of this folder is the same for each participant


MAINDIR = '/imaging/ai05/RED/RED_MEG/ace_resting'

#get ids etc
MEG_id, MR_id, MEG_fname = study_info.get_info_ACE()
#%% define function for envelope corellations
def envelope_corellation(*kw):
    kw = kw[0]
    i = kw.get('i')
    invdir = kw.get('invdir')
    rawdir = kw.get('rawdir')
    outdir = kw.get('outdir')
    fsdir = kw.get('fsdir')
    MEG_fname = kw.get('MEG_fname')
    RED_id = kw.get('RED_id')
    MR_id = kw.get('MR_id')
    method = kw.get('method')
    combine = kw.get('combine')
    lambda2 =kw.get('lambda2')
    pick_ori = kw.get('pick_ori')
    parc = kw.get('parc')
    time_course_mode = kw.get('time_course_mode')
    freq = kw.get('freq')
    freqname = kw.get('freqname')
    
    if os.path.isfile(join(outdir, f'{{RED_id[i]}}_{{freqname}}_aec.npy')):
        print(f'{{RED_id[i]}}_{{freqname}}_aec.npy already exists - skipping')
        return None
    inv_op = mne.minimum_norm.read_inverse_operator(join(invdir, f'{{RED_id[i]}}_inv.fif'))
    # get raw file
    raw = mne.io.read_raw_fif(join(rawdir, MEG_fname[i]), preload=True)
    raw.filter(freq[0],freq[1]) # beta based on Astle WM paper
    raw = raw.resample(150, npad='auto')
    events = mne.make_fixed_length_events(raw, duration=10)
    epochs = mne.Epochs(raw, events=events, tmin=0, tmax=10,
                        baseline=None, preload=True)
    del raw
    #hilbert of sensor level data
    epochs.apply_hilbert()

    # parcellation, inversion and envelope corellation
    #load labels for parcellations
    labels = mne.read_labels_from_annot(MR_id[i], parc=parc,
                                        subjects_dir=fsdir)
    #remove 'unknown' labels
    labels = [lab for lab in labels if 'unknown' not in lab.name]
    
    # invert the eopchs
    stcs = mne.minimum_norm.apply_inverse_epochs(epochs, inv_op,
                                                 lambda2=lambda2,
                                                 pick_ori=pick_ori,
                                                 method=method,
                                                 return_generator=True)
    #extract timecourses from each of those labels
    label_ts = mne.extract_label_time_course(
        stcs, labels, inv_op['src'], return_generator=True, mode=time_course_mode)

    corr = mne.connectivity.envelope_correlation(label_ts, verbose=True, orthogonalize='pairwise', combine=combine)
    np.save(join(outdir, f'{{RED_id[i]}}_{{freqname}}_aec.npy'),corr)
    
# input settings for this function
i = {i}
invdir = join(MAINDIR, 'invs')
rawdir = join(MAINDIR, 'preprocessed')
outdir = join(MAINDIR, 'envelope_cors')
fsdir = join(MAINDIR,'FS_SUBDIR')
MEG_fname = MEG_fname
RED_id = MEG_id
MR_id = MR_id
method = 'MNE'
combine = 'mean'
lambda2 = 1. / 9.
pick_ori = 'normal'
parc = 'aparc'
time_course_mode = 'mean'

#frequencies we are gonna do
freqs = {{'Theta_c':(4,7),
     'Alpha_c':(8,12),
     'Lower Beta_c': (13, 20),
     'Upper Beta_c':(21, 30)}}

# make list of dicts for input (we're gonna use job lib for this)
dictlist = []
for f in freqs:
    freq = freqs[f]
    freqname = f
    dictlist.append(dict(
        i = i,
        invdir = invdir,
        rawdir = rawdir,
        outdir = outdir,
        fsdir = fsdir,
        MEG_fname = MEG_fname,
        RED_id = RED_id,
        MR_id = MR_id,
        method = method,
        combine = combine,
        lambda2 =lambda2,
        pick_ori = pick_ori,
        parc = parc,
        time_course_mode = time_course_mode,
        freq = freq,
        freqname = freqname,
    ))

joblib.Parallel(n_jobs=len(dictlist))(
joblib.delayed(envelope_corellation)(indict) for indict in dictlist)
        """
        # save to file
        print(pycom, file=open(join(MAINDIR, 'cluster_scripts', f'{i}_aec.py'), 'w'))

        # construct csh file
        tcshf = f"""#!/bin/tcsh
            {pythonpath} {join(MAINDIR, 'cluster_scripts', f'{i}_aec.py')}
                    """
        # save to directory
        print(tcshf, file=open(join(MAINDIR, 'cluster_scripts', f'{i}_aec.csh'), 'w'))

        # execute this on the cluster
        os.system(f"sbatch --job-name=AmpEnvConnect_{i} --mincpus=4 -t 0-3:00 {join(MAINDIR, 'cluster_scripts', f'{i}_aec.csh')}")

