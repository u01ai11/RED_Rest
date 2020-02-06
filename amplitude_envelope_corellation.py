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
from REDTools import connectivity
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
#change fsaverage so they can find

#%% define function for envelope corellations
# input settings for this function and then run
i = 4
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
freqs = {'Theta':(4,7),
         'Alpha':(8,12),
         'Lower Beta': (13, 20),
         'Upper Beta':(21, 30)}

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
    joblib.delayed(connectivity.envelope_corellation)(indict) for indict in dictlist)


#%% Calculate average corellation matrix for each brain
aec_dir = join(MAINDIR, 'envelope_cors')
thetas = [f for f in listdir(aec_dir) if 'Theta' in f] # find thetas
thetas = [thetas[[i for i, s in enumerate(thetas) if i_d in s][0]] for i_d in RED_id] # reorder to match RED
#repeat for all bands
alphas =[f for f in listdir(aec_dir) if 'Alpha' in f]
alphas = [alphas[[i for i, s in enumerate(alphas) if i_d in s][0]] for i_d in RED_id]
betalow = [f for f in listdir(aec_dir) if 'Lower Beta' in f]
betalow = [betalow[[i for i, s in enumerate(betalow) if i_d in s][0]] for i_d in RED_id]
betahigh =[f for f in listdir(aec_dir) if 'Upper Beta' in f]
betahigh = [betahigh[[i for i, s in enumerate(betahigh) if i_d in s][0]] for i_d in RED_id]

# Combine each band into an average connectivity matrix
arraylist = []
for band in [thetas, alphas, betalow, betahigh]:
    #intialise empty array
    mat_3d = np.zeros(shape=(68, 68, len(band)))
    for i, f in enumerate(band):
        mat_3d[:,:,i] = np.load(join(aec_dir, f))
    arraylist.append(mat_3d)

#%% make a basic average array for one frequency
freq_labels = ['thetas', 'alphas', 'betalow', 'betahigh']
threshold = 0.12
for i in range(len(freq_labels)):
    thisarray = arraylist[i]
    corr = thisarray.mean(axis=2) # simple mean
    # try thresholding per participants
    bools = thisarray > threshold # where we are above threshold
    chn_mask = np.zeros(thisarray.shape[0:2])
    for i in range(thisarray.shape[0]): #channels where all participants above threshold
        for ii in range
    #% plot corellation matrix
    # let's plot this matrix
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(corr, cmap='viridis', clim=np.percentile(corr, [5, 95]))
    fig.tight_layout()
    plt.savefig(f'/imaging/ai05/images/AEC{freq_labels[i]}_group_mat.png')

    #% plot this on a brain
    threshold_prop = 0.3  # percentage of strongest edges to keep in the graph
    #degree = mne.connectivity.degree(corr, threshold_prop=threshold_prop) # get the degree
    degree = mne.connectivity.degree(corr) # get the degree
    #load in FSAVERAGE for parcellations and inverse for plotting
    labels = mne.read_labels_from_annot('fsaverage_1', parc='aparc',
                                        subjects_dir=join(MAINDIR, 'STRUCTURALS','FS_SUBDIR'))
    labels = labels[0:-1]
    # we know that 99144's inversion contains source from FSAVERAGE so load that
    inv_op = mne.minimum_norm.read_inverse_operator(join(invdir, '99144-inv.fif'))
    stc = mne.labels_to_stc(labels, degree)
    stc = stc.in_label(mne.Label(inv_op['src'][0]['vertno'], hemi='lh') +
                       mne.Label(inv_op['src'][1]['vertno'], hemi='rh'))
    stc.plot(
        clim=dict(kind='percent', lims=[75, 85, 95]), colormap='gnuplot',
        subjects_dir=join(STRUCTDIR, 'FS_SUBDIR'), hemi='lh',
        smoothing_steps=1, time_label=freq_labels[i], backend='matplotlib').savefig(f'/imaging/ai05/images/AEC_{freq_labels[i]}_src_group_lh.png')
    stc.plot(
        clim=dict(kind='percent', lims=[75, 85, 95]), colormap='gnuplot',
        subjects_dir=join(STRUCTDIR, 'FS_SUBDIR'), hemi='rh',
        smoothing_steps=1, time_label=freq_labels[i], backend='matplotlib').savefig(f'/imaging/ai05/images/AEC_{freq_labels[i]}_src_group_rh.png')

    #% make a circle plot
    # First, we reorder the labels based on their location in the left hemi
    label_names = [label.name for label in labels]
    lh_labels = [name for name in label_names if name.endswith('lh')]

    # Get the y-location of the label
    label_ypos = list()
    for name in lh_labels:
        idx = label_names.index(name)
        ypos = np.mean(labels[idx].pos[:, 1])
        label_ypos.append(ypos)

    # Reorder the labels based on their location
    lh_labels = [label for (yp, label) in sorted(zip(label_ypos, lh_labels))]

    # For the right hemi
    rh_labels = [label[:-2] + 'rh' for label in lh_labels]

    # Save the plot order and create a circular layout
    node_order = list()
    node_order.extend(lh_labels[::-1])  # reverse the order
    node_order.extend(rh_labels)

    node_angles = mne.viz.circular_layout(label_names, node_order, start_pos=90,
                                  group_boundaries=[0, len(label_names) / 2])
    label_colors = [label.color for label in labels]
    # Plot the graph using node colors from the FreeSurfer parcellation. We only
    # show the 300 strongest connections.
    mne.viz.plot_connectivity_circle(corr, label_names, n_lines=300,
                             node_angles=node_angles, node_colors=label_colors,
                             title=f'{freq_labels[i]} All-to-All Connectivity')[0].savefig(f'/imaging/ai05/images/AEC_{freq_labels[i]}_circle.png')


// Continous concetantion, downsample after hilbert envelope
// kanads method section of PhD -- threshold free level setting (singular value decomposition)
//