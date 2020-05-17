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
freqs = {'Theta_s':(4,7),
         'Alpha_s':(8,12),
         'Lower Beta_s': (13, 20),
         'Upper Beta_s':(21, 30)}

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

#%% or instead use cluster version
connectivity.cluster_envelope_corellation(MR_id, MAINDIR)

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
threshold = 0.05 # threshold for strength
perc_part = 0.70 # Threshold for how many participants
for i in range(len(freq_labels)):
    thisarray = arraylist[i]
    corr = thisarray.mean(axis=2) # simple mean
    # try thresholding per participants
    bools = thisarray > threshold # where we are above threshold
    chn_mask = np.zeros(thisarray.shape[0:2], dtype=bool)
    corr_masked = np.zeros(thisarray.shape[0:2])
    for ii in range(thisarray.shape[0]): #channels where perce participants above threshold
        for iii in range(thisarray.shape[1]):
            perc = sum(bools[ii,iii])/len(bools[ii,iii])
            print(perc)
            if perc < perc_part:
                chn_mask[ii,iii] = False
                corr_masked[ii,iii] = 0
            else:
                chn_mask[ii,iii] = True
                corr_masked[ii,iii] = corr[ii,iii]
    corr2plot = corr_masked
    nlines=300

    #% plot corellation matrix
    # let's plot this matrix
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(corr2plot, cmap='viridis', clim=np.percentile(corr, [5, 95]))
    fig.tight_layout()
    plt.savefig(f'/imaging/ai05/images/AEC{freq_labels[i]}_group_mat.png')

    #% plot this on a brain
    threshold_prop = 0.3  # percentage of strongest edges to keep in the graph
    #degree = mne.connectivity.degree(corr, threshold_prop=threshold_prop) # get the degree
    degree = mne.connectivity.degree(corr2plot) # get the degree
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
    mne.viz.plot_connectivity_circle(corr2plot, label_names, n_lines=nlines,
                             node_angles=node_angles, node_colors=label_colors,
                             title=f'{freq_labels[i]}')[0].savefig(f'/imaging/ai05/images/AEC_{freq_labels[i]}_circle.png')

    plotting.plot_connectome(corr_masked, coords,
                             edge_threshold="95%",
                             title=f'{freq_labels[i]}').savefig(f'/imaging/ai05/images/AEC_{freq_labels[i]}net.png')
#%% vertex to MNI space
coords = []
for i in range(len(labels)):
    if 'lh' in label_names[i]:
        hem = 1
    else:
        hem = 0
    coord = mne.vertex_to_mni(labels[i].center_of_mass(subjects_dir=fsdir), subject='fsaverage_1',hemis=hem,subjects_dir=fsdir)
    coords.append(coord[0])
#%% nii learn plots
from nilearn import datasets
from nilearn.input_data import NiftiLabelsMasker
from nilearn import plotting
des = datasets.fetch_atlas_surf_destrieux()
# create masker to extract functional data within atlas parcels
masker = NiftiLabelsMasker(labels_img=labels, standardize=True,
                           memory='nilearn_cache')
coordinates = plotting.find_parcellation_cut_coords(labels_img='/imaging/ai05/RED/RED_MEG/resting/analysis/aparcaseg.nii.gz')




#%% Meeting notes from 6/2/2019
#Continous concetantion, downsample after hilbert envelope
#kanads method section of PhD -- threshold free level setting (singular value decomposition)
#

#%% source recon continous raw data
pythonpath = '/home/ai05/.conda/envs/mne_2/bin/python'
for i in range(len(RED_id)):
    pycom = f"""
import sys
import os
from os.path import join
import numpy as np
import mne
sys.path.insert(0, '/imaging/ai05/RED/RED_MEG/resting/analysis/RED_Rest')
from REDTools import study_info

MP_name = 'Series005_CBU_MPRAGE_32chn'
dicoms = '/imaging/rs04/RED/DICOMs'
STRUCTDIR = '/imaging/ai05/RED/RED_MEG/resting/STRUCTURALS'
MAINDIR = '/imaging/ai05/RED/RED_MEG/resting'
folders = [f for f in os.listdir(dicoms) if os.path.isdir(os.path.join(dicoms,f))]
ids = [f.split('_')[0] for f in folders]

#get ids etc
RED_id, MR_id, MEG_fname = study_info.get_info()

def raw2source(meg_f, inv_op, method, outdir):
    raw = mne.io.read_raw_fif(meg_f, preload=True)
    inv = mne.minimum_norm.read_inverse_operator(inv_op)
    start, stop = raw.time_as_index([60, raw.times[-30000]])
    stc = mne.minimum_norm.apply_inverse_raw(raw, inv,
                                             lambda2=1.0/1.0**2,
                                             method=method,
                                             start=start,
                                             stop=stop,
                                             buffer_size=int(len(raw.times)/10))
    np.save(join(outdir, f'{{os.path.basename(meg_f).split("_")[0]}}_{{method}}_stc_data.npy'),stc.data)
    np.save(join(outdir, f'{{os.path.basename(meg_f).split("_")[0]}}_{{method}}_stc_vertices.npy'),stc.vertices)
    np.save(join(outdir, f'{{os.path.basename(meg_f).split("_")[0]}}_{{method}}_stc_times.npy'),stc.times)

invdir = join(MAINDIR, 'inverse_ops')
rawdir = join(MAINDIR, 'preprocessed')
fsdir = join(MAINDIR, 'STRUCTURALS','FS_SUBDIR')
meg_f = join(rawdir, MEG_fname[{i}])
inv_op = join(invdir, f'{{RED_id[{i}]}}-inv.fif')
outdir = join(MAINDIR, 'raw_stcs')
method = 'MNE'
raw2source(meg_f, inv_op, method,outdir)
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
    os.system(f"sbatch --job-name=AmpEnvConnect_{i} --mincpus=2 --mem-per-cpu=40G -t 0-1:00 {join(MAINDIR, 'cluster_scripts', f'{i}_aec.csh')}")

#%% Read in a file
from scipy.signal import hilbert, resample
import scipy as ss
i = 10
outdir = join(MAINDIR, 'raw_stcs')
method = 'MNE'
data = np.load(join(outdir, f'{os.path.basename(MEG_fname[i]).split("_")[0]}_{method}_stc_data.npy'))
#%% Calculate the hilbert envelope for all the voxels and save downsampled version
downsample_factor = 100
#analytic_signal = np.zeros((data.shape[0], int(data.shape[1]/downsample_factor)))
analytic_signal = np.zeros(data.shape)
for i in range(data.shape[0]):
    tmp_sig = hilbert(data[i,:])
    #analytic_signal[i,:] = resample(tmp_sig, int(data.shape[1]/downsample_factor))
    analytic_signal[i,:] = tmp_sig
    print(i/data.shape[0])

#%% downsample
ii = 1
#plot some stuff
fig, ax = plt.subplots(1)

section = data[ii,0:4000]

section = mne.filter.filter_data(section,1000,freqs['Alpha'][0], freqs['Alpha'][1])
#get number of fft
n_fft = mne.filter.next_fast_len(len(section))
env = hilbert(section,N=n_fft, axis =-1)
d_env = resample(np.abs(env),int(len(env)/25))
d_sec = resample(section,int(len(section)/25))
#ax.plot(section[1000:5000])
#ax.plot(np.abs(env)[1000:5000])
#ax.plot(section)
ax.plot(d_sec)
ax.plot(d_env)
fig.savefig('/home/ai05/test.png')

#%% convert example into MNI space
i = 0

def mni_conv(i):
    inv_op = mne.minimum_norm.read_inverse_operator(join(invdir, f'{RED_id[i]}-inv.fif'))
    # get raw file
    raw = mne.io.read_raw_fif(join(rawdir, MEG_fname[i]), preload=False)
    events = mne.make_fixed_length_events(raw, duration=10.)
    epochs = mne.Epochs(raw, events=events, tmin=0, tmax=10.,
                        baseline=None, preload=True)
    del raw
    epochs = epochs[0:1]
    # invert the eopchs
    stcs = mne.minimum_norm.apply_inverse_epochs(epochs, inv_op,
                                                 lambda2=lambda2,
                                                 pick_ori=pick_ori,
                                                 method=method)
    stc = stcs[0]
    del stcs
    mni = mne.vertex_to_mni(stc.vertices, [0,1], MR_id[i], subjects_dir=fsdir) # mni space
    mni = np.vstack([mni[0], mni[1]]) # stack
    outdir = join(MAINDIR, 'mni_coords')
    np.save(join(outdir, f'{os.path.basename(MEG_fname[i]).split("_")[0]}_mni.npy'),mni)

joblib.Parallel(n_jobs=15)(
    joblib.delayed(mni_conv)(i) for i in range(len(MEG_fname)))