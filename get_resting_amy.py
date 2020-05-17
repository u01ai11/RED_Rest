#%% Import stuff
import pandas as pd
import sys
import os
from os import listdir as listdir
from os.path import join
from os.path import isdir
import numpy as np
sys.path.insert(0, '/imaging/ai05/RED/RED_MEG/resting/analysis/RED_Rest')
from REDTools import preprocess
from REDTools import study_info
from REDTools import sourcespace_setup
from REDTools import connectivity
import mne
import matplotlib.pyplot as plt
from nilearn import datasets
from nilearn.input_data import NiftiLabelsMasker
from nilearn import plotting
#%%
path_amyfile = '/imaging/ai05/phono_oddball/complete_amy.csv'
df = pd.read_csv(path_amyfile);
path_ace = '/megdata/cbu/brainnetworks'
path_out = '/imaging/ai05/RED/RED_MEG/ace_resting'
ace_ids = []
meg_ids = []
mri_ids = []
rest_files = []

for i,row in df.iterrows():
    if str(row['MEG ID']) == 'nan':
        continue
    meg_ids.append(row['MEG ID'])
    mri_ids.append(row['MRI ID'])
    #find file
    fold = listdir(join(path_ace, str(row['MEG ID'])))
    file = [f for f in listdir(join(path_ace, str(row['MEG ID']), fold[0])) if 'rest' in f.lower()]

    #move file(s) to new directory
    # for i, f in enumerate(file):
    #     if i == 0:
    #         num = ''
    #     else:
    #         num = '_b'
    #
    #
    #     os.system(f'cp {join(path_ace, row["MEG ID"], fold[0], f)} {join(path_out, row["MEG ID"].split("meg")[1] + num + ".fif")}')

#%% maxfilter
resting_path = join(path_out)
outpath = os.path.join(path_out, 'MaxFiltered_Amy')

for i in range(len(os.listdir(f'{resting_path}/raw'))):

    target_file = os.listdir(f'{resting_path}/raw')[i]

    MF_settings = dict(
        max_cmd='/imaging/local/software/neuromag/bin/util/maxfilter-2.2.12',
        f=os.path.join(resting_path, 'raw', target_file),
        o=os.path.join(outpath, target_file.split('.fif')[0] + '_sss_raw' + '.fif'),
        lg=os.path.join(outpath, target_file.split('.')[0] + '.log'),
        trans='default',
        frame='head',
        regularize='in',
        st='10',
        cor='0.98',
        orig='0 0 45',
        inval='8',
        outval='3',
        movecomp='inter',
        bads_cmd='',
    )

    preprocess.maxFilt(cluster=True, **MF_settings)

#%% check head position using MaxFilter logs
logpath = join(outpath, 'logs')
plot_path = join(outpath, 'diag')
log_files = [f for f in os.listdir(logpath) if '.log' in f] # get all log files

summary_data = []
for log in log_files:
    tmp = preprocess.plot_MaxLog(logpath=join(logpath, log),
                                 outpath=plot_path, plot=True)
    summary_data.append(tmp)

#%% Flag any that are bad
IDs = [f.split('_')[0] for f in log_files] # get unique IDs from log files
# have a look
sums = [np.mean(x, axis=1) for x in summary_data]
print(sums)
# flag bads using thresholds - well use <.9 for fit and average over 2 for movement parameters
bads = []
for i in range(len(log_files)):
    bad = False
    means = np.mean(summary_data[i], axis =1) # this gives us means of all the measures from above
    if np.isnan(means[1]):
        bad = True
    if means[1] < 0.9:
        bad = True
    if any([x>2 for x in means[2:5]]):
        bad = True
    bads.append(bad)

# have a look at this
print(bads)

#%% go through to rename getting rid of the '_'
flist = [f for f in os.listdir(outpath) if 'fif' in f]
for f in flist:
    newname = f.replace('_', '-',1)
    os.system(f'mv {join(outpath, f)} {join(outpath, newname)}')

#%% preprocess initially
flist = [f for f in os.listdir(outpath) if 'fif' in f]
indir = outpath
outdir = os.path.join(path_out, 'preprocessed')
scriptpath = os.path.join(resting_path, 'cluster_scripts')
pythonpath = '/imaging/local/software/miniconda/envs/mne0.19/bin/python'
overwrite = False
preprocess.preprocess_cluster(flist, indir, outdir, scriptpath, pythonpath ,overwrite)

#%%filenames for manual ica -- where components of blinks etc are not clear
man_ica = [f for f in os.listdir(outdir) if 'no' in f]
i = 0
#%%
raw.save(f'{outdir}/{f.split("_")[0]}_{f.split("_")[1]}_clean_raw.fif', overwrite=True)
i +=1
print(f'{i+1} out of {len(man_ica)}')
f = man_ica[i]
raw = mne.io.read_raw_fif(f'{outdir}/{f}', preload=True)
ica = mne.preprocessing.ICA(n_components=25, method='fastica').fit(raw)
comps = ica.plot_components()
comps[0].savefig('/home/ai05/comp1.png')
comps[1].savefig('/home/ai05/comp2.png')
raw.plot(start=240).savefig('/home/ai05/raw1.png')
print(man_ica[i])
#%% change inds and decide
ica.exclude =[0,1,10,20]
ica.apply(raw)
# if you need to plot the channels
raw.plot(start=240).savefig('/home/ai05/raw2.png')
#%% rename messed up filenames
renamed = [f for f in os.listdir(outdir) if 'fif_' in f]
for f in renamed:
    os.system(f'mv {join(outdir,f)} {join(outdir,f.split("_")[0]+"_clean_raw.fif")}')

renamed = [f for f in os.listdir(outdir) if 'sss_' in f]
for f in renamed:
    os.system(f'mv {join(outdir,f)} {join(outdir,f.split("_")[0]+"-b_clean_raw.fif")}')

#%% Get freesurfer directory names and trans-file names

# The IDs are from another study and stored in the 'MNE id' column of the dataframe
fs_sub = []

for i, name in enumerate(meg_ids):
    try:
        mne_no = int(df.loc[df['MEG ID'] == name]['MNE_id'].values[0])
    except:
        fs_sub.append('fsaverage_1')
    if isdir(join(resting_path, 'FS_SUBDIR', "{:04d}".format(mne_no))):
        fs_sub.append("{:04d}".format(mne_no))
    else:
        fs_sub.append('fsaverage_1')

#%% setup sources
from REDTools import sourcespace_setup
mne_src_files = sourcespace_setup.setup_src_multiple(sublist=fs_sub,
                                                     fs_sub_dir=join(resting_path, 'FS_SUBDIR'),
                                                     outdir=join(resting_path, 'src_space'),
                                                     spacing='oct6',
                                                     surface='white',
                                                     src_mode='cortical',
                                                     n_jobs1=15,
                                                     n_jobs2=1)
#%% rad in bem models
mne_bem_files = sourcespace_setup.make_bem_multiple(sublist=fs_sub,
                                                    fs_sub_dir=join(resting_path, 'FS_SUBDIR'),
                                                    outdir=join(resting_path, 'bem'),
                                                    single_layers=True,
                                                    n_jobs1=20)

#%% get list of raws, bems, src, trans
# ordered IDs
MEG_id, MR_id, MEG_fname = study_info.get_info_ACE()
# trans bem src
transdir = join(path_out, 'coreg')
trans = []
bemdir = join(path_out, 'bem')
bems = []
srcdir = join(path_out, 'src_space')
srcs = []
megfs = []
# loop through ids and add in order
for fid, mrid, megf in zip(MEG_id, MR_id, MEG_fname):
    base_no = fid.split('-')[1]
    # Trans file
    tmp = [f for f in listdir(transdir) if base_no in f]
    if len(tmp)> 0:
        trans.append(join(transdir,tmp[0]))
    else:
        print(f'{base_no} does not have trans file' )
        continue

    # Bemfile
    tmp = [f for f in listdir(bemdir) if mrid in f]
    tmp = [f for f in tmp if 'sol' in f]
    if len(tmp)> 0:
        bems.append(join(bemdir,tmp[0]))
    else:
        print(f'{base_no} does not have bem file' )
        del trans[-1]
        continue
    #src space file
    tmp = [f for f in listdir(srcdir) if mrid in f]
    tmp = [f for f in tmp if 'cortical' in f]
    if len(tmp)> 0:
        srcs.append(join(srcdir,tmp[0]))
    else:
        print(f'{base_no} does not have src file' )
        del trans[-1]
        del bems[-1]
        continue
    megfs.append(join(path_out, 'preprocessed', megf))

#%% Now thats done generate the stuff

invfiles = sourcespace_setup.make_inv_multiple(rawfs=megfs,
                                    transfs=trans,
                                    bemfs=bems,
                                    srcfs=srcs,
                                    outdir=join(path_out, 'invs'),
                                    njobs=15)

#%% check that this are sensible by plotting a single brain
i = 0
raw = mne.io.read_raw_fif(megfs[i])
events = mne.make_fixed_length_events(raw, duration=10)
epochs = mne.Epochs(raw, events=events, tmin=0, tmax=10,
                    baseline=None, preload=True)
inv = mne.minimum_norm.read_inverse_operator(invfiles[i])
stcs = mne.minimum_norm.apply_inverse_epochs(epochs[20:21], inv, lambda2=1. / 9.,
                                             return_generator=False)
stc = stcs[0]
hemi = 'lh'
vertno_max, time_max = stc.get_peak(hemi=hemi)
surfer_kwargs = dict(
    hemi=hemi, subjects_dir=join(resting_path, 'FS_SUBDIR'), views='lat',
    initial_time=time_max, time_unit='s', size=(800, 800),
    smoothing_steps=5, backend='matplotlib')
brain = stc.plot(**surfer_kwargs)
brain.savefig(f'/home/ai05/test.png')

#%% now proceed with the full sample recon and AEC
MAINDIR = '/imaging/ai05/RED/RED_MEG/ace_resting'
connectivity.ACE_cluster_envelope_corellation(MR_id, MAINDIR)

#%% and some plotting
#%% Calculate average corellation matrix for each brain
aec_dir = join(MAINDIR, 'envelope_cors')
thetas = [f for f in listdir(aec_dir) if 'Theta' in f] # find thetas
# filter MEG id to match
nMEG_id = [i for i in MEG_id if i in [ii.split('_')[0] for ii in thetas]]
thetas = [thetas[[i for i, s in enumerate(thetas) if i_d in s][0]] for i_d in nMEG_id] # reorder to match RED
#repeat for all bands
alphas =[f for f in listdir(aec_dir) if 'Alpha' in f]
alphas = [alphas[[i for i, s in enumerate(alphas) if i_d in s][0]] for i_d in nMEG_id]
betalow = [f for f in listdir(aec_dir) if 'Lower Beta' in f]
betalow = [betalow[[i for i, s in enumerate(betalow) if i_d in s][0]] for i_d in nMEG_id]
betahigh =[f for f in listdir(aec_dir) if 'Upper Beta' in f]
betahigh = [betahigh[[i for i, s in enumerate(betahigh) if i_d in s][0]] for i_d in nMEG_id]

# Combine each band into an average connectivity matrix
arraylist = []
for band in [thetas, alphas, betalow, betahigh]:
    #intialise empty array
    mat_3d = np.zeros(shape=(68, 68, len(band)))
    for i, f in enumerate(band):
        mat_3d[:,:,i] = np.load(join(aec_dir, f))
    arraylist.append(mat_3d)

#%% vertex to MNI space
labels = mne.read_labels_from_annot('fsaverage_1', parc='aparc',
                                        subjects_dir=join(MAINDIR,'FS_SUBDIR'))
labels = labels[0:-1]
label_names = [label.name for label in labels]
coords = []
for i in range(len(labels)):
    if 'lh' in label_names[i]:
        hem = 1
    else:
        hem = 0
    coord = mne.vertex_to_mni(labels[i].center_of_mass(subjects_dir=join(MAINDIR,'FS_SUBDIR')), subject='fsaverage_1',hemis=hem,subjects_dir=join(MAINDIR,'FS_SUBDIR'))
    coords.append(coord[0])

#%% make a basic average array for one frequency
freq_labels = ['thetas', 'alphas', 'betalow', 'betahigh']
threshold = 0.05 # threshold for strength
perc_part = 0.7 # Threshold for how many participants
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
    plt.savefig(f'/imaging/ai05/images/ace_AEC{freq_labels[i]}_group_mat.png')

    #% plot this on a brain
    threshold_prop = 0.3  # percentage of strongest edges to keep in the graph
    #degree = mne.connectivity.degree(corr, threshold_prop=threshold_prop) # get the degree
    degree = mne.connectivity.degree(corr2plot) # get the degree
    #load in FSAVERAGE for parcellations and inverse for plotting
    labels = mne.read_labels_from_annot('fsaverage_1', parc='aparc',
                                        subjects_dir=join(MAINDIR,'FS_SUBDIR'))
    labels = labels[0:-1]
    # we know that 99144's inversion contains source from FSAVERAGE so load that
    inv_op = mne.minimum_norm.read_inverse_operator(join(MAINDIR,'invs', '15-0198_inv.fif'))
    stc = mne.labels_to_stc(labels, degree)
    stc = stc.in_label(mne.Label(inv_op['src'][0]['vertno'], hemi='lh') +
                       mne.Label(inv_op['src'][1]['vertno'], hemi='rh'))
    stc.plot(
        clim=dict(kind='percent', lims=[75, 85, 95]), colormap='gnuplot',
        subjects_dir=join(MAINDIR, 'FS_SUBDIR'), hemi='lh',
        smoothing_steps=1, time_label=freq_labels[i], backend='matplotlib').savefig(f'/imaging/ai05/images/ace_AEC_{freq_labels[i]}_src_group_lh.png')
    stc.plot(
        clim=dict(kind='percent', lims=[75, 85, 95]), colormap='gnuplot',
        subjects_dir=join(MAINDIR, 'FS_SUBDIR'), hemi='rh',
        smoothing_steps=1, time_label=freq_labels[i], backend='matplotlib').savefig(f'/imaging/ai05/images/ace_AEC_{freq_labels[i]}_src_group_rh.png')

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
                                     title=f'{freq_labels[i]}')[0].savefig(f'/imaging/ai05/images/ace_AEC_{freq_labels[i]}_circle.png')

    plotting.plot_connectome(corr2plot, coords,
                             edge_threshold="95%",
                             title=f'{freq_labels[i]}').savefig(f'/imaging/ai05/images/ace_AEC_{freq_labels[i]}_net.png')

#%% Now let's see if we can build a GLM type analysis
sys.path.insert(0, '/home/ai05/Downloads/glm')
import glmtools
