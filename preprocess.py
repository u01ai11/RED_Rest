import sys
sys.path.insert(0, '/imaging/ai05/RED/RED_MEG/resting/analysis/RED_Rest')
import os
import re
import matplotlib.pyplot as plt
import numpy as np
import collections
import joblib
import mne
from REDTools import preprocess

#%% paths

#%% run maxfilter
resting_path = '/imaging/ai05/RED/RED_MEG/resting'
outpath = os.path.join(resting_path, 'MaxFiltered')
#%%
for i in range(len(os.listdir(f'{resting_path}/raw'))):

    target_file = os.listdir(f'{resting_path}/raw')[i]

    MF_settings = dict(
        max_cmd='/imaging/local/software/neuromag/bin/util/maxfilter-2.2.12',
        f=os.path.join(resting_path, 'raw', target_file),
        o=os.path.join(outpath, target_file.split('raw')[0] + 'sss_raw' + target_file.split('raw')[1]),
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
logpath = '/imaging/ai05/RED/RED_MEG/resting/MaxFiltered/logs'
plot_path = '/imaging/ai05/RED/RED_MEG/resting/MaxFiltered/diagnostics'
log_files = [f for f in os.listdir(logpath) if '.log' in f] # get all log files

summary_data = []
for log in log_files:
    tmp = preprocess.plot_MaxLog(logpath=os.path.join(logpath, log),
                outpath=plot_path, plot=False)
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

#%% preprocess initially
flist = [f for f in os.listdir(outpath) if 'fif' in f]
indir = outpath
outdir = os.path.join(resting_path, 'preprocessed')
scriptpath = os.path.join(resting_path, 'cluster_scripts')
pythonpath = '/home/ai05/.conda/envs/mne_2/bin/python'
overwrite = True
preprocess.preprocess_cluster(flist, indir, outdir, scriptpath, pythonpath ,overwrite)

#%% preprocess Amy
indir = os.path.join('/imaging/ai05/RED/RED_MEG/', 'ace_resting', 'MaxFiltered_Amy')
outdir = os.path.join('/imaging/ai05/RED/RED_MEG/', 'ace_resting', 'preprocessed')
flist = [f for f in os.listdir(indir) if 'fif' in f]
preprocess.preprocess_cluster(flist, indir, outdir, scriptpath, pythonpath ,overwrite)

#%% rename files for amy (oops)
flist = [f for f in os.listdir(outdir) if 'fif' in f]
for f in flist:
    newname = f'{f.split("_")[0]}_clean_raw.fif'
    os.system(f'mv {os.path.join(outdir, f)} {os.path.join(outdir, newname)}')
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
raw.plot(start=120).savefig('/home/ai05/raw1.png')
print(man_ica[i])
#%% change inds and decide
ica.exclude =[0,4]
ica.apply(raw)
# if you need to plot the channels
raw.plot(start=120).savefig('/home/ai05/raw2.png')
#%%

#%% preprocess to remove the 50hz line noise that is still present
