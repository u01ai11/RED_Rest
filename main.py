import sys
import os
import re
import matplotlib.pyplot as plt
import numpy as np
import collections
import joblib
import mne
import REDTools

#%% paths

#%% MaxFilter


def maxFilt(cluster=False,**kw):
    """
    A python wrapper for maxfilters command line functions
    :param cluster:
        Boolean value, if False it will run Maxfilter locally, if True it will submit it to the CBUs SLURM cluster
    :param kw:
        Dictionary containing keyword arguments, all are mandetory, but you can pass '' if you don't want to run that option
        maxf_cmd: The command that runs MaxFilter in your environemnt, this can be a command or address to the executable
        f: the input raw file
        o: the output raw file
        The other arguments are the standard options for MaxFilter
        trans, frame, regularize, st, cor (corr), orig, inval (in), outval(out), movecomp, the full bads_cmd, lg


    :return:
    """
    maxf_cmd = kw.get('max_cmd')
    f = kw.get('f')
    o = kw.get('o')
    trans = kw.get('trans')
    frame = kw.get('frame')
    regularize = kw.get('regularize')
    st = kw.get('st')
    cor = kw.get('cor')
    orig = kw.get('orig')
    inval = kw.get('inval')
    outval = kw.get('outval')
    movecomp = kw.get('movecomp')
    bads_cmd = kw.get('bads_cmd')
    lg = kw.get('lg')

    max_cmd = f"{maxf_cmd} -f {f} -o {o} -trans {trans} -frame {frame} -regularize {regularize}"\
              f" -st {st} -corr {cor} -origin {orig} -in {inval} -out {outval} -movecomp {movecomp}"\
              f" {bads_cmd}-autobad on -force -linefreq 50 -v | tee {lg}"

    if cluster:
        # submit to cluster
        #make bash file
        tcshf =f"""#!/bin/tcsh
        {max_cmd}
        """
        # save in log directory
        tcpath = lg.split('.')[0]+'.tcsh'
        print(tcshf, file=open(tcpath, 'w'))
        # execute this on the cluster
        os.system(f'sbatch --job-name={os.path.basename(tcpath)} --mincpus=5 -t 0-1:00 {tcpath} -constraint=maxfilter ')
    else: # run on current machine
        print(max_cmd)
        os.system(max_cmd)


#%% run maxfilter
resting_path = '/imaging/ai05/RED/RED_MEG/resting'
outpath = os.path.join(resting_path, 'MaxFiltered')

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

    maxFilt(cluster=True, **MF_settings)




def plot_MaxLog(logpath, outpath, plot=False):

    """
    This plots the different variables in MaxFilter log file for checking visually
    It also returns values for each of these people
    :param logpath:
        The path and filename of the .log file
    :param outpath:
        The directory you want the images saving in
    :return:
        It returns a summary of values from these files as well
        an array with tshape time x
            Fitting Error (cm)', 'Goodness of Fit', 'Translation (cm/s)', 'Rotation (Rads/s)', 'Drift (cm)
    """

    # read in file as list of strings
    with open(logpath, "r") as myfile:
        lines = myfile.readlines()

    #get just the lines starting with #t
    pos_only = [f for f in lines if f[0:2] == '#t']
    p = re.compile("\d+\.\d+")

    ts = [float(p.findall(f)[0]) for f in pos_only] # time (seconds)
    es = [float(p.findall(f)[1]) for f in pos_only]# fitting error (cm)
    gs = [float(p.findall(f)[2]) for f in pos_only] # goodness of fit
    vs = [float(p.findall(f)[3]) for f in pos_only] # translation (cm/s)
    rs = [float(p.findall(f)[4]) for f in pos_only] # rotation (rd/s)
    ds = [float(p.findall(f)[5]) for f in pos_only] # drift (cm)

    # labels
    labels = ['Fitting Error (cm)', 'Goodness of Fit', 'Translation (cm/s)', 'Rotation (Rads/s)', 'Drift (cm)']
    if plot:
        plt.close('all')
        objs = plt.plot(ts,es,ts,gs,ts,vs,ts,rs,ts,ds)
        plt.legend(iter(objs), labels)
        plt.title(os.path.basename(logpath))
        plt.savefig(os.path.join(outpath,os.path.basename(logpath).split('.')[0]+'.png'))

    # return np.array([[np.mean(es), np.mean(gs), np.mean(vs), np.mean(rs), np.mean(ds)],
    #         [np.std(es), np.std(gs), np.std(vs), np.std(rs), np.std(ds)]
    #         ])
    summary = np.array([es,gs,vs,rs,ds])
    return summary


#%% check head position using MaxFilter logs
logpath = '/imaging/ai05/RED/RED_MEG/resting/MaxFiltered/logs'
plot_path = '/imaging/ai05/RED/RED_MEG/resting/MaxFiltered/diagnostics'
log_files = [f for f in os.listdir(logpath) if '.log' in f] # get all log files

summary_data = []
for log in log_files:
    tmp = plot_MaxLog(logpath=os.path.join(logpath, log),
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
pythonpath = '/imaging/local/software/miniconda/envs/mne0.19/bin/python'
overwrite = False
preprocess_cluster(flist, indir, outdir, scriptpath, pythonpath ,overwrite)

#%%filenames for manual ica -- where components of blinks etc are not clear
man_ica = [f for f in os.listdir(outdir) if 'no' in f]

#%%