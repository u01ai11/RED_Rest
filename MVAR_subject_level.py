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
from REDTools import plotting as red_plotting
import scipy.stats as ss
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

#%% make welch's periodograms for the parcel timecourses
ch_names = red_plotting.get_labels()[1]
#%%
fmin = 0
fmax = 75
average = False
sample_rate = 150 # sample rate
delays = 25
freq_vect = np.linspace(0, sample_rate/2, 36) #frequency vector representing the model metrics

i = 0

for i in range(len(parcel_files)):
    # load array
    X = np.load(join(parcel_dir, parcel_files[i]))
    # labels

    X = mne.filter.notch_filter(X, Fs=150, freqs=np.arange(50, 75, 50))

    # transpose
    X = X.transpose([1,2,0])

    try:
        X[:,:,0] = sails.orthogonalise.symmetric_orthonormal(X[:,:,0], maintain_mag=False)[0]
    except:
        print('full_rank_error')




    info = mne.create_info(ch_names=ch_names, ch_types='mag', sfreq=150)
    raw = mne.io.RawArray(X[:,:,0], info)



    delay_vect = np.arange(delays)
    #m = sails.OLSLinearModel.fit_model(X, delay_vect)
    m = sails.VieiraMorfLinearModel.fit_model(X, delay_vect)
    Fo = sails.mvar_metrics.FourierMvarMetrics.initialise(m, sample_rate, freq_vect)
    #ar_spec = sails.mvar_metrics.ar_spectrum(m.parameters, m.resid_cov, sample_rate, freq_vect)
    ar_spec = Fo.PSD
    # square each power value then divide by frequency
    plt_spec = ar_spec[:,:,:,0].reshape(ar_spec.shape[0]*ar_spec.shape[1],ar_spec.shape[2]).transpose([1,0])
    plt_spec = np.divide(np.square(plt_spec), np.array([freq_vect]*ar_spec.shape[0]*ar_spec.shape[1]).transpose())

    fig, axs = plt.subplots(3)
    raw.plot_psd(fmin=fmin, fmax=fmax, average=average, ax=axs[0])
    axs[1].set_yscale('log', nonposy='clip')
    axs[2].set_yscale('log', nonposy='clip')

    axs[1].plot(plt_spec)

    mean_spec = np.nanmean(plt_spec, axis=1)
    std_spec = np.nanstd(plt_spec, axis=1)
    axs[2].plot(mean_spec,'k-')
    axs[2].fill_between(list(range(plt_spec.shape[0])),plt_spec.min(axis=1), plt_spec.max(axis=1))

    plt.savefig(join(figdir, 'PSD_parcels',f'psd_{i}_VM.png'))
    plt.close('all')

#%% coherence analysis on one participant

# define some information about our data
sample_rate = 150 # sample rate
freq_vect = np.linspace(0, sample_rate/2, 36) #frequency vector representing the model metrics

# define participant ID
i = 3
# read file, data is parcels x timepoints
X = np.load(join(parcel_dir, parcel_files[i]))

# we also probably want to filter our data slightly (use FIR)
X = mne.filter.notch_filter(X, Fs=150, freqs=np.arange(50, 75, 50))

# transpose
X = X.transpose([1,2,0])
X[:,:,0] = sails.orthogonalise.symmetric_orthonormal(X[:,:,0], maintain_mag=False)[0]

#reshape as sails expects (nsignals, nsamples, ntrials)
#X = X.transpose([1,2,0])


#%%
#% working out model order number

AICs = []
RSQs = []
orders = []

data = X[0:10,:,:]
for delays in range(2, 35):
    print(delays)
    delay_vect = np.arange(delays)
    m = sails.OLSLinearModel.fit_model(data, delay_vect)
    diag = m.compute_diagnostics(data)
    orders.append(m.order)
    AICs.append(diag.AIC)
    RSQs.append(diag.R_square)
#%%
f = plt.figure()
plt.plot(orders, AICs, 'o');
plt.xlabel('Model Order')
plt.ylabel('AIC')
plt.savefig(join(figdir, f'model_order_AIC_{i}.png'))
plt.close(f)

f = plt.figure()
plt.plot(orders, RSQs, 'o');
plt.xlabel('Model Order')
plt.ylabel('R Squared')
plt.savefig(join(figdir, f'model_order_R{i}.png'))
plt.close(f)



#%% investigate the spectral features of the model against the real data for one parcel
# load in real data
sample_rate = 150 # sample rate
freq_vect = np.linspace(0, sample_rate/2, 36) #frequency vector representing the model metrics

parcel_ind = 3
delays = 31


#%% just do the welch's to find good number for nperseg


nperseg = 70

X = np.load(join(parcel_dir, parcel_files[0]))
X = mne.filter.notch_filter(X, Fs=150, freqs=np.arange(50, 75, 50))
X = X.transpose([1,2,0])
welch = scipy.signal.welch(X[parcel_ind,:,0],fs=150,nperseg=nperseg)

welch_array = np.zeros([len(parcel_files), len(welch[1])])
welch_array = np.zeros([len(parcel_files), len(welch[1])])

for i in range(10):
#for i in range(len(parcel_files))[-50:-1]:
    X = np.load(join(parcel_dir, parcel_files[i]))
    X = mne.filter.notch_filter(X, Fs=150, freqs=np.arange(50, 75, 50))
    X[0,:,:] = sails.orthogonalise.symmetric_orthonormal(X[0,:,:], maintain_mag=False)[0]
    X = X.transpose([1,2,0])
    welch_array[i,:] = scipy.signal.welch(X[parcel_ind,:,0],fs=150,nperseg=nperseg)[1]

#%%
f, ax = plt.subplots()

#ax.semilogy(welch[0], welch_array.transpose())
#plt.xlim([0, 40])
#ax.set_xscale('log')
ax.set_yscale('log')
ax.plot(welch[0],welch_array.transpose())
plt.savefig(join(figdir, f'welch_PSD_{parcel_ind}'))
plt.close('all')
#%%

no_parts = len(parcel_files)
sample_rate = 150 # sample rate
freq_vect = np.linspace(0, sample_rate/2, 36) #frequency vector representing the model metrics

parcels = list(range(0,10))
parcel_ind = 5
delays = 25


nperseg = 70

#mod = 'vm'
mod = 'ols'

compare_spectrum = np.zeros([2,36, no_parts])
for i in range(no_parts):
    print(i)
    # load data
    X = np.load(join(parcel_dir, parcel_files[i]))
    # filter


    # transpose
    X = X.transpose([1,2,0])
    X = X[parcels,:,:]
    X[:,:,0] = sails.orthogonalise.symmetric_orthonormal(X[:,:,0], maintain_mag=False)[0]
    # model
    delay_vect = np.arange(delays)
    if mod == 'ols':
        m = sails.OLSLinearModel.fit_model(X, delay_vect)
    else:
        m = sails.VieiraMorfLinearModel.fit_model(X, delay_vect)
    Fo = sails.mvar_metrics.FourierMvarMetrics.initialise(m, sample_rate, freq_vect)
    #ar_spec = sails.mvar_metrics.ar_spectrum(m.parameters, m.get_residuals(X), 150, freq_vect)

    # real decomposition
    welch = scipy.signal.welch(X[parcel_ind,:,0],fs=150,nperseg=nperseg)

    compare_spectrum[0,:,i] = welch[1]
    #compare_spectrum[1,:,i] = ar_spec[parcel_ind, parcel_ind,:,0]
    compare_spectrum[1,:,i]= Fo.PSD[parcel_ind, parcel_ind,:,0]


#%% plot the spectra

f, ax = plt.subplots(1,2)
#
ax[0].set_yscale('log')
ax[1].set_yscale('log')
ax[0].plot(welch[0], compare_spectrum[0])
ax[1].plot(welch[0], compare_spectrum[1])

plt.savefig(join(figdir, f'compare_spectra_parcel_{parcel_ind}_{mod}_delays_{delays}.png'))
plt.close(f)

#%% We need to exclude some parcels from the data, due to rank issues (Maxfilter leads to scan data being rank 64,
#%% ICA denoising reduces rank further, then parcels end up being linear combinations of each other

# get rid of limbic/cingulate brain regions
removelist = ['isthmuscingulate', 'posteriorcingulate', 'caudalanteriorcingulate', 'rostralanteriorcingulate', 'parahippocampal',
              'entorhinal', 'fusiform']
labels, label_names, re_order_ind = red_plotting.get_labels()
removelabels = [f for f in label_names if len([i for i in removelist if i in f]) > 0]
removeinds = [label_names.index(i) for i in removelabels]

#%% looks like 20 is probably an acceptable number of modes
# loop through this and do for all participants in parallel
joblib.Parallel(n_jobs =20)(
    joblib.delayed(MVAR_single)(i,'OLS',25, 'notch',
                                        join(root_dir,'resting', 'MVAR'),
                                        parcel_dir, parcel_files, 150,
                                        'partial_directed_coherence',
                                        [],
                                        True) for i in range(len(parcel_files)))

#%% loop
for i in range(len(parcel_files)):
    print(i)
    MVAR_single(i,'OLS',25, 'notch',
                         join(root_dir,'resting', 'MVAR'),
                         parcel_dir, parcel_files, 150,
                         'partial_directed_coherence')

type= 'OLS'; modes=25; filter='notch'; outdir=join(root_dir,'resting', 'MVAR'); sample_rate = 150;metric ='partial_directed_coherence'


#%% check if passed

made = []
low_rank_ids = []
for i in range(len(parcel_files)):
    id_ = parcel_files[i].split('_')[0]
    made.append(isfile(join(outdir, f'mvar_OLS_{id_}.npy')))
    if made[i] == False:
        low_rank_ids.append(id_)
#%%
np.save(join(root_dir, 'resting', 'low_rank_IDs.npy'), low_rank_ids)