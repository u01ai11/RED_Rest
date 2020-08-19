import sys
from os import listdir as listdir
from os.path import join
from os.path import isfile
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, '/imaging/ai05/RED/RED_MEG/resting/analysis/RED_Rest')
import mne
import sails
import glmtools
import joblib
import scipy
import random
import copy
import pandas as pd
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
X = X.transpose([1,2,0])


#%%
#% working out model order number

AICs = []
RSQs = []
orders = []

data = X
for delays in range(2, 35):
    print(delays)
    delay_vect = np.arange(delays)
    m = sails.OLSLinearModel.fit_model(data, delay_vect)
    diag = m.compute_diagnostics(data)
    orders.append(m.order)
    AICs.append(diag.AIC)
    RSQs.append(diag.R_square)

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
delays = 20


#%% just do the welch's to find good number for nperseg


nperseg = 70

X = np.load(join(parcel_dir, parcel_files[0]))
X = mne.filter.notch_filter(X, Fs=150, freqs=np.arange(50, 75, 50))
X = X.transpose([1,2,0])
welch = scipy.signal.welch(X[parcel_ind,:,0],fs=150,nperseg=nperseg)

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

no_parts = 5
sample_rate = 150 # sample rate
freq_vect = np.linspace(0, sample_rate/2, 36) #frequency vector representing the model metrics

parcel_ind = 1
delays = 25


nperseg = 70

mod = 'vm'
#mod = 'ols'

compare_spectrum = np.zeros([2,36, no_parts])
for i in range(no_parts):
    # load data
    X = np.load(join(parcel_dir, parcel_files[i]))
    # filter
    #X = mne.filter.filter_data(X, sfreq=150, l_freq=0, h_freq=50)
    X = mne.filter.notch_filter(X, Fs=150, freqs=np.arange(50, 75, 50))

    # transpose
    X = X.transpose([1,2,0])
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
#%% looks like 20 is probably an acceptable number of modes

# put the calculation in a function
def MVAR_single(ind, type, modes, filter, outdir, parcel_dir, parcel_files):

    #id
    id_ = parcel_files[ind].split('_')[0]
    if isfile(join(outdir, f'mvar_{type}_{id}.npy')):
        print('file exists, skipping')
        return
    X = np.load(join(parcel_dir, parcel_files[ind]))

    if filter == 'notch':
        X = mne.filter.notch_filter(X, Fs=150, freqs=np.arange(50, 75, 50))
    elif type(filter) == tuple:
        # we also probably want to filter our data slightly (use FIR)
        X = mne.filter.filter_data(X, sfreq=150, l_freq=filter[0], h_freq=filter[1])
    else:
        print(f'{filter} is an unrecognised filter')
        return

    if len(X.shape) == 1:
        print('not correct data input, skipping')
        return
    #reshape as sails expects (nsignals, nsamples, ntrials)
    X = X.transpose([1,2,0])
    # create delay vector from modes
    delay_vect = np.arange(modes)

    # apply model
    if type == 'OLS':
        m = sails.OLSLinearModel.fit_model(X, delay_vect)
    elif type == 'VieiraMorf':
        m = sails.VieiraMorfLinearModel.fit_model(X, delay_vect)
    # get fourier decomp of model coefficients
    Fo = sails.mvar_metrics.FourierMvarMetrics.initialise(m, sample_rate, freq_vect)
    # save to file
    # get name
    np.save(join(outdir, f'mvar_{type}_{id}.npy'),Fo.directed_transfer_function)
    return Fo, m


#Fo, m = MVAR_single(5, 'OLS', 20, (1,30), join(root_dir,'resting', 'MVAR'))

# loop through this and do for all participants in parallel
joblib.Parallel(n_jobs =20)(
    joblib.delayed(MVAR_single)(i,'OLS',25, 'notch', join(root_dir,'resting', 'MVAR'),
                                parcel_dir, parcel_files) for i in range(len(parcel_files)))

#%% now we are going to run a GLM to see how consistent connectivity is accross participants
# The data will be in the form 136x12x12x36 (participant by parcels sender by parcel reciever by frequency power)
# accross participants we would like to know which of these connections and at what frequencies are consistent


#first we need to get age info ready

age = []
IQ = []

ids = [i.split('_')[0] for i in parcel_files]

for id_ in ids:
    #if id is not in meta, nan it
    if id_ not in meta.Alex_ID.to_list():
        age.append(np.nan)
        IQ.append(np.nan)
    else:
        age.append(float(meta[meta.Alex_ID == id_].Age))
        IQ.append(float(meta[meta.Alex_ID == id_].WASI_Mat))

age = np.array(age)
IQ = np.array(IQ)

age[np.isnan(age)] = np.nanmean(age)
IQ[np.isnan(IQ)] = np.nanmean(IQ)
#%%

# then we need to load all of the participants modelled data
glm_data = np.empty((len(parcel_files), 12,12,36))
for i in range(len(parcel_files)):
    id_ = parcel_files[i].split('_')[0]
    glm_data[i,:,:,:] = np.load(join(root_dir,'resting', 'MVAR', f'mvar_OLS_{id_}.npy'))[:,:,:,0]



#%% feed into GLM

dat = glmtools.data.TrialGLMData(data=glm_data, dim_labels=['participants', 'parcel_drivers', 'parcel_recievers', 'frequency'])

regs = list()
contrasts = list()
regs.append(glmtools.regressors.ConstantRegressor(num_observations=dat.info['num_observations']))
contrasts.append(glmtools.design.Contrast(name='Intercept',values=[1,0,0])) # for regressor

# add regressor for Age
regs.append(glmtools.regressors.ParametricRegressor(values=age,
                                                    name='Age',
                                                    preproc='z',
                                                    num_observations=dat.info['num_observations']))
contrasts.append(glmtools.design.Contrast(name='Age',values=[0,1,0]))


# add regressor for IQ
regs.append(glmtools.regressors.ParametricRegressor(values=age,
                                                    name='IQ',
                                                    preproc='z',
                                                    num_observations=dat.info['num_observations']))
contrasts.append(glmtools.design.Contrast(name='Age',values=[0,0,1]))



des = glmtools.design.GLMDesign.initialise(regs,contrasts)
model = glmtools.fit.OLSModel( des, dat )





#%% permute Intercept

P_i = glmtools.permutations.Permutation(des, dat, 1, 1000, metric='tstats' )
thresh_i = P_i.get_thresh(95) #  Thresh is a 12x12 matrix
sig_simple_i = model.tstats[0,...] >= thresh_i

#%% Plot age results
fig = sails.plotting.plot_vector(sig_simple_i([1,2,3,0]), freq_vect, diag=False)

ax = plt.gca()
#ax.set_ylim((0,20))
plt.savefig(join(figdir, f'glm_tstats_intercept.png'), bbox_inches='tight')
plt.close(fig)
#%% permute Age

P_age = glmtools.permutations.Permutation(des, dat, 1, 1000, metric='tstats' )
thresh_age = P_age.get_thresh(95) #  Thresh is a 12x12 matrix
sig_simple_age = model.tstats[1,...] >= thresh_age

#%% Plot age results
fig = sails.plotting.plot_vector(sig_simple_age([1,2,3,0]), freq_vect, diag=False)

ax = plt.gca()
#ax.set_ylim((0,20))
plt.savefig(join(figdir, f'glm_tstats_Age.png'), bbox_inches='tight')
plt.close(fig)

#%% permute IQ

P_IQ = glmtools.permutations.Permutation(des, dat, 2, 1000, metric='tstats' )
thresh_IQ = P_IQ.get_thresh(95) #  Thresh is a 12x12 matrix
sig_simple_IQ = model.tstats[2,...] >= thresh_IQ

#%% Plot IQ results
fig = sails.plotting.plot_vector(sig_simple_age([1,2,3,0]), freq_vect, diag=False)



ax = plt.gca()
#ax.set_ylim((0,20))
plt.savefig(join(figdir, f'glm_tstats_IQ.png'), bbox_inches='tight')
plt.close(fig)


#%% plot the t stats

fig = sails.plotting.plot_vector(model.tstats.transpose([1,2,3,0]), freq_vect, diag=False)

ax = plt.gca()
#ax.set_ylim((0,20))
plt.savefig(join(figdir, f'glm_tstats_group.png'), bbox_inches='tight')
plt.close(fig)
#%%
fig = sails.plotting.plot_vector(model.betas.transpose([1,2,3,0]), freq_vect, diag=False)

ax = plt.gca()

plt.savefig(join(figdir, f'glm_betas_group.png'), bbox_inches='tight')
plt.close(fig)

#%%

perms = 100

null = np.empty(model.tstats.shape + (perms,))
for p in range(perms):
    p_des = copy.deepcopy(des)
    for i in range(len(des.design_matrix)):
        if random.randint(0,1) == 0:
            p_des.design_matrix[i] = 1
        else:
            p_des.design_matrix[i] = -1
    p_model = glmtools.fit.OLSModel(des, dat)
    null[:,:,:,:,p] = p_model.get_tstats()

#%%

P = glmtools.permutations.Permutation(des, dat, 1, 1000, metric='tstats' )
thresh = P.get_thresh(95) #  Thresh is a 12x12 matrix
sig_simple = model.tstats[1,...] >= thresh




#%%

Fo, m = MVAR_single(5, 'OLS', 20, (1,30), join(root_dir,'resting', 'MVAR'))

# plot the directed transfer function of this decomposition
fig = sails.plotting.plot_vector(Fo.directed_transfer_function, freq_vect, diag=True)

plt.savefig(join(figdir, f'coherence_parcelations_20_modes_{i}.png'), bbox_inches='tight')
plt.close(fig)

#%% Now lets try a modal decomp
modes = sails.MvarModalDecomposition.initialise(m, sample_rate)
# Now extract the metrics for the modal model
Mo = sails.ModalMvarMetrics.initialise(m, sample_rate,
                                      freq_vect, sum_modes=False)

# In actual analysis we need to permute to decide a pole threshold
pole_threshold = 10

# Extract the pairs of mode indices
pole_idx = modes.mode_indices

# Find which modes pass threshold
surviving_modes = []

for idx, poles in enumerate(pole_idx):
    # The DTs of a pair of poles will be identical
    # so we can just check the first one
    if modes.dampening_time[poles[0]] > pole_threshold:
        surviving_modes.append(idx)

# Use root plot to plot all modes and then replot
# the surviving modes on top of them

f, ax = plt.subplots()

ax = sails.plotting.root_plot(modes.evals, ax=ax)

low_mode = None
high_mode = None

for mode in surviving_modes:
    # Pick the colour based on the peak frequency
    if modes.peak_frequency[pole_idx[mode][0]] < 0.001:
        color = 'b'
        low_mode = mode
    else:
        color = 'r'
        high_mode = mode

    for poleidx in pole_idx[mode]:
        ax.plot(modes.evals[poleidx].real, modes.evals[poleidx].imag,
                marker='+', color=color)

plt.savefig(join(figdir, f'modal_decomp_modes_{i}.png'), bbox_inches='tight')
plt.close(f)
#%% Look at highest and lowest modes
num_sources = 12
M = Mo
low_freq_idx = np.argmin(np.abs(freq_vect - modes.peak_frequency[pole_idx[low_mode][0]]))
high_freq_idx = np.argmin(np.abs(freq_vect - modes.peak_frequency[pole_idx[high_mode][0]]))

# We can now plot the two graph
plt.figure()

low_psd = np.sum(M.PSD[:, :, :, pole_idx[low_mode]], axis=3)
high_psd = np.sum(M.PSD[:, :, :, pole_idx[high_mode]], axis=3)

# Plot the connectivity patterns as well as the spectra
plt.subplot(2, 2, 1)
plt.pcolormesh(low_psd[:, :, low_freq_idx], cmap='Blues')
plt.xticks(np.arange(num_sources, 0, -1)-.5, np.arange(num_sources, 0, -1), fontsize=6)
plt.yticks(np.arange(num_sources, 0, -1)-.5, np.arange(num_sources, 0, -1), fontsize=6)

plt.subplot(2, 2, 2)
for k in range(num_sources):
    plt.plot(freq_vect, low_psd[k, k, :])

plt.subplot(2, 2, 3)
plt.pcolormesh(high_psd[:, :, high_freq_idx], cmap='Reds')
plt.xticks(np.arange(num_sources, 0, -1)-.5, np.arange(num_sources, 0, -1), fontsize=6)
plt.yticks(np.arange(num_sources, 0, -1)-.5, np.arange(num_sources, 0, -1), fontsize=6)

plt.subplot(2, 2, 4)
for k in range(num_sources):
    plt.plot(freq_vect, high_psd[k, k, :])

plt.xlabel('Frequency (Hz)')

plt.savefig(join(figdir, f'modal_decomp_more_{i}.png'), bbox_inches='tight')