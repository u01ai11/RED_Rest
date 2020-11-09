import sails
import numpy as np
from scipy import signal

# Load data
dataset = '/Users/andrew/Projects/sails/Irvine/2_parcel_timecourse_lowrank.npy'
X = np.load(dataset)
X = np.moveaxis(X,0,2)
sample_rate = 150

# Downsample even more
X = sails.utils.fast_resample(X, ds_factor=2)
sample_rate = sample_rate/2

# Remove some random low variance channels.... probably not great but works for the moment..
keeps = np.argsort(X.std(axis=(1,2)))[18:]
X = X[keeps,:,:]

# Remove some bad segments - just set the to zero.
X = sails.utils.detect_artefacts(X, axis=1,
                                 reject_mode='segments', segment_len=100,
                                 ret_mode='zero_bads', gesd_args={'alpha':0.1})

# Orthogonalise
X,_ = sails.orthogonalise.symmetric_orthonormal(X[:,:,0])
X = X[:,:,None] # Put the dummy dim back in

# Compute model
model = sails.VieiraMorfLinearModel.fit_model(X,np.arange(14))

# Compute power spectra and connectivity
freq_vect = np.linspace(0,sample_rate/2)
metrics = sails.FourierMvarMetrics.initialise(model,sample_rate,freq_vect)

plt.figure()
plt.subplot(121)
f,pxx = signal.welch(X[:,:,0],nperseg=2048, fs=sample_rate,scaling='spectrum')
plt.plot(f,pxx.T)
plt.grid(True)
ax = plt.subplot(122)
metrics.plot_diags('PSD', ax=ax)
