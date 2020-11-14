
from os.path import join
from os.path import isfile
import numpy as np
import mne
import sails
import glmtools
import joblib
from numpy import random


def MVAR_single(ind, type, modes, filter, outdir, parcel_dir, parcel_files, sample_rate, metric, parcel_exclude, overwrite):

    """
    Runs a multivariate auto-regression on timefrequency data, and saves the output

    :param ind: Index of the file
    :param type: type of MVAR to run - veiramorf or OLS
    :param modes: number of modes for the AR model
    :param filter: what type of filtering and what parameters to use
    :param outdir: the directory to save this in
    :param parcel_dir: the directory to find the parcel timecourse files
    :param parcel_files: a list of parcel files to use as data input
    :param sample_rate: sample rate of the data
    :param metric: the metric to use
    :param parcel_exclude: list of parcels index to remove
    :param overwrite: True or False to overwrite file
    :return: returns the model parameters fourier metrics, and the model itself

    """

    #id
    id_ = parcel_files[ind].split('_')[0]
    if overwrite == False:
        if isfile(join(outdir, f'mvar_{type}_{id_}.npy')):
            print('file exists, skipping')
            return
    X = np.load(join(parcel_dir, parcel_files[ind]))

    if len(parcel_exclude) > 0:
        X = np.delete(X,parcel_exclude,1)


    if filter == 'notch':
        X = mne.filter.notch_filter(X, Fs=sample_rate, freqs=np.arange(50, 75, 50), verbose=True)
    elif type(filter) == tuple:
        # we also probably want to filter our data slightly (use FIR)
        X = mne.filter.filter_data(X, sfreq=sample_rate, l_freq=filter[0], h_freq=filter[1], verbose=True)
    else:
        print(f'{filter} is an unrecognised filter, not filtering')

    if len(X.shape) == 1:
        print('not correct data input, skipping')
        return
    #reshape as sails expects (nsignals, nsamples, ntrials)
    X = X.transpose([1,2,0])

    # Remove some bad segments - just set the to zero.
    X = sails.utils.detect_artefacts(X, axis=1,
                                     reject_mode='segments', segment_len=100,
                                     ret_mode='zero_bads', gesd_args={'alpha':0.1})
    # create delay vector from modes
    delay_vect = np.arange(modes)

    # Downsample even more
    X = sails.utils.fast_resample(X, ds_factor=2)
    sample_rate = sample_rate/2


    # Remove some random low variance channels.... probably not great but works for the moment..
    # keeps = np.argsort(X.std(axis=(1,2)))[18:]
    # X = X[keeps,:,:]


    # try to orthoganlise
    try:
        X[:,:,0] = sails.orthogonalise.symmetric_orthonormal(X[:,:,0], maintain_mag=False)[0]
    except:
        print(f'failed othorganilsation for {id_}')

    # apply model
    if type == 'OLS':
        m = sails.OLSLinearModel.fit_model(X, delay_vect)
    elif type == 'VieiraMorf':
        m = sails.VieiraMorfLinearModel.fit_model(X, delay_vect)
    # get fourier decomp of model coefficients

    freq_vect = np.linspace(0, sample_rate/2, 36) #frequency vector representing the model metrics
    Fo = sails.mvar_metrics.FourierMvarMetrics.initialise(m, sample_rate, freq_vect)


    # choose metric
    if metric == 'direct_transfer_function':
        out = Fo.directed_transfer_function
    elif metric == 'partial_directed_coherence':
        out = Fo.partial_directed_coherence

    # save to file
    # get name
    print(ind)
    np.save(join(outdir, f'mvar_{type}_{id_}.npy'),out)
    return Fo, m

def surrogate_MVAR(perm, ind, type, modes, filter, outdir, parcel_dir, parcel_files, sample_rate, metric):

    """
    Creates a surrogate timeseries, with phase shuffled but amplitude and frequency spectra maintained
    Runs a multivariate auto-regression on timefrequency data, and saves the output

    ":param perm: what permutation this is, for file name
    :param ind: Index of the file
    :param type: type of MVAR to run - veiramorf or OLS
    :param modes: number of modes for the AR model
    :param filter: what type of filtering and what parameters to use
    :param outdir: the directory to save this in
    :param parcel_dir: the directory to find the parcel timecourse files
    :param parcel_files: a list of parcel files to use as data input
    :param sample_rate: sample rate of the data
    :return: returns the model parameters fourier metrics, and the model itself

    """



    #id
    id_ = parcel_files[ind].split('_')[0]


    # we only need to fourier transform once, if not first permutation read from cache file
    X = np.load(join(parcel_dir, parcel_files[ind]))

    if filter == 'notch':
        X = mne.filter.notch_filter(X, Fs=sample_rate, freqs=np.arange(50, 75, 50), verbose=False)
    elif type(filter) == tuple:
        # we also probably want to filter our data slightly (use FIR)
        X = mne.filter.filter_data(X, sfreq=sample_rate, l_freq=filter[0], h_freq=filter[1], verbose=False)
    else:
        print(f'{filter} is an unrecognised filter')
        return

    if len(X.shape) == 1:
        print('not correct data input, skipping')
        return

    # fast fourier decomposition
    X_fft = np.fft.rfft(X[0], axis=1)


    #reshape as sails expects (nsignals, nsamples, ntrials) Do this for orthogonalising and segment removal
    X = X.transpose([1,2,0])

    # Remove some bad segments - just set the to zero.
    X = sails.utils.detect_artefacts(X, axis=1,
                                     reject_mode='segments', segment_len=100,
                                     ret_mode='zero_bads', gesd_args={'alpha':0.1})


    # Downsample even more
    X = sails.utils.fast_resample(X, ds_factor=2)
    sample_rate = sample_rate/2


    X[:,:,0] = sails.orthogonalise.symmetric_orthonormal(X[:,:,0], maintain_mag=False)[0]



    # transpose back to expected shape for MNE stuff
    X = X.transpose([2,0,1])

    # shuffle phases
    #  Get shapes
    (N, n_time) = X[0].shape
    len_phase = X_fft.shape[1]

    #  Generate random phases uniformly distributed in the
    #  interval [0, 2*Pi]
    phases = random.uniform(low=0, high=2 * np.pi, size=(N, len_phase))

    #  Add random phases uniformly distributed in the interval [0, 2*Pi]
    X_fft *= np.exp(1j * phases)

    #  Calculate IFFT and take the real part, the remaining imaginary part
    #  is due to numerical errors.

    X_surr = np.ascontiguousarray(np.real(np.fft.irfft(X_fft, n=n_time,
                                                        axis=1)))

    X[0,:,:] = X_surr # add to X
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
    freq_vect = np.linspace(0, sample_rate/2, 36) #frequency vector representing the model metrics
    Fo = sails.mvar_metrics.FourierMvarMetrics.initialise(m, sample_rate, freq_vect)
    # save to file
    # get name

    # choose metric
    if metric == 'direct_transfer_function':
        out = Fo.directed_transfer_function
    elif metric == 'partial_directed_coherence':
        out = Fo.partial_directed_coherence

    np.save(join(outdir, f'mvar_surr_{type}_{id_}_{perm}.npy'),out)
    return Fo, m


def single_perm(type, modes, filter, outdir, parcel_dir,
                parcel_files, sample_rate, glm_regs, perm, metric, outstat):

    """
    Performs a single permutation on a group level GLM on the Direct Transfer Function metric of a
    multivariate autoregression (MVAR) model.

    "
    :param type: type of MVAR to run - veiramorf or OLS
    :param modes: number of modes for the AR model
    :param filter: what type of filtering and what parameters to use
    :param outdir: the directory to save this in
    :param parcel_dir: the directory to find the parcel timecourse files
    :param parcel_files: a list of parcel files to use as data input
    :param sample_rate: sample rate of the data
    :param glm_regs: a list of simple covariates for the regressors
    :param perm: index of permutation to avoid over-writing
    :param metric: the statistic to use from the individual MVAR
    :param outstat: what stat we want back from the glm for the null: beta or tstat
    :return: returns the model parameters fourier metrics, and the model itself

    """

    #loop through all participants and create surrogate data
    joblib.Parallel(n_jobs =10)(
    joblib.delayed(surrogate_MVAR)(perm, i, type, modes, filter, outdir,
                                   parcel_dir, parcel_files, sample_rate, metric) for i in range(len(parcel_files)))

    # get all that surrogate data into format for GLM
    #load one file to get shape info
    X = np.load(join(parcel_dir, parcel_files[0]))
    glm_data = np.empty((len(parcel_files), X.shape[1],X.shape[1],36))
    for i in range(len(parcel_files)):
        id_ = parcel_files[i].split('_')[0]
        glm_data[i,:,:,:] = np.load(join(outdir, f'mvar_surr_{type}_{id_}_{perm}.npy'))[:,:,:,0]

    # Now we need to design the GLM
    dat = glmtools.data.TrialGLMData(data=glm_data, dim_labels=['participants', 'parcel_drivers', 'parcel_recievers', 'frequency'])
    #holders
    regs = list()
    contrasts = list()
    #append the  intercept
    regs.append(glmtools.regressors.ConstantRegressor(num_observations=dat.info['num_observations']))
    contrasts.append(glmtools.design.Contrast(name='Intercept',values=[1]+[0]*len(glm_regs))) # for regressor

    # append each of the regressors passed in
    for i in range(len(glm_regs)):
        # add regressor for Age
        regs.append(glmtools.regressors.ParametricRegressor(values=glm_regs[i],
                                                            name=f'reg_{i}',
                                                            preproc='z',
                                                            num_observations=dat.info['num_observations']))
        tmp_val = [0]*(len(glm_regs)+1)
        tmp_val[i+1] = 1
        contrasts.append(glmtools.design.Contrast(name=f'reg_{i}',values=tmp_val))

    #carry out the glm
    des = glmtools.design.GLMDesign.initialise(regs,contrasts)
    model = glmtools.fit.OLSModel( des, dat )
    if outstat == 'beta':
        output = model.betas
    if outstat == 'tstat':
        output = model.get_tstats()
    #return tstats
    return output


