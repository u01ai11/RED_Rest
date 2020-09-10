import mne
import os
import numpy as np
import joblib


def setup_src_multiple(sublist, fs_sub_dir, outdir, spacing, surface, src_mode,n_jobs1, n_jobs2):

    """
    :param sub:
        Subjects in the freesurfer subjectdir to use
    :param fs_sub_dir:
        Freesurfer subject dir
    :param outdir:
        Where the models are being saved
    :param spacing:
        Input for MNE setup_source_space command. The spacing to use.
    :param surface:
        Input for MNE setup_source_space command. The surface to use
    :param n_jobs1:
        First parallel command, sets the number of joblib jobs to do on group level.
    :param njobs2:
        Second parallel command, sets the number of joblib jobs to submit on the participant level
    :return:
    """

    # set up

    # check if dir exists, make if not

    saved_files = []

    if n_jobs1 == 1:
        for i in range(len(sublist)):
            savedfile = __setup_src_individual(sublist[i], fs_sub_dir, outdir, spacing, surface, src_mode,n_jobs2)
            saved_files.append(savedfile)
    if n_jobs1 > 1:

        saved_files = joblib.Parallel(n_jobs =n_jobs1)(
            joblib.delayed(__setup_src_individual)(thisS, fs_sub_dir, outdir, spacing, surface,src_mode, n_jobs2) for thisS in sublist)

    return saved_files



def __setup_src_individual(sub, fs_sub_dir, outdir, spacing, surface, src_mode, njobs):
    """
    :param sub:
        subject to set up source-space on
    :param fs_sub_dir:
        where to find the fs recon-all files
    :param spacing:
        spacing to use for source-space
    :param surace:
        surface to use for source-space
    :param njobs:
        how many jons to split this up into
    :return:
    """
    # check if already exists.
    fname = outdir + '/' + sub + '_' + surface + '-' + spacing +  '-' + src_mode + '-src.fif'
    if os.path.isfile(fname):
        print(fname + ' already exists')
        return fname
    try:
        if src_mode == 'cortical':
            src_space = mne.setup_source_space(sub, spacing=spacing, surface=surface, subjects_dir=fs_sub_dir, n_jobs=njobs)
        elif src_mode == 'volume':
            src_space = mne.setup_volume_source_space(sub, subjects_dir=fs_sub_dir,pos=8.0)
        else:
            print(src_mode + ' is not a valid source space')
            return ''
        mne.write_source_spaces(fname, src_space)  # write to
        this_sub_dir = fname
    except Exception as e:
        print('something went wrong with setup, skipping ' + sub)
        print(e)
        return ''

    return this_sub_dir




def make_bem_multiple(sublist, fs_sub_dir, outdir, single_layers, n_jobs1):

    """
    :param sublist:
        Subjects in the freesurfer subjectdir to use
    :param fs_sub_dir:
        Freesurfer subject dir
    :param outdir:
        Where the models are being saved
    :param single_layers:
        Boolean weather to allow single layer models or not
    :param n_jobs1:
        First parallel command, sets the number of joblib jobs to do on group level.

    :return:
    """

    # set up

    # check if dir exists, make if not

    saved_files = []

    if n_jobs1 == 1:
        for i in range(len(sublist)):
            savedfile = __make_bem_individual(sublist[i], fs_sub_dir, outdir, single_layers)
            saved_files.append(savedfile)
    if n_jobs1 > 1:

        saved_files = joblib.Parallel(n_jobs=n_jobs1)(
            joblib.delayed(__make_bem_individual)(thisS, fs_sub_dir, outdir, single_layers) for thisS in sublist)

    return saved_files


def __make_bem_individual(sub, fs_sub_dir, outdir, single_layers):
    """

    :param sub:
    :param fs_sub_dir:
    :param single_layers:
    :return:
    """

    #  see if file exists and skip if so
    if os.path.isfile(f'{outdir}/{sub}-5120-5120-5120-bem.fif'):
        print(f'{sub} has full file skipping')
        model = mne.read_bem_surfaces(f'{outdir}/{sub}-5120-5120-5120-bem.fif')
        solname = f'{outdir}/{sub}-5120-5120-5120-bem-sol.fif'
    # if single layers is true check for this, if not we want to try full model
    elif os.path.isfile(f'{outdir}/{sub}-5120-5120-5120-single-bem.fif'):
        if single_layers:
            print(f'{sub} has single layer file skipping')
            model = mne.read_bem_surfaces(f'{outdir}/{sub}-5120-5120-5120-single-bem.fif')
            solname = f'{outdir}/{sub}-5120-5120-5120-single-bem-sol.fif'
    else:

    #  make model
        try:
            model = mne.make_bem_model(sub, subjects_dir=fs_sub_dir)
            bemname = f'{outdir}/{sub}-5120-5120-5120-bem.fif'
            solname = f'{outdir}/{sub}-5120-5120-5120-bem-sol.fif'
        except:
            print('failed to make BEM model with input')
            if single_layers:
                try:
                    print('falling back to single layer model due to BEM suckiness')
                    model = mne.make_bem_model(sub, subjects_dir=fs_sub_dir, conductivity=[0.3])
                    bemname = f'{outdir}/{sub}-5120-5120-5120-single-bem.fif'
                    solname = f'{outdir}/{sub}-5120-5120-5120-single-bem-sol.fif'
                except:
                    print(f'oops that also failed for {sub}')
                    return ''

            else:
                print('wont allow single layer model so skipping')
                return ''

        # save model
        mne.write_bem_surfaces(bemname, model)  # save to source dir

    bem_sol = mne.make_bem_solution(model)  # make bem solution using model
    mne.write_bem_solution(solname, bem_sol) # save as well to the outdir
    return solname



def make_inv_multiple(rawfs, transfs, bemfs, srcfs, outdir, njobs):

    saved_files = []

    if njobs == 1:
        for i in range(len(rawfs)):
            savedfile = __make_inv_individual(rawfs[i], transfs[i], bemfs[i], srcfs[i], outdir)
            saved_files.append(savedfile)
    if njobs > 1:

        saved_files = joblib.Parallel(n_jobs=njobs)(
            joblib.delayed(__make_inv_individual)(raw, tran, bem, src, outdir) for raw, tran, bem, src in zip(rawfs, transfs, bemfs, srcfs))

    return saved_files

def __make_inv_individual(rawf, transf, bemf, srcf, outdir):

    tmpid = os.path.basename(rawf).split("_")[0]
    tmpath = f'{outdir}/{tmpid}_inv.fif'

    if os.path.isfile(tmpath):
        print(f'{tmpid}_inv.fif already exists, skipping')
        return tmpath
    try:
        raw = mne.io.read_raw_fif(rawf)
        cov = mne.compute_raw_covariance(raw)
        src = mne.read_source_spaces(srcf)
        bem = mne.read_bem_solution(bemf)
        fwd = mne.make_forward_solution(raw.info, transf, src, bem)
        inv = mne.minimum_norm.make_inverse_operator(raw.info, fwd, cov)
        del fwd, src

        mne.minimum_norm.write_inverse_operator(f'{outdir}/{tmpid}_inv.fif',inv)
    except:
        print('error')
    return tmpath

