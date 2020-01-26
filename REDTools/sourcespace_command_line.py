import mne
import os
import numpy as np
import joblib


def recon_all_multiple(sublist, struct_dir, fs_sub_dir, fs_script_dir,fs_call, njobs, cbu_clust, cshrc_path):

    """
    :param sublist: 
        A list of subjects for source recon 
    :param struct_dir: 
        A directory containing MRI scans with name format:
        subname + T1w.nii.gz 
    :param fs_sub_dir: 
        The Freesurfer subject directory 
    :param fs_call: 
        The call for freesurfer, specific to the version we want to use 
    :param njobs: 
        If not CBU cluster, we will commit jobs on the local machine.
        Relates to no, of parallel jobs 
    :param cbu_clust: 
        If true this will submit jobs to the CBU cluster queue using 'qsub'
    :param cshrc_path:
        the path to cshrc setup file to ensure the os.system environment called
        by python is setup for cluster calls properly. If this is not set they
        will not run on the cluster properly
    :return: 
    """
    if not cbu_clust:
        # set up
        # TODO: make bash version, this will only work on tcsh terminal
        # check if dir exists, make if not
        if os.path.isdir(fs_sub_dir):
            os.system(f"tcsh -c '{fs_call} && setenv SUBJECTS_DIR {fs_sub_dir}'")
        else:
            os.system(f"tcsh -c '{fs_call} && mkdir {fs_sub_dir} && setenv SUBJECTS_DIR {fs_sub_dir}'")

        saved_files = []

        if njobs == 1:
            for i in range(len(sublist)):
                savedfile = __recon_all_individual(os.path.join(sublist[i], struct_dir, fs_sub_dir))
                saved_files.append(savedfile)
        if njobs > 1:

            saved_files = joblib.Parallel(n_jobs =njobs)(
                joblib.delayed(__recon_all_individual)(os.path.join(thisS, struct_dir, fs_sub_dir)) for thisS in sublist)

        return saved_files

    else:
        #setup environment
        os.system(f'tcsh {cshrc_path}')
        saved_files = []
        # We are using CBU cluster so construct qstat jobs
        for i in range(len(sublist)):
            savedfile = __recon_all_qstat(sublist[i], struct_dir, fs_sub_dir, fs_script_dir)
            saved_files.append(savedfile)

def __recon_all_individual(sub, struct_dir, fs_sub_dir):
    """
    Private function for recon using freesurfer in tcsh shell

    :param sub:
        Subject name/number
    :param struct_dir:
        Where to find that subjects T1 weighted structural MRI scan
    :param fs_sub_dir:
        The subject dir for fressurfer, only used to return the
    :return this_sub_dir:
        The directory where freesurfer is storring it's recon files
    """
    T1_name = sub + '_T1w.nii.gz'

    if os.path.isfile(struct_dir + '/' + T1_name):
        os.system(f"tcsh -c 'recon-all -i {struct_dir}/{T1_name} -s {sub} -all -parallel'")
    else:
        print('no T1 found for ' + sub)

    this_sub_dir = f'{fs_sub_dir}/{sub}'
    return this_sub_dir

def __recon_all_qstat(sub, struct_dir, fs_sub_dir, fs_script_dir):
    """
    Private function for submitting source-recon freesurfer commands to CBU's cluster
    :param sub:
        Subject name/number
    :param struct_dir:
        Where to find that subjects T1 weighted structural MRI scan
    :param fs_sub_dir:
        The subject dir for fressurfer, only used to return the
    :return this_sub_dir:
        The directory where freesurfer is storring it's recon files

    """
    T1_name = sub + '.nii.gz'

    # construct tcsh command
    qsub_com =\
        f"""#!/bin/tcsh
        freesurfer_6.0.0 
        setenv SUBJECTS_DIR {fs_sub_dir}
        recon-all -all -i {struct_dir}/{T1_name} -s {sub} -parallel -openmp 8
        """
    #save to a csh script
    with open (f'{fs_script_dir}/{sub}.csh', "w+") as c_file:
        c_file.write(qsub_com)

    # construct the qsub command and execute
    os.system(f'sbatch --job-name=reco_{sub} --mincpus=8 -t 2-1:10 {fs_script_dir}/{sub}.csh')

    # submit
    this_sub_dir = f'{fs_sub_dir}/{sub}'
    return this_sub_dir

def fs_bem_multiple(sublist, fs_sub_dir, fs_script_dir,fs_call, njobs, cbu_clust, cshrc_path):

    """
    :param sublist:
        A list of subjects for source recon
    :param fs_sub_dir:
        The Freesurfer subject directory
    :param fs_call:
        The call for freesurfer, specific to the version we want to use
    :param njobs:
        If not CBU cluster, we will commit jobs on the local machine.
        Relates to no, of parallel jobs
    :param cbu_clust:
        If true this will submit jobs to the CBU cluster queue using 'qsub'
    :param cshrc_path:
        the path to cshrc setup file to ensure the os.system environment called
        by python is setup for cluster calls properly. If this is not set they
        will not run on the cluster properly
    :return:
    """

    # get numbers from sublist

    if not cbu_clust:
        # set up
        # TODO: make bash version, this will only work on tcsh terminal
        # check if dir exists, make if not
        if os.path.isdir(fs_sub_dir):
            os.system(f"tcsh -c '{fs_call} && setenv SUBJECTS_DIR {fs_sub_dir}'")
        else:
            os.system(f"tcsh -c '{fs_call} && mkdir {fs_sub_dir} && setenv SUBJECTS_DIR {fs_sub_dir}'")

        saved_files = []

        if njobs == 1:
            for i in range(len(sublist)):
                savedfile = __fs_bem_individual(os.path.join(sublist[i], fs_sub_dir))
                saved_files.append(savedfile)
        if njobs > 1:

            saved_files = joblib.Parallel(n_jobs=njobs)(
                joblib.delayed(__fs_bem_individual)(thisS, fs_sub_dir) for thisS in sublist)

        return saved_files

    else:
        #setup environment
        os.system(f'tcsh {cshrc_path}')
        saved_files = []
        # We are using CBU cluster so construct qstat jobs
        for i in range(len(sublist)):
            savedfile = __fs_bem_qstat(sublist[i], fs_sub_dir, fs_script_dir)
            saved_files.append(savedfile)


def __fs_bem_individual(sub, fs_sub_dir):
    """
    Private function for make BEM model using freesurfer watershed_bem

    :param sub:
        Subject name/number
    :param fs_sub_dir:
        The subject dir for fressurfer, only used to return the
    :return this_sub_dir:
        The directory where freesurfer is storring it's recon files
    """

    if os.path.isdir(fs_sub_dir + '/' + sub):
        os.system(f"tcsh -c 'freesurfer_6.0.0 && setenv SUBJECTS_DIR && mne watershed_bem -s {sub} -d {fs_sub_dir}")
    else:
        print('no folder in fs subdir for ' + sub)

    this_sub_dir = f'{fs_sub_dir}/{sub}'
    return this_sub_dir

def __fs_bem_qstat(sub, fs_sub_dir, fs_script_dir):
    """
    Private function for make BEM model using freesurfer watershed_bem - cbu cluster
    :param sub:
        Subject name/number
    :param fs_sub_dir:
        The subject dir for fressurfer, only used to return the
    :return this_sub_dir:
        The directory where freesurfer is storring it's recon files

    """

    # construct tcsh command
    qsub_com =\
        f"""#!/bin/tcsh
        freesurfer_6.0.0 
        setenv SUBJECTS_DIR {fs_sub_dir}
        mne watershed_bem -s {sub} -d {fs_sub_dir}
        """
    #save to a csh script
    with open (f'{fs_script_dir}/{sub}_bem.csh', "w+") as c_file:
        c_file.write(qsub_com)

    # construct the qsub command and execute
    os.system(f'sbatch --job-name=bem_{sub} --mincpus=1 -t 0-1:00 {fs_script_dir}/{sub}_bem.csh')

    # submit
    this_sub_dir = f'{fs_sub_dir}/{sub}'
    return this_sub_dir

