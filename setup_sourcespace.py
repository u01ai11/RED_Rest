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
import joblib

#%% Convert DICOM folder to nifti

#located in imaging/rs04/RED/DICOMs

# The T1s DICOMS are located in a folder in a folder for each participants
# The name of this folder is the same for each participant
MP_name = 'Series005_CBU_MPRAGE_32chn'
dicoms = '/imaging/rs04/RED/DICOMs'
STRUCTDIR = '/imaging/ai05/RED/RED_MEG/resting/STRUCTURALS'
MAINDIR = '/imaging/ai05/RED/RED_MEG/resting'
folders = [f for f in os.listdir(dicoms) if os.path.isdir(os.path.join(dicoms,f))]
ids = [f.split('_')[0] for f in folders]

#%%
for part in folders:
    subfolder = [f for f in listdir(join(dicoms, part)) if isdir(join(dicoms, part,f))]

    if len(subfolder) == 1:
        #find the MPRAGE folder
        mprage_f = [f for f in listdir(join(dicoms, part, subfolder[0])) if 'MPRAGE' in f]
        if len(mprage_f) > 1:
            mprage_f = join(subfolder[0], mprage_f[-1])
        else:
            mprage_f = join(subfolder[0], mprage_f[0])
    else:
        # if multiple folders take the most recent one
        mprages = []
        for sub in subfolder:
            tmp_mprage = [f for f in listdir(join(dicoms, part, sub)) if 'MPRAGE' in f]
            for i in range(len(tmp_mprage)):
                mprages.append(join(sub,tmp_mprage[i]))
        print(mprages)
        mprage_f = mprages[-1]

    if [f for f in mprage_f.split("/")[1] if f.isdigit()][2] == '0':
        outname = '10_cbu_mprage_32chn.nii.gz'
    else:
        outname = [f for f in mprage_f.split("/")[1] if f.isdigit()][2] + '_cbu_mprage_32chn.nii.gz'
    if os.path.isfile(join(STRUCTDIR, "T1", part.split("_")[0]+".nii.gz")):
        print('already there skipping')
    else:
        # Convert to Dicom
        dicom2nifti.convert_directory(join(dicoms, part, mprage_f), join(STRUCTDIR, 'T1'), compression=True, reorient=True)
        # rename to subject name
        os.system(f'mv {join(STRUCTDIR, "T1", outname)} {join(STRUCTDIR, "T1", part.split("_")[0])}.nii.gz')

#%% FREESURFER RECON
fs_recon_list = sourcespace_command_line.recon_all_multiple(sublist=['CBU190573'],
                                                   struct_dir= join(STRUCTDIR, "T1"),
                                                   fs_sub_dir=join(STRUCTDIR, "FS_SUBDIR"),
                                                   fs_script_dir='/imaging/ai05/RED/RED_MEG/resting/cluster_scripts',
                                                   fs_call='freesurfer_6.0.0',
                                                   njobs=1,
                                                   cbu_clust=True,
                                                   cshrc_path='/home/ai05/.cshrc')

#%% Check status of files
fs_subdir = join(STRUCTDIR, "FS_SUBDIR")
folders = [join(fs_subdir, f) for f in listdir(fs_subdir) if 'CBU' in f]
folder_ids = listdir(fs_subdir)
status = []
messages = []
for i, folder in enumerate(folders):
    log_f = join(folder, 'scripts', 'recon-all.log')
    with open(log_f) as myfile:
        lines = myfile.readlines()
    if 'finished without error' in lines[-1]:
        status.append(True)
    else:
        status.append(False)
    messages.append(lines[-1])

print(f'{sum(status)/len(ids)*100}% recon-alls complete without errors')

#%% proceed to generate BEM models from the completed freesurfers
#%% make BEM model
# get all participants who have a fs_dir and error-less logs from recon-all
subs2do = np.array([f.split('/')[-1] for f in folders])[status]
#%%
# run run run run
fs_recon_list = sourcespace_command_line.fs_bem_multiple(sublist=subs2do,
                                                fs_sub_dir=join(STRUCTDIR, "FS_SUBDIR"),
                                                fs_script_dir='/imaging/ai05/RED/RED_MEG/resting/cluster_scripts',
                                                fs_call='freesurfer_6.0.0',
                                                njobs=1,
                                                cbu_clust=True,
                                                cshrc_path='/home/ai05/.cshrc')
#%% Check the BEM model by saving images for each participant
BEM_DIAG_FOLDER = join(STRUCTDIR, 'DIAGNOSTICS', 'BEM_IMAGES')
for sub in subs2do:
    plt.close('all')
    mne.viz.plot_bem(subject=sub,
                     subjects_dir=join(STRUCTDIR, "FS_SUBDIR"),
                     brain_surfaces='white',
                     orientation='coronal').savefig(join(BEM_DIAG_FOLDER, sub+'.png'))

#%% Next we need to coregister our MEG data and MRI models. This has to be done manually, checkout coreg.py for this
os.system('python coreg.py')
#%% Now we have that we need to read in stuff to mne
#get ids etc
RED_id, MR_id, MEG_fname = study_info.get_info()

mne_src_files = sourcespace_setup.setup_src_multiple(sublist=MR_id,
                                                         fs_sub_dir=join(STRUCTDIR, 'FS_SUBDIR'),
                                                         outdir=join(MAINDIR, 'src_space'),
                                                         spacing='oct6',
                                                         surface='white',
                                                         src_mode='volume',
                                                         n_jobs1=14,
                                                         n_jobs2=1)

#%% Bem setup
mne_bem_files = sourcespace_setup.make_bem_multiple(sublist=MR_id,
                                                        fs_sub_dir=join(STRUCTDIR, 'FS_SUBDIR'),
                                                        outdir=join(MAINDIR, 'bem'),
                                                        single_layers=True,
                                                        n_jobs1=20)

#%% Compute covariance, forward model etc
fs_subdir = join(STRUCTDIR, "FS_SUBDIR")
i = 0
raw = mne.io.read_raw_fif(join(MAINDIR, 'preprocessed', MEG_fname[i]))
cov = mne.compute_raw_covariance(raw)
trans = join(MAINDIR, 'coreg', f'{RED_id[i]}-trans.fif')
bem = join(MAINDIR, 'bem', f'{MR_id[i]}-5120-5120-5120-single-bem-sol.fif')
#
src = mne.setup_volume_source_space(MR_id[i], subjects_dir=fs_subdir,pos=8.0)


fwd = mne.make_forward_solution(raw.info, trans, src, bem)
inv = mne.minimum_norm.make_inverse_operator(raw.info, fwd, cov)

events = mne.make_fixed_length_events(raw, duration=10)
epochs = mne.Epochs(raw, events=events, tmin=0, tmax=10,
                    baseline=None, preload=True)

stcs = mne.minimum_norm.apply_inverse_epochs(epochs[5:6], inv, lambda2=1. / 9.,
                            return_generator=False)
img = stcs[0].as_volume(src,
                    mri_resolution=False)  # set True for full MRI resolution


# Save it as a nifti file
# nib.save(img, 'mne_%s_inverse.nii.gz' % method)

t1_fname = mri='/home/ai05/osl/std_masks/MNI152_T1_8mm_brain.mgz'
from nilearn.plotting import plot_stat_map
from nilearn.image import index_img

# Plotting with nilearn ######################################################
mne.viz.plot_stat_map(index_img(img, 61), t1_fname, threshold=8.,
              title='%s (t=%.1f s.)' % ('dspm', stcs[0].times[61]))