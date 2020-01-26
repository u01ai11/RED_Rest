import sys
import os
from os import listdir as listdir
from os.path import join
from os.path import isdir
import dicom2nifti
sys.path.insert(0, '/imaging/ai05/RED/RED_MEG/resting/analysis/RED_Rest')
from REDTools import sourcespace_setup
from REDTools import sourcespace_command_line


#%% Convert DICOM folder to nifti

#located in imaging/rs04/RED/DICOMs

# The T1s DICOMS are located in a folder in a folder for each participants
# The name of this folder is the same for each participant
MP_name = 'Series005_CBU_MPRAGE_32chn'
dicoms = '/imaging/rs04/RED/DICOMs'
STRUCTDIR = '/imaging/ai05/RED/RED_MEG/resting/STUCTURALS'
folders = [f for f in os.listdir(dicoms) if os.path.isdir(os.path.join(dicoms,f))]
ids = [f.split('_')[0] for f in folders]

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

    # Convert to Dicom
    dicom2nifti.convert_directory(join(dicoms, part, mprage_f), join(STRUCTDIR, 'T1'), compression=True, reorient=True)
    # rename to subject name
    try:
        outname = [f for f in mprage_f.split("/")[1] if f.isdigit()][2] + '_cbu_mprage_32chn.nii.gz'
        os.system(f'mv {join(STRUCTDIR, "T1", outname)} {join(STRUCTDIR, "T1", part.split("_")[0])}.nii.gz')
    except:
        outname = '10_cbu_mprage_32chn.nii.gz'
        os.system(f'mv {join(STRUCTDIR, "T1", outname)} {join(STRUCTDIR, "T1", part.split("_")[0])}.nii.gz')

#%% FREESURFER RECON
subnames_only = list(set([x.split('_')[0] for x in flist])) # get a unique list of IDs

fs_recon_list = red_sourcespace_cmd.recon_all_multiple(sublist=subnames_only,
                                                   struct_dir= struct_dir,
                                                   fs_sub_dir=fs_sub_dir,
                                                   fs_script_dir='/imaging/ai05/phono_oddball/fs_scripts',
                                                   fs_call='freesurfer_6.0.0',
                                                   njobs=1,
                                                   cbu_clust=True,
                                                   cshrc_path='/home/ai05/.cshrc')
