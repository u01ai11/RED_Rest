import mne
import os
import numpy as np
import joblib

def cluster_lcmv(RED_ids):
    pycom = """
import sys
import os
from os.path import join
import numpy as np
import mne
sys.path.insert(0, '/imaging/ai05/RED/RED_MEG/resting/analysis/RED_Rest')
from REDTools import study_info

MP_name = 'Series005_CBU_MPRAGE_32chn'
dicoms = '/imaging/rs04/RED/DICOMs'
STRUCTDIR = '/imaging/ai05/RED/RED_MEG/resting/STRUCTURALS'
MAINDIR = '/imaging/ai05/RED/RED_MEG/resting'
folders = [f for f in os.listdir(dicoms) if os.path.isdir(os.path.join(dicoms,f))]
ids = [f.split('_')[0] for f in folders]

#get ids etc
RED_id, MR_id, MEG_fname = study_info.get_info()

def lcmvsrc(meg_f, red_id, method, outdir):
    raw = mne.io.read_raw_fif(meg_f, preload=True)
    cov = 
    start, stop = raw.time_as_index([60, raw.times[-30000]])
    stc = mne.minimum_norm.apply_inverse_raw(raw, inv,
                                             lambda2=1.0/1.0**2,
                                             method=method,
                                             start=start,
                                             stop=stop,
                                             buffer_size=int(len(raw.times)/10))

invdir = join(MAINDIR, 'inverse_ops')
rawdir = join(MAINDIR, 'preprocessed')
fsdir = join(MAINDIR, 'STRUCTURALS','FS_SUBDIR')
meg_f = join(rawdir, MEG_fname[{i}])
inv_op = join(invdir, f'{{RED_id[{i}]}}-inv.fif')
outdir = join(MAINDIR, 'raw_stcs')
method = 'MNE'
raw2source(meg_f, inv_op, method,outdir)
    """