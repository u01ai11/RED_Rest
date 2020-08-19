import mne
import matplotlib.pyplot as plt
from nilearn import plotting
import pandas as pd
import sys
import time
import os
from os import listdir as listdir
from os.path import join
from os.path import isdir
from os.path import isfile
import numpy as np

def plot_aparc_parcels(matrix, ax, fig, title):
    """
    :param matrix:
        The connectivity matrix (real or simulated) representing aparc parcels

    :param ax:
        axes to plot on
    :return:
    """

    # get labels and coordinates
    labels = mne.read_labels_from_annot('fsaverage_1', parc='aparc', subjects_dir=join('/imaging/ai05/RED/RED_MEG/resting/STRUCTURALS','FS_SUBDIR'))
    labels = labels[0:-1]
    label_names = [label.name for label in labels]
    coords = []
    # TODO: Find a better way to get centre of parcel
    #get coords of centre of mass
    for i in range(len(labels)):
        if 'lh' in label_names[i]:
            hem = 1
        else:
            hem = 0
        coord = mne.vertex_to_mni(labels[i].center_of_mass(subjects_dir=join('/imaging/ai05/RED/RED_MEG/resting/STRUCTURALS','FS_SUBDIR')), subject='fsaverage_1',hemis=hem,subjects_dir=join(MAINDIR,'FS_SUBDIR'))
        coords.append(coord[0])

    plotting.plot_connectome_strength(matrix,
                                      node_coords=coords,
                                      title=title,
                                      figure=fig,
                                      axes=ax,
                                      cmap=plt.cm.YlOrRd)

