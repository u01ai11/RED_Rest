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
import holoviews as hv
hv.extension('bokeh')
from holoviews import opts, dim

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



def hv_chord(contrast, frequency, threshold, stats, re_order_ind, label_names, des, freq_vect):

    """
    Makes a holoview/boken chord diagram
    :param contrast:
        Which contrast in the glm data are we looking at
    :param frequency:
        The index of the frequency in the freq_vect we are looking at
    :param threshold:
        The percentile threshold for plotting (so the plot isn't messy)
    :param stats:
        The stats array we are plotting from
    :param re_order_ind:
        The indices for re-ordering the data into the Y-order of the parcels in the brain
    :param label_names:
        The names for each parcel
    :param des:
        The design matrix from the GLM
    :param freq_vect:
        The frequency vector containing the actual frequencies of each
    :return:
    """
    dtypes = np.dtype([
        ('source', int),
        ('target', int),
        ('value', int),
    ])

    data = np.empty(0, dtype=dtypes)
    links = pd.DataFrame(data)

    square = stats[contrast,:,:,frequency]


    #square = stats[contrast,:,:,:].sum(axis=2)
    thresh_mask = square > np.percentile(square, threshold)
    square[~thresh_mask] = 0

    #reorder
    X_sorted = np.copy(square)
    # Sort along first dim
    X_sorted = X_sorted[re_order_ind,:]
    # Sort along second dim
    X_sorted = X_sorted[:,re_order_ind]

    labels_sorted = np.array(label_names)[re_order_ind]
    # loop through Y axis of matrix
    counter = 0
    for i in range(X_sorted.shape[0]):
        for ii in range(X_sorted.shape[1]):
            links.loc[counter] = [i, ii, int(X_sorted[i,ii])]
            counter +=1

    # make label index
    dtypes = np.dtype([
        ('name', int),
        ('group', int),
    ])

    data = np.empty(0, dtype=dtypes)
    nodes = pd.DataFrame(data)

    for i in range(X_sorted.shape[0]):
        nodes.loc[i] = [labels_sorted[i], 1]

    graph = hv.Chord((links, hv.Dataset(nodes, 'index')))
    graph.opts(
        opts.Chord(cmap='Category20', edge_cmap='Category20', edge_color=dim('source').str(),
                   labels='name', node_color=dim('index').str())
    )
    graph.relabel('Directed Graph').opts(directed=True)
    graph.opts(title=f'{des.contrast_names[contrast]} Partial Directed Coherence @ {int(freq_vect[frequency])}Hz')
    return graph

def get_labels():
    MAINDIR = join('/imaging/ai05/RED/RED_MEG/resting/STRUCTURALS')
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
        coord = mne.vertex_to_mni(labels[i].center_of_mass(subjects_dir=join('/imaging/ai05/RED/RED_MEG/resting/STRUCTURALS','FS_SUBDIR')),
                                  subject='fsaverage_1',hemis=hem,subjects_dir=join(MAINDIR,'FS_SUBDIR'))
        coords.append(coord[0])


    # First, we reorder the labels based on their location in the left hemi
    lh_labels = [name for name in label_names if name.endswith('lh')]
    rh_labels = [name for name in label_names if name.endswith('rh')]

    # Get the y-location of the label
    label_ypos = list()
    for name in lh_labels:
        idx = label_names.index(name)
        ypos = np.mean(labels[idx].pos[:, 1])
        label_ypos.append(ypos)

    lh_labels = [label for (yp, label) in sorted(zip(label_ypos, lh_labels))]

    # For the right hemi
    rh_labels = [label[:-2] + 'rh' for label in lh_labels]

    # make a list with circular plot order
    node_order = list()
    node_order.extend(lh_labels[::-1])  # reverse the order
    node_order.extend(rh_labels)

    node_order = node_order[::-1] # reverse the whole thing

    # get a mapping to the original list
    re_order_ind = [label_names.index(x) for x in node_order]

    return labels, label_names, re_order_ind