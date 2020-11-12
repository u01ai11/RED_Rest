import sys
from os import listdir as listdir
from os.path import join
from os.path import isfile
import numpy as np
sys.path.insert(0, '/imaging/ai05/RED/RED_MEG/resting/analysis/RED_Rest')
from REDTools import plotting as red_plotting
import mne
import joblib
import copy
import pandas as pd
import os
from nilearn import plotting
import sails
import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib import patches
import scipy as ss
import holoviews as hv
hv.extension('bokeh')
from holoviews import opts, dim
from bokeh.io import output_file, save, show
from bokeh.io import curdoc
from bokeh.layouts import layout
from bokeh.models import Slider, Button
import networkx as nx

#% setup
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

# define some information about our data
sample_rate = 150 # sample rate
freq_vect = np.linspace(0, sample_rate/2, 36) #frequency vector representing the model metrics

outdir = join(root_dir,'resting', 'MVAR')

#%% load data for the null montecarlo distribution and the model
model = np.load(join(outdir, 'MVAR_GLM_MODEL.npy'), allow_pickle=True)[()]

# if we already have the thresholds saved load them, otherwise calculate the new ones and save
if isfile(join(outdir, 'MVAR_GLM_95_thresh.npy')):
    thresh = nulls_surr = np.load(join(outdir, 'MVAR_GLM_99_thresh.npy'))

    # This will take a while due to numpy loading being slow
else:
    nulls_surr = np.load(join(outdir, 'MVAR_GLM_NULLS_surr.npy'))
    nulls_shuff = np.load(join(outdir, 'MVAR_GLM_NULLS_shuff.npy'))
    # get threshold
    thresh_surr = np.percentile(nulls_surr, 99, axis=4)
    thresh_shuff = np.percentile(nulls_shuff, 99, axis=4)

    thresh  = np.concatenate((thresh_surr, thresh_shuff))
    np.save(join(outdir, 'MVAR_GLM_99_thresh.npy'), thresh, allow_pickle=True)

sig_= model.tstats >= thresh # mask for data

# make copy with zero-out
stats = copy.deepcopy(model.tstats)
stats[~sig_] = 0

betas = copy.deepcopy(model.betas)
betas[~sig_] = 0
#%% get labels and coordinates for plotting in brains
# use our plotting function to look at the results
# get labels and coordinates
labels, label_names, re_order_ind = red_plotting.get_labels()

#%% Render interactive chord plots
renderer = hv.renderer('bokeh')
hv.output(size=250)
freq_range = [0, 5, 10, 15, 20, 25, 30, 35]
start = 0
end = 36
#%% Intercept
hmap = hv.HoloMap({i: red_plotting.hv_chord(0, i,  0, stats, re_order_ind, label_names, model, freq_vect) for i in freq_range})
plot = renderer.get_plot(hmap)
renderer.save(plot, join(figdir, 'chord_MVAR_Intercept_animated'))
#%% AGE
hmap = hv.HoloMap({i: red_plotting.hv_chord(1, i, 0, stats, re_order_ind, label_names, model, freq_vect) for i in freq_range})
plot = renderer.get_plot(hmap)
renderer.save(plot, join(figdir, 'chord_MVAR_AGE_animated'))
#%% IQ
hmap = hv.HoloMap({i: red_plotting.hv_chord(2, i,  0, stats, re_order_ind, label_names, model, freq_vect) for i in freq_range})
plot = renderer.get_plot(hmap)
renderer.save(plot, join(figdir, 'chord_MVAR_IQ_animated'))

#%% MRI



#%% use sails to plot sub section of N parcellations

plt_stat = copy.deepcopy(stats)
n = 7
cont = 2
sub_section = np.zeros([n,n,len(freq_vect),1])
sub_section[:,:,:,0] = plt_stat[cont, 0:n,0:n,:]

fig, ax= plt.subplots()
sails.plotting.plot_vector(sub_section,x_vect=freq_vect,F=fig)
plt.savefig(join(figdir, f'test_vect_betas_IQ.png'))
plt.close(fig)
 #%% Make stacked bar charts / Sankey graphs
plt_stat = copy.deepcopy(stats)
contrast = 1 # 0 = Intercept; 1 = Age; 2 = IQ
top_n = 68 # choose the top X connected parcels to plot
color_n = 5 # top X parcels to be coloured
rev_opt = False
direction = 1 # 1 = outgoing, 0 = ingoing


def plot_stacked(plt_stat, contrast, top_n, color_n, rev_opt, direction, freq_vect, label_names, cust_name):
    """
    :param plt_stat:
        Stats to create the plot, must be a contast x parcel x parcel x frequency array
    :param contrast:
        The contrast from the GLM we want to plot
    :param top_n:
        The top n parcels included in the plot
    :param color_n:
        N parcels we want to be coloured
    :param rev_opt:
        Reverese the order of the plot
    :param direction:
        Are we measuring ingoing or outgoing connections
    :param freq_vect:
        Frequency vector
    :param label_names:
        Names of the n parcels included in the data
    :return:
    """

    # make a matrix frequency * top_n matrix
    stacked = np.zeros([len(freq_vect), top_n])
    s_labels =[['']*top_n]*len(freq_vect) # make one to hold the labels for each value
    # for each frequency work out the top n parcels and the percentage of the top they possess

    for i in range(len(freq_vect)):
        this_data = plt_stat[contrast, :,:,i].sum(axis=direction) # sum accross outgoing connections
         # order labels by magnitude
        this_labels = [label for (yp, label) in sorted(zip(this_data, label_names), reverse=rev_opt)]
        this_values = sorted(this_data, reverse=rev_opt)

        #take top n parcels
        if rev_opt == True:
            this_labels = this_labels[0:top_n]
            this_values = this_values[0:top_n]
        else:
            this_labels = this_labels[len(this_labels)-top_n:len(this_labels)]
            this_values = this_values[len(this_values)-top_n:len(this_values)]

        #percentage
        this_perc = np.array([i/sum(this_values) for i in this_values])

        stacked[i,:] = this_perc
        s_labels[i] = this_labels

    N = len(freq_vect)
    ind = np.arange(N)    # the x locations for the groups
    width = 1       # the width of the bars: can also be len(x) sequence

    # generate a colour pallete for each unique label
    unique_parcels = list(np.unique(s_labels))
    # order them by the most frequent
    weights = []
    for uparc in unique_parcels: # loop through parcels
        weights.append(stacked.transpose()[np.argwhere(np.array(s_labels) == uparc)].sum()) # append sum of values over all freq

    unique_parcels = [label for (yp, label) in sorted(zip(weights, unique_parcels), reverse=~rev_opt)]
    # get opnes to colour
    unique_parcels_colour = unique_parcels[0:color_n]
    unique_parcels_grey = unique_parcels[color_n:len(unique_parcels)]
    palette_c = sns.color_palette(None, len(unique_parcels_colour))
    palette_g = sns.color_palette('husl', len(unique_parcels_grey), desat=0)

    #generate a legend based on these colours
    legend_elements = [patches.Patch(facecolor=x, edgecolor=x, label=y) for x,y in zip(palette_c, unique_parcels_colour)]

    palette = palette_c + palette_g

    # frequency labels
    freq_labels = [f'{int(x)}Hz' for x in freq_vect]

    fig, ax = plt.subplots(figsize=(20, 6))
    plts = []
    bottoms = stacked[:,0]
    for i in range(top_n):
        # get colours for each bar
        these_labels = [x[i] for x in s_labels]
        these_inds = [unique_parcels.index(x) for x in these_labels]
        these_colours = np.array(palette)[these_inds]
        if i == 0:
            tmp_plt = ax.bar(freq_labels, stacked[:,i], width,
                             color=these_colours,
                             label=str(i))
        else:
            tmp_plt = ax.bar(freq_labels, stacked[:,i],width,
                             color=these_colours,
                              bottom=bottoms,
                             label=str(i))
            bottoms = bottoms + stacked[:,i]

    ax.legend(handles=legend_elements, bbox_to_anchor=(1, 1), loc='upper left')
    ax.set_title(f'{cust_name} Connections corellated with {model.contrast_names[contrast]} by Parcel')
    ax.set_xlabel('Frequency Band')
    ax.set_ylabel('Proportion of total outgoing connections')
    plt.autoscale()
    plt.savefig(join(figdir, f'freq_stacked_bar_{model.contrast_names[contrast]}_{cust_name}'))
    plt.close(fig)

#%% plot the different graphs

stacked_args = dict(plt_stat = copy.deepcopy(stats),top_n=68,color_n=6,rev_opt=False, freq_vect=freq_vect, label_names=label_names)

thislab = 'new99'

plot_stacked(**stacked_args, contrast=0, direction=1, cust_name=f'Outgoing_{thislab}')
plot_stacked(**stacked_args, contrast=0, direction=0, cust_name=f'Incoming_{thislab}')

plot_stacked(**stacked_args, contrast=1, direction=1, cust_name=f'Outgoing_{thislab}')
plot_stacked(**stacked_args, contrast=1, direction=0, cust_name=f'Incoming_{thislab}')

plot_stacked(**stacked_args, contrast=2, direction=1, cust_name=f'Outgoing_{thislab}')
plot_stacked(**stacked_args, contrast=2, direction=0, cust_name=f'Incoming_{thislab}')


#%% community detection in network X models
contrast = 1
k = 2
cli = []
for frequency in range(len(freq_vect)):
    meg_graph = nx.Graph(stats[contrast,:,:,frequency])
    cli.append(list(nx.algorithms.community.k_clique_communities(meg_graph,k)))
    #cli.append(nx.algorithms.rich_club_coefficient(meg_graph))
[len(i) for i in cli]

#%% simulated knock-out on global efficiency

def knock_out(graph):
    for i in range(len())

#%% have a look at the results for MRI
MR_glm_results = np.load('/imaging/ai05/RED/RED_MEG/resting/analysis/RED_Rest/MRI_GLM_RESULTS.npy')
MRstats = np.zeros([2,68,68,1])
MRstats[:,:,:,0] = MR_glm_results
stacked_args = dict(plt_stat = MRstats,top_n=68,color_n=6,rev_opt=False, freq_vect=[0], label_names=label_names)
plot_stacked(**stacked_args, contrast=0, direction=0, cust_name='MRI')
plot_stacked(**stacked_args, contrast=1, direction=0, cust_name='MRI')

#%% compare sparsity of intercepts

def calc_sparse(matrix):
    return np.sum(matrix!= 0) / np.prod(matrix.shape)

sp_MEG = calc_sparse(stats[0,:,:,7])
sp_MRI = calc_sparse(MRstats[0])

print(sp_MEG, sp_MRI)
#%% chord plot

mr_chord = red_plotting.hv_chord(1, 0,  0, MRstats, re_order_ind, label_names, model, freq_vect)
renderer.save(mr_chord, join(figdir, 'chord_MR_age'))
#%% plot ROI's from above on a brain


#%% compare Age's ROIs across MEG and IQ

in_common = np.intersect1d(unique_parcels[0:24], node_names)

#%% use network x to calculate thresholded connectome based on degree for each frequency
def get_graph_mets(graph):
    #make sure it's digraph
    graph = nx.DiGraph(graph)
    degree = graph.degree() # get node degree
    age_nodes = [f[0] for f in list(degree)] # get indices of nodes with more than two connections
    #get weighted degree for those nodes
    weighted_deg = np.array([list(graph.degree(weight='weight'))[i][1] for i in age_nodes])
    weighted_deg_out = np.array([list(graph.out_degree(weight='weight'))[i][1] for i in age_nodes])
    weighted_deg_in = np.array([list(graph.in_degree(weight='weight'))[i][1] for i in age_nodes])
    all_degrees = np.zeros([len(graph.nodes), 4])
    all_degrees[:, 0] = np.array([f[1] for f in list(degree)])
    all_degrees[:, 1] = weighted_deg
    all_degrees[:, 2] = weighted_deg_out
    all_degrees[:, 3] = weighted_deg_in

    return all_degrees

#%% Download MRI graphs
age_mri_graph = nx.read_gpickle('/imaging/ai05/RED/RED_MEG/resting/analysis/RED_Rest/age_graph.npy')
intercept_mri_graph = nx.read_gpickle('/imaging/ai05/RED/RED_MEG/resting/analysis/RED_Rest/intercept_graph.npy')

#%% Look at the MEG connectomes

alpha = .05/len(freq_vect)
for ii in range(len(freq_vect)):
    connec_thresh = 6
    contrast = 1
    metric = 0
    frequencies = ii

    if frequencies == 'all':
        MEG_deg = np.zeros([stats.shape[1], 4, len(freq_vect)])

        for i in range(len(freq_vect)):
            MEG_deg[:,:,i] = get_graph_mets(nx.from_numpy_matrix(stats[contrast,:,:,i]))

        meg_rank = MEG_deg[:,metric,:].mean(axis=1)
    else:
        meg_rank = get_graph_mets(nx.from_numpy_matrix(stats[contrast,:,:,frequencies]))[:, metric]

    # if degree take int from average
    if metric == 0:
        meg_rank = np.array([int(i) for i in meg_rank])
    #% compare that data with MRI
    mri_rank = get_graph_mets(age_mri_graph)[:,metric]

    mri_sorted_names = [label for (yp, label) in sorted(zip(mri_rank, label_names), reverse=True)]
    meg_sorted_names = [label for (yp, label) in sorted(zip(meg_rank, label_names), reverse=True)]

    in_common = np.intersect1d(mri_sorted_names, meg_sorted_names)
    results = ss.stats.spearmanr(mri_rank, meg_rank)
    print(results.pvalue <= alpha, ii, freq_vect[ii],results)

#%% look at that specific frequency identified a bit more
ii = 12
metric = 2
meg_graph = nx.from_numpy_matrix(stats[contrast,:,:,frequencies])
meg_rank = get_graph_mets(meg_graph)[:,metric]
mri_rank = get_graph_mets(age_mri_graph)[:,metric]
mri_sorted_names = [label for (yp, label) in sorted(zip(mri_rank, label_names), reverse=True)]
meg_sorted_names = [label for (yp, label) in sorted(zip(meg_rank, label_names), reverse=True)]
results = ss.stats.spearmanr(mri_rank[mri_rank < 15], meg_rank[mri_rank <15])
in_common = np.intersect1d(unique_parcels[0:10], node_names)

#%% plot the corellation
plt.scatter(mri_rank, meg_rank)
plt.savefig(join(figdir, 'MRI2MEG.png'))
plt.close('all')
#%% plot top MEG brain areas

for i in range(5):
    fig, ax = plot_roi([mri_sorted_names[i]])
    plt.savefig(join(figdir, f'{meg_sorted_names[i]}.png'))
    plt.close('all')


#%%

metric = 2

degree_mean = all_degrees[metric].mean(axis=1)

degree_sorted_values = sorted(degree_mean, reverse=~rev_opt)
degree_sorted_names = [label for (yp, label) in sorted(zip(degree_mean, label_names), reverse=~rev_opt)]

top_12 = degree_sorted_names[0:12]

MRIMEG_degree_common = np.intersect1d(top_12, node_names)

MEGMEG_degree_common = np.intersect1d(top_12, unique_parcels[0:12])
