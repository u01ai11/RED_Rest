#%% Import stuff
import pandas as pd
import sys
from os import listdir as listdir
from os.path import join
import numpy as np
sys.path.insert(0, '/imaging/ai05/RED/RED_MEG/resting/analysis/RED_Rest')
#sys.path.insert(0, '/home/ai05/Downloads/glm')
from REDTools import plotting
import glmtools
import networkx as nx
from scipy.io import loadmat
import matplotlib.pyplot as plt
import mne
#%% Paths and meta-deta
root_dir = '/imaging/ai05/RED/RED_MEG'
envcor_path = join(root_dir, 'aec_combined')
meta_dat_path = join(root_dir, 'Combined2.csv')
meta = pd.read_csv(meta_dat_path)
MAINDIR = join(root_dir, 'resting', 'STRUCTURALS')

# Calculate corellation matrix for each brain

tmp_id = meta.Alex_ID.to_list()# get IDS
tmp_id = [str(f).replace('_','-') for f in tmp_id] # replace to match filenames

betahigh =[f for f in listdir(envcor_path) if 'Upper Beta' in f] # find upper betas
nMEG_id = [i for i in tmp_id if i in [ii.split('_')[0] for ii in betahigh]]
betahigh = [betahigh[[i for i, s in enumerate(betahigh) if i_d in s][0]] for i_d in nMEG_id] # get them in order

# Read into one big matrix nparcels x nparcels x participants
mat_3d = np.zeros(shape=(len(betahigh), 68, 68, ))
for i, f in enumerate(betahigh):
    mat_3d[i,:,:] = np.load(join(envcor_path, f))

# load in stuctural connectomes
# read in matlab objects
str_path = join('/imaging', 'ai05', 'RED', 'RED_MEG', 'resting', 'Structural_connectomes')
red_struct = loadmat(join(str_path, 'red_data68.mat'))['stackdata68']
ace_struct = loadmat(join(str_path, 'amy_data68.mat'))['stackdata68']
# get labels for each column in the above
f= open(join(str_path, 'info.txt'), "r")
stack_labels =f.read().split("\n")[0:-1]
f.close()

# reorder to match participants above
# first values with summary (i.e. one value per parcel, per participant) [part,measure, parcel]
struct_sum = np.zeros((len(nMEG_id), (len([f for f in red_struct[0] if len(f.shape)<3])), 68))

# then whole connectomes per participant [part, measure, parcel_dim1, parcel_dim2]
struct_main = np.zeros((len(nMEG_id), len([f.shape for f in red_struct[0] if len(f.shape)>2]),68,68))

#isnan counter for participants!
is_data_nan = []
# loop through participants
#for i, row in meta.iterrows():
for i, megid in enumerate([f.replace('-', '_') for f in nMEG_id]):
    row = meta[meta.Alex_ID == megid]
    # what struct are we using
    if row.Study.array[0] == 1:
        this_str = ace_struct
    elif row.Study.array[0] == 2:
        this_str = red_struct
    # loop through cells and construct row for each sub structure
    tmp_sum = []
    tmp_main = []
    dan_ind = row.Dan_ID.array[0]
    if ~np.isnan(dan_ind):
        dan_ind = int(dan_ind)-1
        is_data_nan.append(True)
    else:
        is_data_nan.append(False)
    nan3 = np.empty((68,68))
    nan3[:] = np.nan
    struct_sum_labs = []
    struct_main_labs = []
    for cell, lab in zip(this_str[0], stack_labels):
        # parcel summary?
        if len(cell.shape) <3:
            if np.isnan(dan_ind):
                tmp_sum.append(np.array([np.nan]*68))
            else:
                tmp_sum.append(cell[dan_ind,:])
            struct_sum_labs.append(lab)
        # full connectomes
        elif len(cell.shape) >2:
            if np.isnan(dan_ind):
                tmp_main.append(nan3)
            else:
                tmp_main.append(cell[dan_ind,:,:])
            struct_main_labs.append(lab)
    # add these onto the main arrays
    struct_sum[i,:,:] = np.array(tmp_sum)
    struct_main[i,:,:] = np.array(tmp_main)

#reformat IDS
meta.Alex_ID = [str(f).replace('_', '-') for f in meta.Alex_ID]

#%% prep meta data for our regression


age = meta[meta.Alex_ID.isin(nMEG_id)][is_data_nan].Age
age[np.isnan(age)] = np.mean(age)

#%% prep the connectivity matrices
# use structural data, as data to predict
in_struct = struct_main[is_data_nan, 4, :,:]

#only use lower triangle of connectivity matrices
struct_lowT = np.array([x[np.tril_indices(x.shape[0], -1)] for x in in_struct])

#%% Now look at how our sample's cognitive variables vary with age



# load it into glmtools
dat = glmtools.data.TrialGLMData(data=struct_lowT, dim_labels=['participants', 'connections'])

# contrasts list
contrasts = []

# add regressor for intercept
regs = list()
regs.append(glmtools.regressors.ConstantRegressor(num_observations=dat.info['num_observations']))
contrasts.append(glmtools.design.Contrast(name='Intercept',values=[1,0])) # for regressor


# add regressor for Age
regs.append(glmtools.regressors.ParametricRegressor(values=age,
                                                    name='Age',
                                                    preproc='z',
                                                    num_observations=dat.info['num_observations']))
contrasts.append(glmtools.design.Contrast(name='Age',values=[0,1]))

# contrasts
des = glmtools.design.GLMDesign.initialise(regs,contrasts)
model = glmtools.fit.OLSModel( des, dat )

# unflatten t-stats
#make empty connectivity matrix for mask
tstats = np.empty(in_struct[0].shape)
tstats2 = np.empty(in_struct[0].shape)
#get indices of lower triangle
i_low = np.tril_indices(tstats.shape[0], -1)
# assign mask values
tstats[i_low] = model.tstats[1]
tstats2[i_low] = model.tstats[0]
# Copy lower triangle to upper triangle to make the matrix symmertical.
tstats.T[i_low] = tstats[i_low]
tstats2.T[i_low] = tstats2[i_low]

#%% permute
P = glmtools.permutations.Permutation(design=des, data=dat, contrast_idx=1, nperms=5000, metric='tstats', nprocesses=10)
thresh = P.get_thresh(95) #  Thresh is a 12x12 matrix
sig_simple = model.tstats[1,...] >= thresh

#%% permute intersepct
P2 = glmtools.permutations.Permutation(design=des, data=dat, contrast_idx=0, nperms=5000, metric='tstats', nprocesses=10)
thresh2 = P2.get_thresh(95) #  Thresh is a 12x12 matrix
sig_simple2 = model.tstats[0,...] >= thresh2
#%% recreate triangle and work out nodes/parcellations that have connections most linked with age

#make empty connectivity matrix for mask
mask = np.zeros(in_struct[0].shape, dtype=bool)
mask2 = np.zeros(in_struct[0].shape, dtype=bool)
# assign mask values
mask[i_low] = sig_simple
mask2[i_low] = sig_simple2
# Copy lower triangle to upper triangle to make the matrix symmertical.
mask.T[i_low] = mask[i_low]
mask2.T[i_low] = mask2[i_low]

#mask of our t values
age_glm_results = tstats
age_glm_results[[~mask]] = 0

intercept_glm_results = tstats2
intercept_glm_results[[~mask2]] = 0
#%% now get some degree info for these nodes
#read into netwokx
age_graph = nx.from_numpy_matrix(age_glm_results)
intercept_graph = nx.from_numpy_matrix(intercept_glm_results)
x_connec = 2 # threshold for no connections

degree = age_graph.degree() # get node degree
#age_nodes = [f[0] for f in list(degree) if f[1] > x_connec] # get indices of nodes with more than two connections
age_nodes = [f[0] for f in list(degree) if f[1]]
#get weighted degree for those nodes
weighted_deg = np.array([list(age_graph.degree(weight='weight'))[i][1] for i in age_nodes])

# get the labels
labels = mne.read_labels_from_annot('fsaverage_1', parc='aparc', subjects_dir=join(MAINDIR,'FS_SUBDIR'))
labels = labels[0:-1]
label_names = [label.name for label in labels]


age_node_names = [label_names[i] for i in age_nodes] # names
age_node_values = [list(degree)[i] for i in age_nodes]

#%% save results

np.save('/imaging/ai05/RED/RED_MEG/resting/analysis/RED_Rest/age_nodes.npy', age_nodes)
np.save('/imaging/ai05/RED/RED_MEG/resting/analysis/RED_Rest/age_node_names.npy', age_node_names)
np.save('/imaging/ai05/RED/RED_MEG/resting/analysis/RED_Rest/age_node_vals.npy', age_node_values)
nx.write_gpickle(age_graph, '/imaging/ai05/RED/RED_MEG/resting/analysis/RED_Rest/age_graph.npy')
nx.write_gpickle(intercept_graph, '/imaging/ai05/RED/RED_MEG/resting/analysis/RED_Rest/intercept_graph.npy')
#%% plot results to determine areas

plt.close('all')
fig, axs = plt.subplots(nrows=1, ncols=2) # two plots
# plot the brain
# fake connectivity matrix with one connection which is the weighted degree for that node
fake_connect = np.zeros((len(labels), len(labels))) # intialise empty
for ind in range(len(age_nodes)):
    fake_connect[age_nodes[ind],age_nodes[ind]] = weighted_deg[ind]

plotting.plot_aparc_parcels(fake_connect, axs[0], fig, 'Age-related parcels')
top_labels = np.array(age_node_names)
top_vals = weighted_deg
top_vals = top_vals.round(3).astype(str)
to_table = [top_vals.tolist(), top_labels.tolist()]
rotated = list(zip(*to_table[::-1]))
table = axs[1].table(rotated, loc=9)
table.auto_set_font_size(False)
table.set_fontsize(12)

fig.set_size_inches(17.0, 6)
fig.savefig('/imaging/ai05/images/age_related_areas.png')

#%% how do these specific areas relate to cognitive and behavioural measures?

