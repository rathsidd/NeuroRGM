#%% Imports
import pandas as pd
import numpy as np

#import neuprint as npr
from neuprint import Client, fetch_traced_adjacencies, fetch_adjacencies
from neuprint import fetch_synapse_connections
from neuprint import fetch_synapse_connections, NeuronCriteria as NC, SynapseCriteria as SC
from neuprint.utils import connection_table_to_matrix,merge_neuron_properties

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt

import seaborn as sns;

from scipy.spatial.distance import euclidean
import os

root = "C:\\Users\\knity\\Documents\\Nitya\\School\\RESEARCH\\GEMSEC_Projects\\NeuroRGM"
os.chdir(root)
#%% API Connection

TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6Im5pdHlha0B1dy5lZHUiLCJsZXZlbCI6Im5vYXV0aCIsImltYWdlLXVybCI6Imh0dHBzOi8vbGg2Lmdvb2dsZXVzZXJjb250ZW50LmNvbS8tcV9pZTRrcTBCQU0vQUFBQUFBQUFBQUkvQUFBQUFBQUFBQUEvQUFLV0pKTXo1dFBQQnY1ME0xOWRmcEg4TmNaNXlrNEVtZy9waG90by5qcGc_c3o9NTA_c3o9NTAiLCJleHAiOjE3NjY1MzYxNzB9.gNTqVgZNHTfIoGkfwlFidgzOkjWxwAJqNyY3cSpaw0M"

c = Client('neuprint.janelia.org', 'hemibrain:v1.0.1', TOKEN)

#%% Adjacency
#neurons_df, roi_conn_df = npr.queries.fetch_adjacencies(None, None, 200, 'Neuron', None)


sources = [329566174, 425790257, 424379864, 329599710]
targets = [425790257, 424379864, 329566174, 329599710, 420274150]
neuron_df, connection_df = fetch_adjacencies(sources, targets)

#results = npr.fetch_simple_connections(upstream_bodyId = 1224941044)

#%% Create plot for one neuron

def plot_neuron(s, x=[], y=[], z=[]):

    # bodyId = '917669951' #Inputs and outputs in AB(L) and FB
    #s = c.fetch_skeleton(bodyId, format='pandas')
    X, Y, Z = np.array(s['x']), np.array(s['y']), np.array(s['z'])
    rad = np.array(s['radius'])
    
    fig = plt.figure(figsize=(20,15))
    ax = fig.gca(projection='3d')
    ax.scatter(X, Y, Z, s=rad, c = rad, cmap = plt.cm.seismic, linewidths = rad*0.02, alpha = 0.5)

    #plt.savefig(f'./Plots/skeleton_neuron{bodyId}_withRad.png')  
#s = c.fetch_skeleton('917669951', format='pandas')
#plot_neuron(s, np.array(s['x']), np.array(s['y']), np.array(s['z'])

#%% Distance between 2 neurons - get min Dists

### try pairwise distances

bodyId1 = '917669951'
bodyId2 = '917674256'

s1 = c.fetch_skeleton(bodyId1, format='pandas')
s2 = c.fetch_skeleton(bodyId2, format='pandas')
minDists = pd.DataFrame(columns = ['p1', 'p2', 'dist'])
for index, row in s1.iterrows():
    p1 = [row.x, row.y, row.z]
    minDist = float("inf")
    minp2 = p1
    
    for index2, row2 in s2.iterrows():
        p2 = [row2.x, row2.y, row2.z]
        dist = euclidean(p1, p2)
        
        if dist < minDist:
            minp2 = p2
            minDist = dist
            
    minDists = minDists.append({'p1':p1, 'p2':minp2,'dist':minDist}, ignore_index=True)

#%% Distance between 2 neurons - plot
p1_X = [item[0] for item in minDists['p1']]
p1_Y = [item[1] for item in minDists['p1']]
p1_Z = [item[2] for item in minDists['p1']]

p2_X = [item[0] for item in minDists['p2']]
p2_Y = [item[1] for item in minDists['p2']]
p2_Z = [item[2] for item in minDists['p2']]

X1, Y1, Z1 = np.array(s1['x']), np.array(s1['y']), np.array(s1['z'])
rad1 = np.array(s1['radius'])
X2, Y2, Z2 = np.array(s2['x']), np.array(s2['y']), np.array(s2['z'])
rad2 = np.array(s2['radius'])

fig = plt.figure(figsize=(20,15))
ax = fig.gca(projection='3d')
ax.scatter(X1, Y1, Z1, s = rad1, c = rad1, cmap = plt.cm.seismic, linewidths = rad1*0.02, alpha = 0.5)
ax.scatter(X2, Y2, Z2, s = rad2, c = rad2, cmap = plt.cm.seismic, linewidths = rad2*0.02, alpha = 0.5)
ax.scatter(p1_X, p1_Y, p1_Z, s = 2, cmap = 'green')

#%% Plot distances - histogram
plt.hist(minDists['dist'], bins = 20, 
         color = '#0504aa', alpha=0.7, rwidth = 0.9)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Minimum distances')
plt.ylabel('Frequency')
plt.title('Minimum euclidean distances from neuron 1 to neuron 2')
plt.savefig(f'./Plots/minDists_neuron{bodyId1}_neuron{bodyId2}.png') 

#%% Adjacency
# neurons_df, roi_conn_df = fetch_traced_adjacencies('exported-connections')
# conn_df = merge_neuron_properties(neurons_df, roi_conn_df, ['type', 'instance'])
# conn_mat = connection_table_to_matrix(conn_df, 'bodyId', sort_by='type')

neuron_criteria = NC(status='Traced', cropped=False)
eb_syn_criteria = SC(primary_only=False)
eb_conns = fetch_synapse_connections(neuron_criteria, None, eb_syn_criteria)

# plt.matshow(conn_mat)
# plt.title('Adjacencies - traced neurons')
# plt.savefig(f'./Plots/adj_traced.png') 
