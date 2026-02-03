import sys
import numpy as np
import pandas as pd
from time import time, time_ns
import os

import pgmpy
from AicScore import AicScore
from pgmpy.estimators import HillClimbSearch
from EfficientTimeShuffling import EfficientShuffle
import random

sys.path.append('/home/ags72/Documents/MTWDBN/Tools')
from GraphFunctions import graph_to_matrix_6pop_3timeslice, generate_starting_graph

bootstrap = int(sys.argv[1])
iteration = int(sys.argv[2])
shuffle = sys.argv[3] in ['True', 'true']
drop = int(sys.argv[4])

#Check if this search has been performed previously. If so, abort.
if shuffle:
    if os.path.isfile('DBN Outputs/DAGs/DAGs_drop{drop}_bootstrap{bootstrap}_iteration{iteration}_shuffle.npy'.format(drop = drop, bootstrap = bootstrap, iteration = iteration)):
        sys.exit(0)
else:
    if os.path.isfile('DBN Outputs/DAGs/DAGs_drop{drop}_bootstrap{bootstrap}_iteration{iteration}.npy'.format(drop = drop, bootstrap = bootstrap, iteration = iteration)):
        sys.exit(0)

#Hard coded variables
timelags = 3
num_starting_points = 120

#Load spike table
spikes_df_all = pd.read_csv('Alec Dataframes/laminar2_spikes_drop_{drop}_iteration_{iteration}.csv'.format(drop=drop, iteration = iteration), index_col = 0)

#Sample data for the bootstrap, set seed equal to bootstrap
spikes_df = spikes_df_all.sample(8000,replace = True, random_state = bootstrap)

#Shuffle data after already selecting bootstrap
if shuffle:
    random.seed(round(time_ns()))
    spikes_df, df_shufflekeys = EfficientShuffle(spikes_df,seed = random.randint(0,2**32-1))

#Arrays for storing starting points, resulting DAGs, and scores
DAGs = np.zeros((num_starting_points,18,6))
starting_graphs = np.zeros((num_starting_points,18,6))
scores = np.zeros((num_starting_points))

all_pops = list(spikes_df_all.columns)
from_pops = [pop for pop in all_pops if not pop.endswith('t{}'.format(timelags))]
to_pops = [pop for pop in all_pops if pop.endswith('t{}'.format(timelags))]

# causal_whitelist = [(from_pop,to_pop) for from_pop in from_pops for to_pop in to_pops]
acausal_whitelist = [(from_pop, to_pop) for from_pop in all_pops for to_pop in to_pops] #Allow edges in the last time slice

#Scoring criteria
aic = AicScore(spikes_df)

for starting_point in range(num_starting_points):

    #Create Random Starting Point (seed not controlled to increase randomness, but starting graph is saved)
    random.seed(round(time_ns()))
    np.random.seed(random.randint(0,2**32-1))
    starting_graph = generate_starting_graph(nodes = all_pops, whitelist = acausal_whitelist, max_degree = 4, last_time_nodes = to_pops)
    starting_graphs[starting_point,:,:] = graph_to_matrix_6pop_3timeslice(list(starting_graph.edges())) #Save the starting graph now, otherwise it appears that pgmpy modifies it in place to resemble the final graph

    #Perform hill search for DAG
    hc = HillClimbSearch(spikes_df)
    model = hc.estimate(tabu_length= 7, max_indegree=None, white_list = acausal_whitelist, scoring_method = AicScore(spikes_df), start_dag = starting_graph)

    #Save resulting DAG and score
    DAGs[starting_point,:,:] = graph_to_matrix_6pop_3timeslice(list(model.edges()))
    scores[starting_point] = aic.score(model)

#Save all results
if shuffle:
    np.save('DBN Outputs/DAGs/DAGs_drop{drop}_bootstrap{bootstrap}_iteration{iteration}_shuffle.npy'.format(drop = drop, bootstrap = bootstrap, iteration = iteration), DAGs)
    np.save('DBN Outputs/Starting Graphs/startGraphs_drop{drop}_bootstrap{bootstrap}_iteration{iteration}_shuffle.npy'.format(drop = drop, bootstrap = bootstrap, iteration = iteration), starting_graphs)
    np.save('DBN Outputs/Scores/scores_drop{drop}_bootstrap{bootstrap}_iteration{iteration}_shuffle.npy'.format(drop = drop, bootstrap = bootstrap, iteration = iteration), scores)

else:
    np.save('DBN Outputs/DAGs/DAGs_drop{drop}_bootstrap{bootstrap}_iteration{iteration}.npy'.format(drop = drop, bootstrap = bootstrap, iteration = iteration), DAGs)
    np.save('DBN Outputs/Starting Graphs/startGraphs_drop{drop}_bootstrap{bootstrap}_iteration{iteration}.npy'.format(drop = drop, bootstrap = bootstrap, iteration = iteration), starting_graphs)
    np.save('DBN Outputs/Scores/scores_drop{drop}_bootstrap{bootstrap}_iteration{iteration}.npy'.format(drop = drop, bootstrap = bootstrap, iteration = iteration), scores)

   