# function - train the dynamic bayesian network - Alec's methods

########################################
def graph_to_matrix(edges,nevents,eventnames):

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy
    import string
    import warnings
    import pickle

    from pgmpy.models import BayesianModel
    from pgmpy.models import DynamicBayesianNetwork as DBN
    from pgmpy.estimators import BayesianEstimator
    from pgmpy.estimators import HillClimbSearch,BicScore
    import networkx as nx

    output_matrix = np.zeros((nevents,nevents)) 
    
    row = 0
    for from_layer in np.arange(0,nevents,1):
        column = 0
        #Loop through the receiving nodes (the last timeslice of each population)
        for to_layer in np.arange(0,nevents,1): 
            from_pop = eventnames[from_layer]+'_t0'
            to_pop = eventnames[to_layer]+'_t1'

            if (from_pop, to_pop) in edges:
                output_matrix[row,column] = 1
            else:
                output_matrix[row,column] = 0
            column+=1
        row+=1
                    
    return output_matrix



########################################
def get_weighted_dags(binaryDags,nNewBootstraps = 100, seed=1):

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy
    import string
    import warnings
    import pickle

    from pgmpy.models import BayesianModel
    from pgmpy.models import DynamicBayesianNetwork as DBN
    from pgmpy.estimators import BayesianEstimator
    from pgmpy.estimators import HillClimbSearch,BicScore
    import networkx as nx

    ### Step 1: Create Bootstraps of "discrete" DAGs for weighted DAGs
    np.random.seed(seed)
    [nTrials,frNodes,toNodes] = binaryDags.shape
    
    bootstrap_graphs = np.zeros([nNewBootstraps,nTrials,frNodes,toNodes])
    for iBootstrap in range(nNewBootstraps):
        bootstrap_graphs[iBootstrap,:,:,:] = binaryDags[np.random.randint(nTrials, size=(nTrials)),:,:]

    ### Step 2: Get Weighted DAGs
    wtd_graphs = np.nanmean(bootstrap_graphs, axis=1)
    
    return wtd_graphs



########################################
def get_significant_edges(weighted_graphs, shuffled_weighted_graphs):    

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy
    import string
    import warnings
    import pickle

    from scipy import stats
    from statsmodels.stats.proportion import proportions_ztest

    from pgmpy.models import BayesianModel
    from pgmpy.models import DynamicBayesianNetwork as DBN
    from pgmpy.estimators import BayesianEstimator
    from pgmpy.estimators import HillClimbSearch,BicScore
    import networkx as nx

    [nBootstraps,FrNode,ToNode] = weighted_graphs.shape
    sig_edges = np.zeros((FrNode,ToNode))
        
    counter=0
    for i in range(FrNode):
        for j in range(ToNode):
            edges = weighted_graphs[:,i,j].flatten() #Get all weights for this edge from all bootstraps

            shuffled_edges = shuffled_weighted_graphs[:,i,j].flatten()

            stat,p_value = stats.mannwhitneyu(edges,shuffled_edges,alternative = 'greater')
            # stat,p_value = proportions_ztest(count = [np.count_nonzero(edges), np.count_nonzero(shuffled_edges)],nobs = [len(edges), len(shuffled_edges)],alternative = 'larger')


            if (p_value) < 0.05:
                sig_edges[i,j] = 1
            
    return sig_edges

########################################
def threshold_edges(weighted_graphs, threshold=0.5):

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy
    import string
    import warnings
    import pickle

    from pgmpy.models import BayesianModel
    from pgmpy.models import DynamicBayesianNetwork as DBN
    from pgmpy.estimators import BayesianEstimator
    from pgmpy.estimators import HillClimbSearch,BicScore
    import networkx as nx

    mean_graph = weighted_graphs.mean(axis = 0)
    
    return mean_graph >= threshold


########################################
def train_DBN_alec_create_df_only(totalsess_time, session_start_time, temp_resolu, time_point_pull1, time_point_pull2, oneway_gaze1, oneway_gaze2, mutual_gaze1, mutual_gaze2):
# temp_resolu: temporal resolution, the time different between each step
# e.g. temp_resolu = 0.5s, each step is 0.5s
# totalsess_time: the time (s) of the total session

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy
    import string
    import warnings
    import pickle

    
    # optional - conbine mutual gaze and one way gaze
    oneway_gaze1 = np.sort(np.concatenate((oneway_gaze1,mutual_gaze1)))
    oneway_gaze2 = np.sort(np.concatenate((oneway_gaze2,mutual_gaze2)))
    
    total_time = int(np.floor((totalsess_time - session_start_time)/temp_resolu))
    time_point_pull1_round = np.floor(time_point_pull1/temp_resolu).reset_index(drop = True).astype(int)
    time_point_pull1_round = time_point_pull1_round[time_point_pull1_round<total_time]
    time_point_pull2_round  = np.floor(time_point_pull2/temp_resolu).reset_index(drop = True).astype(int)
    time_point_pull2_round = time_point_pull2_round[time_point_pull2_round<total_time]
    time_point_onewaygaze1_round = np.floor(pd.Series(oneway_gaze1)/temp_resolu).reset_index(drop = True).astype(int)
    time_point_onewaygaze2_round = np.floor(pd.Series(oneway_gaze2)/temp_resolu).reset_index(drop = True).astype(int)
    time_point_mutualgaze1_round = np.floor(pd.Series(mutual_gaze1)/temp_resolu).reset_index(drop = True).astype(int)
    time_point_mutualgaze2_round = np.floor(pd.Series(mutual_gaze2)/temp_resolu).reset_index(drop = True).astype(int)
    time_point_onewaygaze1_round = time_point_onewaygaze1_round[(time_point_onewaygaze1_round>0)&(time_point_onewaygaze1_round<total_time)]
    time_point_onewaygaze2_round = time_point_onewaygaze2_round[(time_point_onewaygaze2_round>0)&(time_point_onewaygaze2_round<total_time)]
    time_point_mutualgaze1_round = time_point_mutualgaze1_round[(time_point_mutualgaze1_round>0)&(time_point_mutualgaze1_round<total_time)]
    time_point_mutualgaze2_round = time_point_mutualgaze2_round[(time_point_mutualgaze2_round>0)&(time_point_mutualgaze2_round<total_time)]
    # t1 - current
    pull1_t1 = np.zeros((total_time+1,1))
    pull1_t1[np.array(time_point_pull1_round)] = 1
    pull2_t1 = np.zeros((total_time+1,1))
    pull2_t1[np.array(time_point_pull2_round)] = 1
    owgaze1_t1 = np.zeros((total_time+1,1))
    owgaze1_t1[np.array(time_point_onewaygaze1_round)] = 1
    owgaze2_t1 = np.zeros((total_time+1,1))
    owgaze2_t1[np.array(time_point_onewaygaze2_round)] = 1
    mtgaze1_t1 = np.zeros((total_time+1,1))
    mtgaze1_t1[np.array(time_point_mutualgaze1_round)] = 1
    mtgaze2_t1 = np.zeros((total_time+1,1))
    mtgaze2_t1[np.array(time_point_mutualgaze2_round)] = 1
    # t0 - last step
    pull1_t0 = np.zeros((total_time+1,1))
    pull1_t0[np.array(time_point_pull1_round)+1] = 1
    pull2_t0 = np.zeros((total_time+1,1))
    pull2_t0[np.array(time_point_pull2_round)+1] = 1
    owgaze1_t0 = np.zeros((total_time+1,1))
    owgaze1_t0[np.array(time_point_onewaygaze1_round)+1] = 1
    owgaze2_t0 = np.zeros((total_time+1,1))
    owgaze2_t0[np.array(time_point_onewaygaze2_round)+1] = 1
    mtgaze1_t0 = np.zeros((total_time+1,1))
    mtgaze1_t0[np.array(time_point_mutualgaze1_round)+1] = 1
    mtgaze2_t0 = np.zeros((total_time+1,1))
    mtgaze2_t0[np.array(time_point_mutualgaze2_round)+1] = 1
    
    # clean up dataframe
    data = np.concatenate((pull1_t0,pull2_t0,owgaze1_t0,owgaze2_t0,pull1_t1,pull2_t1,owgaze1_t1,owgaze2_t1),axis = 1)
    colnames = ["pull1_t0","pull2_t0","owgaze1_t0","owgaze2_t0","pull1_t1","pull2_t1","owgaze1_t1","owgaze2_t1"]
    eventnames = ["pull1","pull2","owgaze1","owgaze2"]
    nevents = np.size(eventnames)
    bhv_df = pd.DataFrame(data, columns=colnames)

    return bhv_df 



########################################
def train_DBN_alec_training_only(bhv_df,starting_graph):

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy
    import string
    import warnings
    import pickle

    from pgmpy.models import BayesianModel
    from pgmpy.models import DynamicBayesianNetwork as DBN
    from pgmpy.estimators import BayesianEstimator

    from pgmpy.estimators import HillClimbSearch,BicScore
    import networkx as nx

    from AicScore import AicScore

    from ana_functions.train_DBN_alec import get_weighted_dags
    from ana_functions.train_DBN_alec import graph_to_matrix
    
    colnames = ["pull1_t0","pull2_t0","owgaze1_t0","owgaze2_t0","pull1_t1","pull2_t1","owgaze1_t1","owgaze2_t1"]
    eventnames = ["pull1","pull2","owgaze1","owgaze2"]
    nevents = np.size(eventnames)    

    # define DBN structures
    all_pops = list(bhv_df.columns)
    from_pops = [pop for pop in all_pops if not pop.endswith('t1')]
    to_pops = [pop for pop in all_pops if pop.endswith('t1')]
    causal_whitelist = [(from_pop,to_pop) for from_pop in from_pops for to_pop in to_pops]

    # train the DBN for the edges
    bhv_hc = HillClimbSearch(bhv_df)
    best_model = bhv_hc.estimate(max_indegree=None, white_list=causal_whitelist, scoring_method=AicScore(bhv_df), start_dag=starting_graph)
    edges = best_model.edges()
    
    nFromNodes = nevents
    nToNodes = nevents
    nTrials = 1
    DAGs = np.zeros((nTrials, nFromNodes, nToNodes))
    DAGs[0,:,:] = graph_to_matrix(list(edges),nevents,eventnames)

    return best_model, edges, DAGs, eventnames, from_pops, to_pops



########################################
def train_DBN_alec(totalsess_time, session_start_time, temp_resolu, time_point_pull1, time_point_pull2, oneway_gaze1, oneway_gaze2, mutual_gaze1, mutual_gaze2):
# temp_resolu: temporal resolution, the time different between each step
# e.g. temp_resolu = 0.5s, each step is 0.5s
# totalsess_time: the time (s) of the total session

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy
    import string
    import warnings
    import pickle

    from pgmpy.models import BayesianModel
    from pgmpy.models import DynamicBayesianNetwork as DBN
    from pgmpy.estimators import BayesianEstimator
    from pgmpy.estimators import HillClimbSearch,BicScore
    import networkx as nx

    from AicScore import AicScore
    
    from ana_functions.train_DBN_alec import get_weighted_dags
    from ana_functions.train_DBN_alec import graph_to_matrix
    
    # optional - conbine mutual gaze and one way gaze
    oneway_gaze1 = np.sort(np.concatenate((oneway_gaze1,mutual_gaze1)))
    oneway_gaze2 = np.sort(np.concatenate((oneway_gaze2,mutual_gaze2)))
    
    total_time = int(np.floor((totalsess_time - session_start_time)/temp_resolu))
    time_point_pull1_round = np.floor(time_point_pull1/temp_resolu).reset_index(drop = True).astype(int)
    time_point_pull1_round = time_point_pull1_round[time_point_pull1_round<total_time]
    time_point_pull2_round  = np.floor(time_point_pull2/temp_resolu).reset_index(drop = True).astype(int)
    time_point_pull2_round = time_point_pull2_round[time_point_pull2_round<total_time]
    time_point_onewaygaze1_round = np.floor(pd.Series(oneway_gaze1)/temp_resolu).reset_index(drop = True).astype(int)
    time_point_onewaygaze2_round = np.floor(pd.Series(oneway_gaze2)/temp_resolu).reset_index(drop = True).astype(int)
    time_point_mutualgaze1_round = np.floor(pd.Series(mutual_gaze1)/temp_resolu).reset_index(drop = True).astype(int)
    time_point_mutualgaze2_round = np.floor(pd.Series(mutual_gaze2)/temp_resolu).reset_index(drop = True).astype(int)
    time_point_onewaygaze1_round = time_point_onewaygaze1_round[(time_point_onewaygaze1_round>0)&(time_point_onewaygaze1_round<total_time)]
    time_point_onewaygaze2_round = time_point_onewaygaze2_round[(time_point_onewaygaze2_round>0)&(time_point_onewaygaze2_round<total_time)]
    time_point_mutualgaze1_round = time_point_mutualgaze1_round[(time_point_mutualgaze1_round>0)&(time_point_mutualgaze1_round<total_time)]
    time_point_mutualgaze2_round = time_point_mutualgaze2_round[(time_point_mutualgaze2_round>0)&(time_point_mutualgaze2_round<total_time)]
    # t1
    pull1_t1 = np.zeros((total_time+1,1))
    pull1_t1[np.array(time_point_pull1_round)] = 1
    pull2_t1 = np.zeros((total_time+1,1))
    pull2_t1[np.array(time_point_pull2_round)] = 1
    owgaze1_t1 = np.zeros((total_time+1,1))
    owgaze1_t1[np.array(time_point_onewaygaze1_round)] = 1
    owgaze2_t1 = np.zeros((total_time+1,1))
    owgaze2_t1[np.array(time_point_onewaygaze2_round)] = 1
    mtgaze1_t1 = np.zeros((total_time+1,1))
    mtgaze1_t1[np.array(time_point_mutualgaze1_round)] = 1
    mtgaze2_t1 = np.zeros((total_time+1,1))
    mtgaze2_t1[np.array(time_point_mutualgaze2_round)] = 1
    # t0
    pull1_t0 = np.zeros((total_time+1,1))
    pull1_t0[np.array(time_point_pull1_round)+1] = 1
    pull2_t0 = np.zeros((total_time+1,1))
    pull2_t0[np.array(time_point_pull2_round)+1] = 1
    owgaze1_t0 = np.zeros((total_time+1,1))
    owgaze1_t0[np.array(time_point_onewaygaze1_round)+1] = 1
    owgaze2_t0 = np.zeros((total_time+1,1))
    owgaze2_t0[np.array(time_point_onewaygaze2_round)+1] = 1
    mtgaze1_t0 = np.zeros((total_time+1,1))
    mtgaze1_t0[np.array(time_point_mutualgaze1_round)+1] = 1
    mtgaze2_t0 = np.zeros((total_time+1,1))
    mtgaze2_t0[np.array(time_point_mutualgaze2_round)+1] = 1
    
    # clean up dataframe
    data = np.concatenate((pull1_t0,pull2_t0,owgaze1_t0,owgaze2_t0,pull1_t1,pull2_t1,owgaze1_t1,owgaze2_t1),axis = 1)
    colnames = ["pull1_t0","pull2_t0","owgaze1_t0","owgaze2_t0","pull1_t1","pull2_t1","owgaze1_t1","owgaze2_t1"]
    eventnames = ["pull1","pull2","owgaze1","owgaze2"]
    nevents = np.size(eventnames)
    bhv_df = pd.DataFrame(data, columns=colnames)

    # define DBN structures
    all_pops = list(bhv_df.columns)
    from_pops = [pop for pop in all_pops if not pop.endswith('t1')]
    to_pops = [pop for pop in all_pops if pop.endswith('t1')]
    causal_whitelist = [(from_pop,to_pop) for from_pop in from_pops for to_pop in to_pops]

    # train the DBN for the edges
    bhv_hc = HillClimbSearch(bhv_df)
    best_model = bhv_hc.estimate(max_indegree=None, white_list = causal_whitelist, scoring_method=AicScore(bhv_df))
    edges = best_model.edges()
    
    nFromNodes = nevents
    nToNodes = nevents
    nTrials = 1
    DAGs = np.zeros((nTrials, nFromNodes, nToNodes))
    DAGs[0,:,:] = graph_to_matrix(list(edges),nevents,eventnames)

    return best_model, edges, DAGs, eventnames, from_pops, to_pops






########################################
def train_DBN_alec_eachtrial(totalsess_time, session_start_time, temp_resolu, time_point_pull1, time_point_pull2, oneway_gaze1, oneway_gaze2, mutual_gaze1, mutual_gaze2):
# temp_resolu: temporal resolution, the time different between each step
# e.g. temp_resolu = 0.5s, each step is 0.5s
# totalsess_time: the time (s) of the total session

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy
    import string
    import warnings
    import pickle

    from pgmpy.models import BayesianModel
    from pgmpy.models import DynamicBayesianNetwork as DBN
    from pgmpy.estimators import BayesianEstimator
    from pgmpy.estimators import HillClimbSearch,BicScore
    import networkx as nx
    
    from AicScore import AicScore

    from ana_functions.train_DBN_alec import get_weighted_dags
    from ana_functions.train_DBN_alec import graph_to_matrix
    
    # optional - conbine mutual gaze and one way gaze
    oneway_gaze1 = np.sort(np.concatenate((oneway_gaze1,mutual_gaze1)))
    oneway_gaze2 = np.sort(np.concatenate((oneway_gaze2,mutual_gaze2)))
    
    total_time = int((totalsess_time - session_start_time)/temp_resolu)
    time_point_pull1_round = np.floor(time_point_pull1/temp_resolu).reset_index(drop = True).astype(int)
    time_point_pull1_round = time_point_pull1_round[time_point_pull1_round<total_time]
    time_point_pull2_round  = np.floor(time_point_pull2/temp_resolu).reset_index(drop = True).astype(int)
    time_point_pull2_round = time_point_pull2_round[time_point_pull2_round<total_time]
    time_point_onewaygaze1_round = np.floor(pd.Series(oneway_gaze1)/temp_resolu).reset_index(drop = True).astype(int)
    time_point_onewaygaze2_round = np.floor(pd.Series(oneway_gaze2)/temp_resolu).reset_index(drop = True).astype(int)
    time_point_mutualgaze1_round = np.floor(pd.Series(mutual_gaze1)/temp_resolu).reset_index(drop = True).astype(int)
    time_point_mutualgaze2_round = np.floor(pd.Series(mutual_gaze2)/temp_resolu).reset_index(drop = True).astype(int)
    time_point_onewaygaze1_round = time_point_onewaygaze1_round[(time_point_onewaygaze1_round>0)&(time_point_onewaygaze1_round<total_time)]
    time_point_onewaygaze2_round = time_point_onewaygaze2_round[(time_point_onewaygaze2_round>0)&(time_point_onewaygaze2_round<total_time)]
    time_point_mutualgaze1_round = time_point_mutualgaze1_round[(time_point_mutualgaze1_round>0)&(time_point_mutualgaze1_round<total_time)]
    time_point_mutualgaze2_round = time_point_mutualgaze2_round[(time_point_mutualgaze2_round>0)&(time_point_mutualgaze2_round<total_time)]
    # t1
    pull1_t1 = np.zeros((total_time+1,1))
    pull1_t1[np.array(time_point_pull1_round)] = 1
    pull2_t1 = np.zeros((total_time+1,1))
    pull2_t1[np.array(time_point_pull2_round)] = 1
    owgaze1_t1 = np.zeros((total_time+1,1))
    owgaze1_t1[np.array(time_point_onewaygaze1_round)] = 1
    owgaze2_t1 = np.zeros((total_time+1,1))
    owgaze2_t1[np.array(time_point_onewaygaze2_round)] = 1
    mtgaze1_t1 = np.zeros((total_time+1,1))
    mtgaze1_t1[np.array(time_point_mutualgaze1_round)] = 1
    mtgaze2_t1 = np.zeros((total_time+1,1))
    mtgaze2_t1[np.array(time_point_mutualgaze2_round)] = 1
    
    # t0
    pull1_t0 = np.zeros((total_time+1,1))
    pull1_t0[np.array(time_point_pull1_round)+1] = 1
    pull2_t0 = np.zeros((total_time+1,1))
    pull2_t0[np.array(time_point_pull2_round)+1] = 1
    owgaze1_t0 = np.zeros((total_time+1,1))
    owgaze1_t0[np.array(time_point_onewaygaze1_round)+1] = 1
    owgaze2_t0 = np.zeros((total_time+1,1))
    owgaze2_t0[np.array(time_point_onewaygaze2_round)+1] = 1
    mtgaze1_t0 = np.zeros((total_time+1,1))
    mtgaze1_t0[np.array(time_point_mutualgaze1_round)+1] = 1
    mtgaze2_t0 = np.zeros((total_time+1,1))
    mtgaze2_t0[np.array(time_point_mutualgaze2_round)+1] = 1
    
    # clean up dataframe
    data = np.concatenate((pull1_t0,pull2_t0,owgaze1_t0,owgaze2_t0,pull1_t1,pull2_t1,owgaze1_t1,owgaze2_t1),axis = 1)
    colnames = ["pull1_t0","pull2_t0","owgaze1_t0","owgaze2_t0","pull1_t1","pull2_t1","owgaze1_t1","owgaze2_t1"]
    eventnames = ["pull1","pull2","owgaze1","owgaze2"]
    nevents = np.size(eventnames)
    bhv_df = pd.DataFrame(data, columns=colnames)
    
    # every 10s (20 0.5s steps) as a "trial"
    # total 720s "session" will be 72 "trials"
    ntrials = int(np.floor(totalsess_time/10))
    nFromNodes = nevents
    nToNodes = nevents
    for itrial in np.arange(0,ntrials,1):
        bhv_df_itrial = bhv_df.iloc[itrial*(10/temp_resolu):(itrial+1)*(10/temp_resolu)]
        # define DBN structures
        all_pops = list(bhv_df_itrial.columns)
        from_pops = [pop for pop in all_pops if not pop.endswith('t1')]
        to_pops = [pop for pop in all_pops if pop.endswith('t1')]
        causal_whitelist = [(from_pop,to_pop) for from_pop in from_pops for to_pop in to_pops]
    
        # train the DBN for the edges
        bhv_hc = HillClimbSearch(bhv_df_itrial)
        best_model = bhv_hc.estimate(max_indegree=None, white_list = causal_whitelist, scoring_method=AicScore(bhv_df))
        edges_itrial = best_model.edges()   
    
        DAGs_itrial = np.zeros((ntrials, nFromNodes, nToNodes))
        DAGs_itrial[0,:,:] = graph_to_matrix(list(edges_itrial),nevents,eventnames)

    weighted_graphs = get_weighted_dags(DAGs_itrial,nNewBootstraps = 1)
    return DAGs_itrial, weighted_graphs, eventnames, from_pops, to_pops







########################################
def train_DBN_gaze_start_stop(gaze_thresold, totalsess_time, session_start_time, temp_resolu, time_point_pull1, time_point_pull2, oneway_gaze1, oneway_gaze2, mutual_gaze1, mutual_gaze2):
# temp_resolu: temporal resolution, the time different between each step
# e.g. temp_resolu = 0.5s, each step is 0.5s
# totalsess_time: the time (s) of the total session

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy
    import string
    import warnings
    import pickle

    from pgmpy.models import BayesianModel
    from pgmpy.models import DynamicBayesianNetwork as DBN
    from pgmpy.estimators import BayesianEstimator
    from pgmpy.estimators import HillClimbSearch,BicScore
    import networkx as nx

    from AicScore import AicScore

    from ana_functions.train_DBN_alec import get_weighted_dags
    from ana_functions.train_DBN_alec import graph_to_matrix

    # define gaze start and stop
    animal1_gaze = np.concatenate([oneway_gaze1, mutual_gaze1])
    animal1_gaze = np.sort(np.unique(animal1_gaze))
    animal1_gaze_stop = animal1_gaze[np.concatenate(((animal1_gaze[1:]-animal1_gaze[0:-1]>gaze_thresold)*1,[1]))==1]
    animal1_gaze_start = np.concatenate(([animal1_gaze[0]],animal1_gaze[np.where(animal1_gaze[1:]-animal1_gaze[0:-1]>gaze_thresold)[0]+1]))
    #
    animal2_gaze = np.concatenate([oneway_gaze2, mutual_gaze2])
    animal2_gaze = np.sort(np.unique(animal2_gaze))
    animal2_gaze_stop = animal2_gaze[np.concatenate(((animal2_gaze[1:]-animal2_gaze[0:-1]>gaze_thresold)*1,[1]))==1]
    animal2_gaze_start = np.concatenate(([animal2_gaze[0]],animal2_gaze[np.where(animal2_gaze[1:]-animal2_gaze[0:-1]>gaze_thresold)[0]+1]))


    # define time points 
    total_time = int(np.floor((totalsess_time - session_start_time)/temp_resolu))
    time_point_pull1_round = np.floor(time_point_pull1/temp_resolu).reset_index(drop = True).astype(int)
    time_point_pull1_round = time_point_pull1_round[time_point_pull1_round<total_time]
    time_point_pull2_round  = np.floor(time_point_pull2/temp_resolu).reset_index(drop = True).astype(int)
    time_point_pull2_round = time_point_pull2_round[time_point_pull2_round<total_time]
    time_point_gaze1start_round = np.floor(pd.Series(animal1_gaze_start)/temp_resolu).reset_index(drop = True).astype(int)
    time_point_gaze2start_round = np.floor(pd.Series(animal2_gaze_start)/temp_resolu).reset_index(drop = True).astype(int)
    time_point_gaze1stop_round = np.floor(pd.Series(animal1_gaze_stop)/temp_resolu).reset_index(drop = True).astype(int)
    time_point_gaze2stop_round = np.floor(pd.Series(animal2_gaze_stop)/temp_resolu).reset_index(drop = True).astype(int)
    time_point_gaze1start_round = time_point_gaze1start_round[(time_point_gaze1start_round>0)&(time_point_gaze1start_round<total_time)]
    time_point_gaze2start_round = time_point_gaze2start_round[(time_point_gaze2start_round>0)&(time_point_gaze2start_round<total_time)]
    time_point_gaze1stop_round = time_point_gaze1stop_round[(time_point_gaze1stop_round>0)&(time_point_gaze1stop_round<total_time)]
    time_point_gaze2stop_round = time_point_gaze2stop_round[(time_point_gaze2stop_round>0)&(time_point_gaze2stop_round<total_time)]
    # t1
    pull1_t1 = np.zeros((total_time+1,1))
    pull1_t1[np.array(time_point_pull1_round)] = 1
    pull2_t1 = np.zeros((total_time+1,1))
    pull2_t1[np.array(time_point_pull2_round)] = 1
    gaze1start_t1 = np.zeros((total_time+1,1))
    gaze1start_t1[np.array(time_point_gaze1start_round)] = 1
    gaze2start_t1 = np.zeros((total_time+1,1))
    gaze2start_t1[np.array(time_point_gaze2start_round)] = 1
    gaze1stop_t1 = np.zeros((total_time+1,1))
    gaze1stop_t1[np.array(time_point_gaze1stop_round)] = 1
    gaze2stop_t1 = np.zeros((total_time+1,1))
    gaze2stop_t1[np.array(time_point_gaze2stop_round)] = 1
    # t0
    pull1_t0 = np.zeros((total_time+1,1))
    pull1_t0[np.array(time_point_pull1_round)+1] = 1
    pull2_t0 = np.zeros((total_time+1,1))
    pull2_t0[np.array(time_point_pull2_round)+1] = 1
    gaze1start_t0 = np.zeros((total_time+1,1))
    gaze1start_t0[np.array(time_point_gaze1start_round)+1] = 1
    gaze2start_t0 = np.zeros((total_time+1,1))
    gaze2start_t0[np.array(time_point_gaze2start_round)+1] = 1
    gaze1stop_t0 = np.zeros((total_time+1,1))
    gaze1stop_t0[np.array(time_point_gaze1stop_round)+1] = 1
    gaze2stop_t0 = np.zeros((total_time+1,1))
    gaze2stop_t0[np.array(time_point_gaze2stop_round)+1] = 1
    
    # clean up dataframe
    data = np.concatenate((pull1_t0,pull2_t0,gaze1start_t0,gaze2start_t0,gaze1stop_t0,gaze2stop_t0,pull1_t1,pull2_t1,gaze1start_t1,gaze2start_t1,gaze1stop_t1,gaze2stop_t1),axis = 1)
    colnames = ["pull1_t0","pull2_t0","gaze1start_t0","gaze2start_t0","gaze1stop_t0","gaze2stop_t0","pull1_t1","pull2_t1","gaze1start_t1","gaze2start_t1","gaze1stop_t1","gaze2stop_t1"]
    eventnames = ["pull1","pull2","gaze1start","gaze2start","gaze1stop","gaze2stop"]
    nevents = np.size(eventnames)
    bhv_df = pd.DataFrame(data, columns=colnames)

    # define DBN structures
    all_pops = list(bhv_df.columns)
    from_pops = [pop for pop in all_pops if not pop.endswith('t1')]
    to_pops = [pop for pop in all_pops if pop.endswith('t1')]
    # causal_whitelist = [(from_pop,to_pop) for from_pop in from_pops for to_pop in to_pops]
    causal_whitelist = [ ('pull1_t0', 'pull1_t1'),
                         ('pull1_t0', 'pull2_t1'),
                         ('pull1_t0', 'gaze1start_t1'),
                         ('pull1_t0', 'gaze2start_t1'),                    
                         ('pull2_t0', 'pull1_t1'),
                         ('pull2_t0', 'pull2_t1'),
                         ('pull2_t0', 'gaze1start_t1'),
                         ('pull2_t0', 'gaze2start_t1'),                     
                         ('gaze1stop_t0', 'pull1_t1'),
                         ('gaze1stop_t0', 'pull2_t1'),
                         ('gaze1stop_t0', 'gaze1start_t1'),
                         ('gaze1stop_t0', 'gaze2start_t1'),
                         ('gaze2stop_t0', 'pull1_t1'),
                         ('gaze2stop_t0', 'pull2_t1'),
                         ('gaze2stop_t0', 'gaze1start_t1'),
                         ('gaze2stop_t0', 'gaze2start_t1')]


    # train the DBN for the edges
    bhv_hc = HillClimbSearch(bhv_df)
    best_model = bhv_hc.estimate(max_indegree=None, white_list = causal_whitelist, scoring_method=AicScore(bhv_df))
    edges = best_model.edges()
    
    nFromNodes = nevents
    nToNodes = nevents
    nTrials = 1
    DAGs = np.zeros((nTrials, nFromNodes, nToNodes))
    DAGs[0,:,:] = graph_to_matrix(list(edges),nevents,eventnames)

    return best_model, edges, DAGs, eventnames, from_pops, to_pops, causal_whitelist






