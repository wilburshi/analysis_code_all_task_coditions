# function - train the dynamic bayesian network - multi lags: three lags

########################################
def graph_to_matrix(edges,nFromNodes,nToNodes,eventnames):

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

    output_matrix = np.zeros((nFromNodes,nToNodes)) 
    
    row = 0
    for from_timeslice in ['t0','t1','t2']:

        for from_pop in eventnames:

            #Loop through the receiving nodes (the last timeslice of each population)
            column = 0
            for to_pop in eventnames:

                from_pop_t = '{from_pop}_{from_timeslice}'.format(from_pop = from_pop, from_timeslice = from_timeslice)
                to_pop_t = '{to_pop}_t3'.format(to_pop = to_pop)

                if (from_pop_t, to_pop_t) in edges:
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


            if (p_value) < 0.001:
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
def Modulation_Index(weighted_graphs_1, weighted_graphs_2, sig_edges_1, sig_edges_2, nrepairs = 1000):

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy
    import string
    import warnings
    import pickle
    
    nbtstp1 = np.shape(weighted_graphs_1)[0]
    nbtstp2 = np.shape(weighted_graphs_2)[0]

    sig_edges_delta = ((sig_edges_1+sig_edges_2)>0)*1

    graph1_ids = np.random.randint(np.zeros((1,nrepairs))[0],nbtstp1)
    graph2_ids = np.random.randint(np.zeros((1,nrepairs))[0],nbtstp2)

    MI_delta = (weighted_graphs_2[graph2_ids,:,:]-weighted_graphs_1[graph1_ids,:,:])/(abs(weighted_graphs_2[graph2_ids,:,:])+abs(weighted_graphs_1[graph1_ids,:,:])+0.002)

    return MI_delta, sig_edges_delta



########################################
def train_DBN_multiLag_create_df_only(totalsess_time, session_start_time, temp_resolu, time_point_pull1, time_point_pull2, oneway_gaze1, oneway_gaze2, mutual_gaze1, mutual_gaze2,lever_gaze1,lever_gaze2,tube_gaze1,tube_gaze2):
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
    time_point_levergaze1_round = np.floor(pd.Series(lever_gaze1)/temp_resolu).reset_index(drop = True).astype(int)
    time_point_levergaze2_round = np.floor(pd.Series(lever_gaze2)/temp_resolu).reset_index(drop = True).astype(int)
    time_point_tubegaze1_round = np.floor(pd.Series(tube_gaze1)/temp_resolu).reset_index(drop = True).astype(int)
    time_point_tubegaze2_round = np.floor(pd.Series(tube_gaze2)/temp_resolu).reset_index(drop = True).astype(int)
    time_point_levergaze1_round = time_point_levergaze1_round[(time_point_levergaze1_round>0)&(time_point_levergaze1_round<total_time)]
    time_point_levergaze2_round = time_point_levergaze2_round[(time_point_levergaze2_round>0)&(time_point_levergaze2_round<total_time)]
    time_point_tubegaze1_round = time_point_tubegaze1_round[(time_point_tubegaze1_round>0)&(time_point_tubegaze1_round<total_time)]
    time_point_tubegaze2_round = time_point_tubegaze2_round[(time_point_tubegaze2_round>0)&(time_point_tubegaze2_round<total_time)]

	
	# t3 - current
    pull1_t3 = np.zeros((total_time+3,1))
    pull1_t3[np.array(time_point_pull1_round)] = 1
    pull2_t3 = np.zeros((total_time+3,1))
    pull2_t3[np.array(time_point_pull2_round)] = 1
    owgaze1_t3 = np.zeros((total_time+3,1))
    owgaze1_t3[np.array(time_point_onewaygaze1_round)] = 1
    owgaze2_t3 = np.zeros((total_time+3,1))
    owgaze2_t3[np.array(time_point_onewaygaze2_round)] = 1
    mtgaze1_t3 = np.zeros((total_time+3,1))
    mtgaze1_t3[np.array(time_point_mutualgaze1_round)] = 1
    mtgaze2_t3 = np.zeros((total_time+3,1))
    mtgaze2_t3[np.array(time_point_mutualgaze2_round)] = 1
    lvgaze1_t3 = np.zeros((total_time+3,1))
    lvgaze1_t3[np.array(time_point_levergaze1_round)] = 1
    lvgaze2_t3 = np.zeros((total_time+3,1))
    lvgaze2_t3[np.array(time_point_levergaze2_round)] = 1
    tbgaze1_t3 = np.zeros((total_time+3,1))
    tbgaze1_t3[np.array(time_point_tubegaze1_round)] = 1
    tbgaze2_t3 = np.zeros((total_time+3,1))
    tbgaze2_t3[np.array(time_point_tubegaze2_round)] = 1
    # t2 - one step back
    pull1_t2 = np.zeros((total_time+3,1))
    pull1_t2[np.array(time_point_pull1_round)+1] = 1
    pull2_t2 = np.zeros((total_time+3,1))
    pull2_t2[np.array(time_point_pull2_round)+1] = 1
    owgaze1_t2 = np.zeros((total_time+3,1))
    owgaze1_t2[np.array(time_point_onewaygaze1_round)+1] = 1
    owgaze2_t2 = np.zeros((total_time+3,1))
    owgaze2_t2[np.array(time_point_onewaygaze2_round)+1] = 1
    mtgaze1_t2 = np.zeros((total_time+3,1))
    mtgaze1_t2[np.array(time_point_mutualgaze1_round)+1] = 1
    mtgaze2_t2 = np.zeros((total_time+3,1))
    mtgaze2_t2[np.array(time_point_mutualgaze2_round)+1] = 1
    lvgaze1_t2 = np.zeros((total_time+3,1))
    lvgaze1_t2[np.array(time_point_levergaze1_round)+1] = 1
    lvgaze2_t2 = np.zeros((total_time+3,1))
    lvgaze2_t2[np.array(time_point_levergaze2_round)+1] = 1
    tbgaze1_t2 = np.zeros((total_time+3,1))
    tbgaze1_t2[np.array(time_point_tubegaze1_round)+1] = 1
    tbgaze2_t2 = np.zeros((total_time+3,1))
    tbgaze2_t2[np.array(time_point_tubegaze2_round)+1] = 1
    # t1 - two steps back
    pull1_t1 = np.zeros((total_time+3,1))
    pull1_t1[np.array(time_point_pull1_round)+2] = 1
    pull2_t1 = np.zeros((total_time+3,1))
    pull2_t1[np.array(time_point_pull2_round)+2] = 1
    owgaze1_t1 = np.zeros((total_time+3,1))
    owgaze1_t1[np.array(time_point_onewaygaze1_round)+2] = 1
    owgaze2_t1 = np.zeros((total_time+3,1))
    owgaze2_t1[np.array(time_point_onewaygaze2_round)+2] = 1
    mtgaze1_t1 = np.zeros((total_time+3,1))
    mtgaze1_t1[np.array(time_point_mutualgaze1_round)+2] = 1
    mtgaze2_t1 = np.zeros((total_time+3,1))
    mtgaze2_t1[np.array(time_point_mutualgaze2_round)+2] = 1
    lvgaze1_t1 = np.zeros((total_time+3,1))
    lvgaze1_t1[np.array(time_point_levergaze1_round)+2] = 1
    lvgaze2_t1 = np.zeros((total_time+3,1))
    lvgaze2_t1[np.array(time_point_levergaze2_round)+2] = 1
    tbgaze1_t1 = np.zeros((total_time+3,1))
    tbgaze1_t1[np.array(time_point_tubegaze1_round)+2] = 1
    tbgaze2_t1 = np.zeros((total_time+3,1))
    tbgaze2_t1[np.array(time_point_tubegaze2_round)+2] = 1
    # t0 - three steps back
    pull1_t0 = np.zeros((total_time+3,1))
    pull1_t0[np.array(time_point_pull1_round)+3] = 1
    pull2_t0 = np.zeros((total_time+3,1))
    pull2_t0[np.array(time_point_pull2_round)+3] = 1
    owgaze1_t0 = np.zeros((total_time+3,1))
    owgaze1_t0[np.array(time_point_onewaygaze1_round)+3] = 1
    owgaze2_t0 = np.zeros((total_time+3,1))
    owgaze2_t0[np.array(time_point_onewaygaze2_round)+3] = 1
    mtgaze1_t0 = np.zeros((total_time+3,1))
    mtgaze1_t0[np.array(time_point_mutualgaze1_round)+3] = 1
    mtgaze2_t0 = np.zeros((total_time+3,1))
    mtgaze2_t0[np.array(time_point_mutualgaze2_round)+3] = 1
    lvgaze1_t0 = np.zeros((total_time+3,1))
    lvgaze1_t0[np.array(time_point_levergaze1_round)+3] = 1
    lvgaze2_t0 = np.zeros((total_time+3,1))
    lvgaze2_t0[np.array(time_point_levergaze2_round)+3] = 1
    tbgaze1_t0 = np.zeros((total_time+3,1))
    tbgaze1_t0[np.array(time_point_tubegaze1_round)+3] = 1
    tbgaze2_t0 = np.zeros((total_time+3,1))
    tbgaze2_t0[np.array(time_point_tubegaze2_round)+3] = 1
    
    # clean up dataframe
    # data = np.concatenate((pull1_t0,pull2_t0,owgaze1_t0,owgaze2_t0,lvgaze1_t0,lvgaze2_t0,tbgaze1_t0,tbgaze2_t0,pull1_t1,pull2_t1,owgaze1_t1,owgaze2_t1,lvgaze1_t1,lvgaze2_t1,tbgaze1_t1,tbgaze2_t1,pull1_t2,pull2_t2,owgaze1_t2,owgaze2_t2,lvgaze1_t2,lvgaze2_t2,tbgaze1_t2,tbgaze2_t2,pull1_t3,pull2_t3,owgaze1_t3,owgaze2_t3,lvgaze1_t3,lvgaze2_t3,tbgaze1_t3,tbgaze2_t3),axis = 1)
    # colnames = ["pull1_t0","pull2_t0","owgaze1_t0","owgaze2_t0","lvgaze1_t0","lvgaze2_t0","tbgaze1_t0","tbgaze2_t0","pull1_t1","pull2_t1","owgaze1_t1","owgaze2_t1","lvgaze1_t1","lvgaze2_t1","tbgaze1_t1","tbgaze2_t1","pull1_t2","pull2_t2","owgaze1_t2","owgaze2_t2","lvgaze1_t2","lvgaze2_t2","tbgaze1_t2","tbgaze2_t2","pull1_t3","pull2_t3","owgaze1_t3","owgaze2_t3","lvgaze1_t3","lvgaze2_t3","tbgaze1_t3","tbgaze2_t3"]
    # eventnames = ["pull1","pull2","owgaze1","owgaze2","lvgaze1","lvgaze2","tbgaze1","tbgaze2"]
    data = np.concatenate((pull1_t0,pull2_t0,owgaze1_t0,owgaze2_t0,tbgaze1_t0,tbgaze2_t0,pull1_t1,pull2_t1,owgaze1_t1,owgaze2_t1,tbgaze1_t1,tbgaze2_t1,pull1_t2,pull2_t2,owgaze1_t2,owgaze2_t2,tbgaze1_t2,tbgaze2_t2,pull1_t3,pull2_t3,owgaze1_t3,owgaze2_t3,tbgaze1_t3,tbgaze2_t3),axis = 1)
    colnames = ["pull1_t0","pull2_t0","owgaze1_t0","owgaze2_t0","tbgaze1_t0","tbgaze2_t0","pull1_t1","pull2_t1","owgaze1_t1","owgaze2_t1","tbgaze1_t1","tbgaze2_t1","pull1_t2","pull2_t2","owgaze1_t2","owgaze2_t2","tbgaze1_t2","tbgaze2_t2","pull1_t3","pull2_t3","owgaze1_t3","owgaze2_t3","tbgaze1_t3","tbgaze2_t3"]
    eventnames = ["pull1","pull2","owgaze1","owgaze2","tbgaze1","tbgaze2"]
    nevents = np.size(eventnames)
    bhv_df = pd.DataFrame(data, columns=colnames)

    return bhv_df, colnames, eventnames 



########################################
def train_DBN_multiLag_training_only(bhv_df,starting_graph,colnames,eventnames,from_pops,to_pops):

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

    from ana_functions.train_DBN_multiLag import get_weighted_dags
    from ana_functions.train_DBN_multiLag import graph_to_matrix
    
    nevents = np.size(eventnames)    

    # define DBN structures
    all_pops = list(bhv_df.columns)
    causal_whitelist = [(from_pop,to_pop) for from_pop in from_pops for to_pop in to_pops]

    # train the DBN for the edges
    bhv_hc = HillClimbSearch(bhv_df)
    best_model = bhv_hc.estimate(max_indegree=None, white_list=causal_whitelist, scoring_method=AicScore(bhv_df), start_dag=starting_graph)
    edges = best_model.edges()
    
    nFromNodes = np.shape(from_pops)[0]
    nToNodes = np.shape(to_pops)[0]

    nTrials = 1
    DAGs = np.zeros((nTrials, nFromNodes, nToNodes))
    DAGs[0,:,:] = graph_to_matrix(list(edges),nFromNodes,nToNodes,eventnames)

    return best_model, edges, DAGs



########################################
def train_DBN_multiLag(totalsess_time, session_start_time, temp_resolu, time_point_pull1, time_point_pull2, oneway_gaze1, oneway_gaze2, mutual_gaze1, mutual_gaze2,lever_gaze1,lever_gaze2,tube_gaze1,tube_gaze2):
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
    
    from ana_functions.train_DBN_multiLag import get_weighted_dags
    from ana_functions.train_DBN_multiLag import graph_to_matrix
    
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
    time_point_levergaze1_round = np.floor(pd.Series(lever_gaze1)/temp_resolu).reset_index(drop = True).astype(int)
    time_point_levergaze2_round = np.floor(pd.Series(lever_gaze2)/temp_resolu).reset_index(drop = True).astype(int)
    time_point_tubegaze1_round = np.floor(pd.Series(tube_gaze1)/temp_resolu).reset_index(drop = True).astype(int)
    time_point_tubegaze2_round = np.floor(pd.Series(tube_gaze2)/temp_resolu).reset_index(drop = True).astype(int)
    time_point_levergaze1_round = time_point_levergaze1_round[(time_point_levergaze1_round>0)&(time_point_levergaze1_round<total_time)]
    time_point_levergaze2_round = time_point_levergaze2_round[(time_point_levergaze2_round>0)&(time_point_levergaze2_round<total_time)]
    time_point_tubegaze1_round = time_point_tubegaze1_round[(time_point_tubegaze1_round>0)&(time_point_tubegaze1_round<total_time)]
    time_point_tubegaze2_round = time_point_tubegaze2_round[(time_point_tubegaze2_round>0)&(time_point_tubegaze2_round<total_time)]

	
	# t3 - current
    pull1_t3 = np.zeros((total_time+3,1))
    pull1_t3[np.array(time_point_pull1_round)] = 1
    pull2_t3 = np.zeros((total_time+3,1))
    pull2_t3[np.array(time_point_pull2_round)] = 1
    owgaze1_t3 = np.zeros((total_time+3,1))
    owgaze1_t3[np.array(time_point_onewaygaze1_round)] = 1
    owgaze2_t3 = np.zeros((total_time+3,1))
    owgaze2_t3[np.array(time_point_onewaygaze2_round)] = 1
    mtgaze1_t3 = np.zeros((total_time+3,1))
    mtgaze1_t3[np.array(time_point_mutualgaze1_round)] = 1
    mtgaze2_t3 = np.zeros((total_time+3,1))
    mtgaze2_t3[np.array(time_point_mutualgaze2_round)] = 1
    lvgaze1_t3 = np.zeros((total_time+3,1))
    lvgaze1_t3[np.array(time_point_levergaze1_round)] = 1
    lvgaze2_t3 = np.zeros((total_time+3,1))
    lvgaze2_t3[np.array(time_point_levergaze2_round)] = 1
    tbgaze1_t3 = np.zeros((total_time+3,1))
    tbgaze1_t3[np.array(time_point_tubegaze1_round)] = 1
    tbgaze2_t3 = np.zeros((total_time+3,1))
    tbgaze2_t3[np.array(time_point_tubegaze2_round)] = 1
    # t2 - one step back
    pull1_t2 = np.zeros((total_time+3,1))
    pull1_t2[np.array(time_point_pull1_round)+1] = 1
    pull2_t2 = np.zeros((total_time+3,1))
    pull2_t2[np.array(time_point_pull2_round)+1] = 1
    owgaze1_t2 = np.zeros((total_time+3,1))
    owgaze1_t2[np.array(time_point_onewaygaze1_round)+1] = 1
    owgaze2_t2 = np.zeros((total_time+3,1))
    owgaze2_t2[np.array(time_point_onewaygaze2_round)+1] = 1
    mtgaze1_t2 = np.zeros((total_time+3,1))
    mtgaze1_t2[np.array(time_point_mutualgaze1_round)+1] = 1
    mtgaze2_t2 = np.zeros((total_time+3,1))
    mtgaze2_t2[np.array(time_point_mutualgaze2_round)+1] = 1
    lvgaze1_t2 = np.zeros((total_time+3,1))
    lvgaze1_t2[np.array(time_point_levergaze1_round)+1] = 1
    lvgaze2_t2 = np.zeros((total_time+3,1))
    lvgaze2_t2[np.array(time_point_levergaze2_round)+1] = 1
    tbgaze1_t2 = np.zeros((total_time+3,1))
    tbgaze1_t2[np.array(time_point_tubegaze1_round)+1] = 1
    tbgaze2_t2 = np.zeros((total_time+3,1))
    tbgaze2_t2[np.array(time_point_tubegaze2_round)+1] = 1
    # t1 - two steps back
    pull1_t1 = np.zeros((total_time+3,1))
    pull1_t1[np.array(time_point_pull1_round)+2] = 1
    pull2_t1 = np.zeros((total_time+3,1))
    pull2_t1[np.array(time_point_pull2_round)+2] = 1
    owgaze1_t1 = np.zeros((total_time+3,1))
    owgaze1_t1[np.array(time_point_onewaygaze1_round)+2] = 1
    owgaze2_t1 = np.zeros((total_time+3,1))
    owgaze2_t1[np.array(time_point_onewaygaze2_round)+2] = 1
    mtgaze1_t1 = np.zeros((total_time+3,1))
    mtgaze1_t1[np.array(time_point_mutualgaze1_round)+2] = 1
    mtgaze2_t1 = np.zeros((total_time+3,1))
    mtgaze2_t1[np.array(time_point_mutualgaze2_round)+2] = 1
    lvgaze1_t1 = np.zeros((total_time+3,1))
    lvgaze1_t1[np.array(time_point_levergaze1_round)+2] = 1
    lvgaze2_t1 = np.zeros((total_time+3,1))
    lvgaze2_t1[np.array(time_point_levergaze2_round)+2] = 1
    tbgaze1_t1 = np.zeros((total_time+3,1))
    tbgaze1_t1[np.array(time_point_tubegaze1_round)+2] = 1
    tbgaze2_t1 = np.zeros((total_time+3,1))
    tbgaze2_t1[np.array(time_point_tubegaze2_round)+2] = 1
    # t0 - three steps back
    pull1_t0 = np.zeros((total_time+3,1))
    pull1_t0[np.array(time_point_pull1_round)+3] = 1
    pull2_t0 = np.zeros((total_time+3,1))
    pull2_t0[np.array(time_point_pull2_round)+3] = 1
    owgaze1_t0 = np.zeros((total_time+3,1))
    owgaze1_t0[np.array(time_point_onewaygaze1_round)+3] = 1
    owgaze2_t0 = np.zeros((total_time+3,1))
    owgaze2_t0[np.array(time_point_onewaygaze2_round)+3] = 1
    mtgaze1_t0 = np.zeros((total_time+3,1))
    mtgaze1_t0[np.array(time_point_mutualgaze1_round)+3] = 1
    mtgaze2_t0 = np.zeros((total_time+3,1))
    mtgaze2_t0[np.array(time_point_mutualgaze2_round)+3] = 1
    lvgaze1_t0 = np.zeros((total_time+3,1))
    lvgaze1_t0[np.array(time_point_levergaze1_round)+3] = 1
    lvgaze2_t0 = np.zeros((total_time+3,1))
    lvgaze2_t0[np.array(time_point_levergaze2_round)+3] = 1
    tbgaze1_t0 = np.zeros((total_time+3,1))
    tbgaze1_t0[np.array(time_point_tubegaze1_round)+3] = 1
    tbgaze2_t0 = np.zeros((total_time+3,1))
    tbgaze2_t0[np.array(time_point_tubegaze2_round)+3] = 1
    
    # clean up dataframe
    # data = np.concatenate((pull1_t0,pull2_t0,owgaze1_t0,owgaze2_t0,lvgaze1_t0,lvgaze2_t0,tbgaze1_t0,tbgaze2_t0,pull1_t1,pull2_t1,owgaze1_t1,owgaze2_t1,lvgaze1_t1,lvgaze2_t1,tbgaze1_t1,tbgaze2_t1,pull1_t2,pull2_t2,owgaze1_t2,owgaze2_t2,lvgaze1_t2,lvgaze2_t2,tbgaze1_t2,tbgaze2_t2,pull1_t3,pull2_t3,owgaze1_t3,owgaze2_t3,lvgaze1_t3,lvgaze2_t3,tbgaze1_t3,tbgaze2_t3),axis = 1)
    # colnames = ["pull1_t0","pull2_t0","owgaze1_t0","owgaze2_t0","lvgaze1_t0","lvgaze2_t0","tbgaze1_t0","tbgaze2_t0","pull1_t1","pull2_t1","owgaze1_t1","owgaze2_t1","lvgaze1_t1","lvgaze2_t1","tbgaze1_t1","tbgaze2_t1","pull1_t2","pull2_t2","owgaze1_t2","owgaze2_t2","lvgaze1_t2","lvgaze2_t2","tbgaze1_t2","tbgaze2_t2","pull1_t3","pull2_t3","owgaze1_t3","owgaze2_t3","lvgaze1_t3","lvgaze2_t3","tbgaze1_t3","tbgaze2_t3"]
    # eventnames = ["pull1","pull2","owgaze1","owgaze2","lvgaze1","lvgaze2","tbgaze1","tbgaze2"]
    data = np.concatenate((pull1_t0,pull2_t0,owgaze1_t0,owgaze2_t0,tbgaze1_t0,tbgaze2_t0,pull1_t1,pull2_t1,owgaze1_t1,owgaze2_t1,tbgaze1_t1,tbgaze2_t1,pull1_t2,pull2_t2,owgaze1_t2,owgaze2_t2,tbgaze1_t2,tbgaze2_t2,pull1_t3,pull2_t3,owgaze1_t3,owgaze2_t3,tbgaze1_t3,tbgaze2_t3),axis = 1)
    colnames = ["pull1_t0","pull2_t0","owgaze1_t0","owgaze2_t0","tbgaze1_t0","tbgaze2_t0","pull1_t1","pull2_t1","owgaze1_t1","owgaze2_t1","tbgaze1_t1","tbgaze2_t1","pull1_t2","pull2_t2","owgaze1_t2","owgaze2_t2","tbgaze1_t2","tbgaze2_t2","pull1_t3","pull2_t3","owgaze1_t3","owgaze2_t3","tbgaze1_t3","tbgaze2_t3"]
    eventnames = ["pull1","pull2","owgaze1","owgaze2","tbgaze1","tbgaze2"]
    nevents = np.size(eventnames)
    bhv_df = pd.DataFrame(data, columns=colnames)

    # define DBN structures
    all_pops = list(bhv_df.columns)
    from_pops = [pop for pop in all_pops if not pop.endswith('t3')]
    to_pops = [pop for pop in all_pops if pop.endswith('t3')]
    causal_whitelist = [(from_pop,to_pop) for from_pop in from_pops for to_pop in to_pops]

    # train the DBN for the edges
    bhv_hc = HillClimbSearch(bhv_df)
    best_model = bhv_hc.estimate(max_indegree=None, white_list = causal_whitelist, scoring_method=AicScore(bhv_df))
    edges = best_model.edges()
    
    nFromNodes = np.shape(from_pops)[0]
    nToNodes = np.shape(to_pops)[0]
    nTrials = 1
    DAGs = np.zeros((nTrials, nFromNodes, nToNodes))
    DAGs[0,:,:] = graph_to_matrix(list(edges),nFromNodes,nToNodes,eventnames)

    return best_model, edges, DAGs, eventnames, from_pops, to_pops






########################################
def train_DBN_multiLag_eachtrial(totalsess_time, session_start_time, temp_resolu, time_point_pull1, time_point_pull2, oneway_gaze1, oneway_gaze2, mutual_gaze1, mutual_gaze2,lever_gaze1,lever_gaze2,tube_gaze1,tube_gaze2):
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

    from ana_functions.train_DBN_multiLag import get_weighted_dags
    from ana_functions.train_DBN_multiLag import graph_to_matrix
    
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
    time_point_levergaze1_round = np.floor(pd.Series(lever_gaze1)/temp_resolu).reset_index(drop = True).astype(int)
    time_point_levergaze2_round = np.floor(pd.Series(lever_gaze2)/temp_resolu).reset_index(drop = True).astype(int)
    time_point_tubegaze1_round = np.floor(pd.Series(tube_gaze1)/temp_resolu).reset_index(drop = True).astype(int)
    time_point_tubegaze2_round = np.floor(pd.Series(tube_gaze2)/temp_resolu).reset_index(drop = True).astype(int)
    time_point_levergaze1_round = time_point_levergaze1_round[(time_point_levergaze1_round>0)&(time_point_levergaze1_round<total_time)]
    time_point_levergaze2_round = time_point_levergaze2_round[(time_point_levergaze2_round>0)&(time_point_levergaze2_round<total_time)]
    time_point_tubegaze1_round = time_point_tubegaze1_round[(time_point_tubegaze1_round>0)&(time_point_tubegaze1_round<total_time)]
    time_point_tubegaze2_round = time_point_tubegaze2_round[(time_point_tubegaze2_round>0)&(time_point_tubegaze2_round<total_time)]

	
	# t3 - current
    pull1_t3 = np.zeros((total_time+3,1))
    pull1_t3[np.array(time_point_pull1_round)] = 1
    pull2_t3 = np.zeros((total_time+3,1))
    pull2_t3[np.array(time_point_pull2_round)] = 1
    owgaze1_t3 = np.zeros((total_time+3,1))
    owgaze1_t3[np.array(time_point_onewaygaze1_round)] = 1
    owgaze2_t3 = np.zeros((total_time+3,1))
    owgaze2_t3[np.array(time_point_onewaygaze2_round)] = 1
    mtgaze1_t3 = np.zeros((total_time+3,1))
    mtgaze1_t3[np.array(time_point_mutualgaze1_round)] = 1
    mtgaze2_t3 = np.zeros((total_time+3,1))
    mtgaze2_t3[np.array(time_point_mutualgaze2_round)] = 1
    lvgaze1_t3 = np.zeros((total_time+3,1))
    lvgaze1_t3[np.array(time_point_levergaze1_round)] = 1
    lvgaze2_t3 = np.zeros((total_time+3,1))
    lvgaze2_t3[np.array(time_point_levergaze2_round)] = 1
    tbgaze1_t3 = np.zeros((total_time+3,1))
    tbgaze1_t3[np.array(time_point_tubegaze1_round)] = 1
    tbgaze2_t3 = np.zeros((total_time+3,1))
    tbgaze2_t3[np.array(time_point_tubegaze2_round)] = 1
    # t2 - one step back
    pull1_t2 = np.zeros((total_time+3,1))
    pull1_t2[np.array(time_point_pull1_round)+1] = 1
    pull2_t2 = np.zeros((total_time+3,1))
    pull2_t2[np.array(time_point_pull2_round)+1] = 1
    owgaze1_t2 = np.zeros((total_time+3,1))
    owgaze1_t2[np.array(time_point_onewaygaze1_round)+1] = 1
    owgaze2_t2 = np.zeros((total_time+3,1))
    owgaze2_t2[np.array(time_point_onewaygaze2_round)+1] = 1
    mtgaze1_t2 = np.zeros((total_time+3,1))
    mtgaze1_t2[np.array(time_point_mutualgaze1_round)+1] = 1
    mtgaze2_t2 = np.zeros((total_time+3,1))
    mtgaze2_t2[np.array(time_point_mutualgaze2_round)+1] = 1
    lvgaze1_t2 = np.zeros((total_time+3,1))
    lvgaze1_t2[np.array(time_point_levergaze1_round)+1] = 1
    lvgaze2_t2 = np.zeros((total_time+3,1))
    lvgaze2_t2[np.array(time_point_levergaze2_round)+1] = 1
    tbgaze1_t2 = np.zeros((total_time+3,1))
    tbgaze1_t2[np.array(time_point_tubegaze1_round)+1] = 1
    tbgaze2_t2 = np.zeros((total_time+3,1))
    tbgaze2_t2[np.array(time_point_tubegaze2_round)+1] = 1
    # t1 - two steps back
    pull1_t1 = np.zeros((total_time+3,1))
    pull1_t1[np.array(time_point_pull1_round)+2] = 1
    pull2_t1 = np.zeros((total_time+3,1))
    pull2_t1[np.array(time_point_pull2_round)+2] = 1
    owgaze1_t1 = np.zeros((total_time+3,1))
    owgaze1_t1[np.array(time_point_onewaygaze1_round)+2] = 1
    owgaze2_t1 = np.zeros((total_time+3,1))
    owgaze2_t1[np.array(time_point_onewaygaze2_round)+2] = 1
    mtgaze1_t1 = np.zeros((total_time+3,1))
    mtgaze1_t1[np.array(time_point_mutualgaze1_round)+2] = 1
    mtgaze2_t1 = np.zeros((total_time+3,1))
    mtgaze2_t1[np.array(time_point_mutualgaze2_round)+2] = 1
    lvgaze1_t1 = np.zeros((total_time+3,1))
    lvgaze1_t1[np.array(time_point_levergaze1_round)+2] = 1
    lvgaze2_t1 = np.zeros((total_time+3,1))
    lvgaze2_t1[np.array(time_point_levergaze2_round)+2] = 1
    tbgaze1_t1 = np.zeros((total_time+3,1))
    tbgaze1_t1[np.array(time_point_tubegaze1_round)+2] = 1
    tbgaze2_t1 = np.zeros((total_time+3,1))
    tbgaze2_t1[np.array(time_point_tubegaze2_round)+2] = 1
    # t0 - three steps back
    pull1_t0 = np.zeros((total_time+3,1))
    pull1_t0[np.array(time_point_pull1_round)+3] = 1
    pull2_t0 = np.zeros((total_time+3,1))
    pull2_t0[np.array(time_point_pull2_round)+3] = 1
    owgaze1_t0 = np.zeros((total_time+3,1))
    owgaze1_t0[np.array(time_point_onewaygaze1_round)+3] = 1
    owgaze2_t0 = np.zeros((total_time+3,1))
    owgaze2_t0[np.array(time_point_onewaygaze2_round)+3] = 1
    mtgaze1_t0 = np.zeros((total_time+3,1))
    mtgaze1_t0[np.array(time_point_mutualgaze1_round)+3] = 1
    mtgaze2_t0 = np.zeros((total_time+3,1))
    mtgaze2_t0[np.array(time_point_mutualgaze2_round)+3] = 1
    lvgaze1_t0 = np.zeros((total_time+3,1))
    lvgaze1_t0[np.array(time_point_levergaze1_round)+3] = 1
    lvgaze2_t0 = np.zeros((total_time+3,1))
    lvgaze2_t0[np.array(time_point_levergaze2_round)+3] = 1
    tbgaze1_t0 = np.zeros((total_time+3,1))
    tbgaze1_t0[np.array(time_point_tubegaze1_round)+3] = 1
    tbgaze2_t0 = np.zeros((total_time+3,1))
    tbgaze2_t0[np.array(time_point_tubegaze2_round)+3] = 1
    
    # clean up dataframe
    # data = np.concatenate((pull1_t0,pull2_t0,owgaze1_t0,owgaze2_t0,lvgaze1_t0,lvgaze2_t0,tbgaze1_t0,tbgaze2_t0,pull1_t1,pull2_t1,owgaze1_t1,owgaze2_t1,lvgaze1_t1,lvgaze2_t1,tbgaze1_t1,tbgaze2_t1,pull1_t2,pull2_t2,owgaze1_t2,owgaze2_t2,lvgaze1_t2,lvgaze2_t2,tbgaze1_t2,tbgaze2_t2,pull1_t3,pull2_t3,owgaze1_t3,owgaze2_t3,lvgaze1_t3,lvgaze2_t3,tbgaze1_t3,tbgaze2_t3),axis = 1)
    # colnames = ["pull1_t0","pull2_t0","owgaze1_t0","owgaze2_t0","lvgaze1_t0","lvgaze2_t0","tbgaze1_t0","tbgaze2_t0","pull1_t1","pull2_t1","owgaze1_t1","owgaze2_t1","lvgaze1_t1","lvgaze2_t1","tbgaze1_t1","tbgaze2_t1","pull1_t2","pull2_t2","owgaze1_t2","owgaze2_t2","lvgaze1_t2","lvgaze2_t2","tbgaze1_t2","tbgaze2_t2","pull1_t3","pull2_t3","owgaze1_t3","owgaze2_t3","lvgaze1_t3","lvgaze2_t3","tbgaze1_t3","tbgaze2_t3"]
    # eventnames = ["pull1","pull2","owgaze1","owgaze2","lvgaze1","lvgaze2","tbgaze1","tbgaze2"]
    data = np.concatenate((pull1_t0,pull2_t0,owgaze1_t0,owgaze2_t0,tbgaze1_t0,tbgaze2_t0,pull1_t1,pull2_t1,owgaze1_t1,owgaze2_t1,tbgaze1_t1,tbgaze2_t1,pull1_t2,pull2_t2,owgaze1_t2,owgaze2_t2,tbgaze1_t2,tbgaze2_t2,pull1_t3,pull2_t3,owgaze1_t3,owgaze2_t3,tbgaze1_t3,tbgaze2_t3),axis = 1)
    colnames = ["pull1_t0","pull2_t0","owgaze1_t0","owgaze2_t0","tbgaze1_t0","tbgaze2_t0","pull1_t1","pull2_t1","owgaze1_t1","owgaze2_t1","tbgaze1_t1","tbgaze2_t1","pull1_t2","pull2_t2","owgaze1_t2","owgaze2_t2","tbgaze1_t2","tbgaze2_t2","pull1_t3","pull2_t3","owgaze1_t3","owgaze2_t3","tbgaze1_t3","tbgaze2_t3"]
    eventnames = ["pull1","pull2","owgaze1","owgaze2","tbgaze1","tbgaze2"]
    nevents = np.size(eventnames)
    bhv_df = pd.DataFrame(data, columns=colnames)
    
    # every 10s (20 0.5s steps) as a "trial"
    # total 720s "session" will be 72 "trials"
    ntrials = int(np.floor(totalsess_time/10))
    
    for itrial in np.arange(0,ntrials,1):
        bhv_df_itrial = bhv_df.iloc[itrial*(10/temp_resolu):(itrial+1)*(10/temp_resolu)]
        # define DBN structures
        all_pops = list(bhv_df_itrial.columns)
        from_pops = [pop for pop in all_pops if not pop.endswith('t3')]
        to_pops = [pop for pop in all_pops if pop.endswith('t3')]
        causal_whitelist = [(from_pop,to_pop) for from_pop in from_pops for to_pop in to_pops]
    
        nFromNodes = np.shape(from_pops)[0]
        nToNodes = np.shape(to_pops)[0]
        # train the DBN for the edges
        bhv_hc = HillClimbSearch(bhv_df_itrial)
        best_model = bhv_hc.estimate(max_indegree=None, white_list = causal_whitelist, scoring_method=AicScore(bhv_df))
        edges_itrial = best_model.edges()   
    
        DAGs_itrial = np.zeros((ntrials, nFromNodes, nToNodes))
        DAGs_itrial[0,:,:] = graph_to_matrix(list(edges_itrial),nFromNodes,nToNodes,eventnames)

    weighted_graphs = get_weighted_dags(DAGs_itrial,nNewBootstraps = 1)
    return DAGs_itrial, weighted_graphs, eventnames, from_pops, to_pops






