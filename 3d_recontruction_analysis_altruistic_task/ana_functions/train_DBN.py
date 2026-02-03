# train the dynamic bayesian network

def train_DBN(totalsess_time, temp_resolu, time_point_pull1, time_point_pull2, oneway_gaze1, oneway_gaze2, mutual_gaze1, mutual_gaze2):

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

# temp_resolu: temporal resolution, the time different between each step
# e.g. temp_resolu = 0.5s, each step is 0.5s
    
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
    # t0
    pull1_t0 = np.zeros((total_time+1,1))
    pull1_t0[np.array(time_point_pull1_round)] = 1
    pull2_t0 = np.zeros((total_time+1,1))
    pull2_t0[np.array(time_point_pull2_round)] = 1
    owgaze1_t0 = np.zeros((total_time+1,1))
    owgaze1_t0[np.array(time_point_onewaygaze1_round)] = 1
    owgaze2_t0 = np.zeros((total_time+1,1))
    owgaze2_t0[np.array(time_point_onewaygaze2_round)] = 1
    mtgaze1_t0 = np.zeros((total_time+1,1))
    mtgaze1_t0[np.array(time_point_mutualgaze1_round)] = 1
    mtgaze2_t0 = np.zeros((total_time+1,1))
    mtgaze2_t0[np.array(time_point_mutualgaze2_round)] = 1
    # t1
    pull1_t1 = np.zeros((total_time+1,1))
    pull1_t1[np.array(time_point_pull1_round)+1] = 1
    pull2_t1 = np.zeros((total_time+1,1))
    pull2_t1[np.array(time_point_pull2_round)+1] = 1
    owgaze1_t1 = np.zeros((total_time+1,1))
    owgaze1_t1[np.array(time_point_onewaygaze1_round)+1] = 1
    owgaze2_t1 = np.zeros((total_time+1,1))
    owgaze2_t1[np.array(time_point_onewaygaze2_round)+1] = 1
    mtgaze1_t1 = np.zeros((total_time+1,1))
    mtgaze1_t1[np.array(time_point_mutualgaze1_round)+1] = 1
    mtgaze2_t1 = np.zeros((total_time+1,1))
    mtgaze2_t1[np.array(time_point_mutualgaze2_round)+1] = 1
    
    ## create dataframe
    # data = np.concatenate((pull1_t0,pull2_t0,owgaze1_t0,owgaze2_t0,mtgaze1_t0,mtgaze2_t0,pull1_t1,pull2_t1,owgaze1_t1,owgaze2_t1,mtgaze1_t1,mtgaze2_t1),axis = 1)
    # colnames = [("pull1",0),("pull2",0),("owgaze1",0),("owgaze2",0),("mtgaze1",0),("mtgaze2",0),("pull1",1),("pull2",1),("owgaze1",1),("owgaze2",1),("mtgaze1",1),("mtgaze2",1)]
    # df = pd.DataFrame(data, columns=colnames)
    data = np.concatenate((pull1_t0,pull2_t0,owgaze1_t0,owgaze2_t0,pull1_t1,pull2_t1,owgaze1_t1,owgaze2_t1),axis = 1)
    colnames = [("pull1",0),("pull2",0),("owgaze1",0),("owgaze2",0),("pull1",1),("pull2",1),("owgaze1",1),("owgaze2",1)]
    df = pd.DataFrame(data, columns=colnames)

    ## built the model structure
    # model = DBN(
    #    [
    #        (("owgaze1",0), ("pull1",1)),
    #        (("owgaze1",0), ("mtgaze1",1)),
    #        (("mtgaze1",0), ("pull1",1)),
    #        (("pull1",0), ("owgaze1",1)),
    #        (("owgaze2",0), ("pull2",1)),
    #        (("owgaze2",0), ("mtgaze2",1)),
    #        (("mtgaze2",0), ("pull2",1)),
    #        (("pull2",0), ("owgaze2",1)),
    #    ]
    # )
    model = DBN(
        [
            (("owgaze1",0), ("pull1",1)),
            (("pull1",0), ("owgaze1",1)),
            (("owgaze2",0), ("pull2",1)),
            (("pull2",0), ("owgaze2",1)),
        ]
    )
    model.fit(df)
    
    return model
