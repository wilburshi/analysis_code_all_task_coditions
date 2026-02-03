# function - define time point of behavioral events

def bhv_events_timepoint_singlecam_otherlever(bhv_data, look_at_otherlever_or_not_merge):
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy
    import string
    import warnings
    import pickle

    # round the time of looking at the levers or tubes
    ind_lookatotherlever1 = np.where(np.array(look_at_otherlever_or_not_merge['dodson'])==1)
    time_point_lookatotherlever1 = look_at_otherlever_or_not_merge["time_in_second"][ind_lookatotherlever1] 
    ind_lookatotherlever2 = np.where(np.array(look_at_otherlever_or_not_merge['scorch'])==1)
    time_point_lookatotherlever2 = look_at_otherlever_or_not_merge["time_in_second"][ind_lookatotherlever2]
    
    #
    time_point_lookatotherlever1 = np.round(time_point_lookatotherlever1,1)
    time_point_lookatotherlever1 = np.unique(np.sort(time_point_lookatotherlever1))
    time_point_lookatotherlever2 = np.round(time_point_lookatotherlever2,1)
    time_point_lookatotherlever2 = np.unique(np.sort(time_point_lookatotherlever2))
    
  

    output_time_points_otherlever = {"time_point_lookatotherlever1":time_point_lookatotherlever1,"time_point_lookatotherlever2":time_point_lookatotherlever2}
    
    return output_time_points_otherlever
