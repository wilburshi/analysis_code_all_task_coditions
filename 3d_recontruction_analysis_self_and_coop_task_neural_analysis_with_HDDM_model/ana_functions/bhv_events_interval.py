# function - interval between all behavioral events

#####################################
def bhv_events_interval(totalsess_time, session_start_time, time_point_pull1, time_point_pull2, oneway_gaze1, oneway_gaze2, mutual_gaze1, mutual_gaze2):
    
	import pandas as pd
	import numpy as np
	import matplotlib.pyplot as plt
	import scipy
	import string
	import warnings
	import pickle

	total_time = int((totalsess_time))
	time_point_pull1_round = time_point_pull1.reset_index(drop = True)
	time_point_pull1_round = time_point_pull1_round[time_point_pull1_round<total_time]
	time_point_pull2_round  = time_point_pull2.reset_index(drop = True)
	time_point_pull2_round = time_point_pull2_round[time_point_pull2_round<total_time]
	time_point_onewaygaze1_round = pd.Series(oneway_gaze1).reset_index(drop = True)
	time_point_onewaygaze2_round = pd.Series(oneway_gaze2).reset_index(drop = True)
	time_point_mutualgaze1_round = pd.Series(mutual_gaze1).reset_index(drop = True)
	time_point_mutualgaze2_round = pd.Series(mutual_gaze2).reset_index(drop = True)
	time_point_onewaygaze1_round = time_point_onewaygaze1_round[(time_point_onewaygaze1_round>0)&(time_point_onewaygaze1_round<total_time)]
	time_point_onewaygaze2_round = time_point_onewaygaze2_round[(time_point_onewaygaze2_round>0)&(time_point_onewaygaze2_round<total_time)]
	time_point_mutualgaze1_round = time_point_mutualgaze1_round[(time_point_mutualgaze1_round>0)&(time_point_mutualgaze1_round<total_time)]
	time_point_mutualgaze2_round = time_point_mutualgaze2_round[(time_point_mutualgaze2_round>0)&(time_point_mutualgaze2_round<total_time)]
	#     
	time_point_bhv_events = time_point_pull1_round
	time_point_bhv_events = time_point_bhv_events.append(time_point_pull2_round)
	time_point_bhv_events = time_point_bhv_events.append(time_point_mutualgaze1_round)
	time_point_bhv_events = time_point_bhv_events.append(time_point_mutualgaze2_round)
	time_point_bhv_events = time_point_bhv_events.append(time_point_onewaygaze1_round)
	time_point_bhv_events = time_point_bhv_events.append(time_point_onewaygaze2_round)

	time_point_bhv_events = time_point_bhv_events.reset_index(drop=True)
	time_point_bhv_events = np.sort(time_point_bhv_events)
	nevents = np.shape(time_point_bhv_events)[0]
	bhv_events_interval = time_point_bhv_events[1:nevents]-time_point_bhv_events[0:nevents-1]

	pull1_num = np.shape(time_point_pull1_round)[0]
	other_to_pull1_interval = np.zeros((1,pull1_num))[0]
	other_to_pull1_interval[:] = np.nan
	pull1_to_other_interval = np.zeros((1,pull1_num))[0]
	pull1_to_other_interval[:] = np.nan
	#
	time_point_except_pull1 = time_point_pull2_round
	time_point_except_pull1 = time_point_except_pull1.append(time_point_mutualgaze1_round)
	time_point_except_pull1 = time_point_except_pull1.append(time_point_mutualgaze2_round)
	time_point_except_pull1 = time_point_except_pull1.append(time_point_onewaygaze1_round)
	time_point_except_pull1 = time_point_except_pull1.append(time_point_onewaygaze2_round)
	time_point_except_pull1 = time_point_except_pull1.reset_index(drop=True)
	time_point_except_pull1 = np.sort(time_point_except_pull1)
	#
	for ipull1 in np.arange(0,pull1_num,1):

		aa = np.array(time_point_pull1_round)[ipull1]-time_point_except_pull1
		try:
		    other_to_pull1_interval[ipull1] = np.nanmin(aa[aa>0])
		except:
		    other_to_pull1_interval[ipull1] = np.nan
		try:
		    pull1_to_other_interval[ipull1] = abs(np.nanmax(aa[aa<0])) 
		except:
		    pull1_to_other_interval[ipull1] = np.nan
	# # 
	pull2_num = np.shape(time_point_pull2_round)[0]
	other_to_pull2_interval = np.zeros((1,pull2_num))[0]
	other_to_pull2_interval[:] = np.nan
	pull2_to_other_interval = np.zeros((1,pull2_num))[0]
	pull2_to_other_interval[:] = np.nan
	#
	time_point_except_pull2 = time_point_pull1_round
	time_point_except_pull2 = time_point_except_pull2.append(time_point_mutualgaze1_round)
	time_point_except_pull2 = time_point_except_pull2.append(time_point_mutualgaze2_round)
	time_point_except_pull2 = time_point_except_pull2.append(time_point_onewaygaze1_round)
	time_point_except_pull2 = time_point_except_pull2.append(time_point_onewaygaze2_round)
	time_point_except_pull2 = time_point_except_pull2.reset_index(drop=True)
	time_point_except_pull2 = np.sort(time_point_except_pull2)
	#
	for ipull2 in np.arange(0,pull2_num,1):

		aa = np.array(time_point_pull2_round)[ipull2]-time_point_except_pull2
		try:
		    other_to_pull2_interval[ipull2] = np.nanmin(aa[aa>0])
		except:
		    other_to_pull2_interval[ipull2] = np.nan
		try:
		    pull2_to_other_interval[ipull2] = abs(np.nanmax(aa[aa<0]))
		except:
		    pull2_to_other_interval[ipull2] = np.nan
	# #    
	other_to_pull_interval = np.concatenate((other_to_pull1_interval,other_to_pull2_interval))
	pull_to_other_interval = np.concatenate((pull1_to_other_interval,pull2_to_other_interval))



	Q1 = np.quantile(bhv_events_interval,0.25)
	Q2 = np.quantile(bhv_events_interval,0.5)
	Q3 = np.quantile(bhv_events_interval,0.75)
	low_lim = Q1 - 1.5 * (Q3-Q1)
	up_lim = Q3 + 1.5 * (Q3-Q1)
	# low_lim = Q1
	# up_lim = Q3

	low_lim = np.round(low_lim*10)/10
	up_lim = np.round(up_lim*10)/10

	if low_lim < 0.1:
		low_lim = 0.1
	if up_lim <0.2:
		# up_lim = 0.2
		up_lim = 1  
	# if up_lim < 1:
	#     up_lim = np.max(bhv_events_interval)/2
	if up_lim > 2:
		up_lim = 2

	return low_lim, up_lim, bhv_events_interval, pull_to_other_interval, other_to_pull_interval




#####################################
def bhv_events_interval_certainEdges(totalsess_time, session_start_time, time_point_pull1, time_point_pull2, oneway_gaze1, oneway_gaze2, mutual_gaze1, mutual_gaze2):
    
	import pandas as pd
	import numpy as np
	import matplotlib.pyplot as plt
	import scipy
	import string
	import warnings
	import pickle

	# initiate
	all_pull_edges_intervals = {}

	# optional - conbine mutual gaze and one way gaze
	oneway_gaze1 = np.sort(np.concatenate((oneway_gaze1,mutual_gaze1)))
	oneway_gaze2 = np.sort(np.concatenate((oneway_gaze2,mutual_gaze2)))
		
	total_time = int((totalsess_time))
	time_point_pull1_round = time_point_pull1.reset_index(drop = True)
	time_point_pull1_round = time_point_pull1_round[time_point_pull1_round<total_time]
	time_point_pull2_round  = time_point_pull2.reset_index(drop = True)
	time_point_pull2_round = time_point_pull2_round[time_point_pull2_round<total_time]
	time_point_onewaygaze1_round = pd.Series(oneway_gaze1).reset_index(drop = True)
	time_point_onewaygaze2_round = pd.Series(oneway_gaze2).reset_index(drop = True)
	time_point_mutualgaze1_round = pd.Series(mutual_gaze1).reset_index(drop = True)
	time_point_mutualgaze2_round = pd.Series(mutual_gaze2).reset_index(drop = True)
	time_point_onewaygaze1_round = time_point_onewaygaze1_round[(time_point_onewaygaze1_round>0)&(time_point_onewaygaze1_round<total_time)]
	time_point_onewaygaze2_round = time_point_onewaygaze2_round[(time_point_onewaygaze2_round>0)&(time_point_onewaygaze2_round<total_time)]
	time_point_mutualgaze1_round = time_point_mutualgaze1_round[(time_point_mutualgaze1_round>0)&(time_point_mutualgaze1_round<total_time)]
	time_point_mutualgaze2_round = time_point_mutualgaze2_round[(time_point_mutualgaze2_round>0)&(time_point_mutualgaze2_round<total_time)]

	pull1_num = np.shape(time_point_pull1_round)[0]
	pull2_num = np.shape(time_point_pull2_round)[0]

	# pull1 <-> pull2
	# pull1 -> pull2
	# pull2 -> pull1
	if pull1_num < pull2_num:
		pull_num_less = pull1_num
	else:
		pull_num_less = pull2_num
	#
	pull_to_pull_interval = np.zeros((1,pull_num_less))[0]
	pull_to_pull_interval[:] = np.nan
	pull1_to_pull2_interval = np.zeros((1,pull_num_less))[0]
	pull1_to_pull2_interval[:] = np.nan
	pull2_to_pull1_interval = np.zeros((1,pull_num_less))[0]
	pull2_to_pull1_interval[:] = np.nan
	# 
	for ipull in np.arange(0,pull_num_less,1):
		#
		if pull1_num < pull2_num:
			aa = np.array(time_point_pull1_round)[ipull]-time_point_pull2_round
			try:
				pull2_to_pull1_interval[ipull] = np.nanmin(aa[aa>0])
			except:
				pull2_to_pull1_interval[ipull] = np.nan
			try:
				pull1_to_pull2_interval[ipull] = abs(np.nanmax(aa[aa<0]))
			except:
				pull1_to_pull2_interval[ipull] = np.nan
		else: 
			aa = np.array(time_point_pull2_round)[ipull]-time_point_pull1_round
			try:
				pull1_to_pull2_interval[ipull] = np.nanmin(aa[aa>0])
			except:
				pull1_to_pull2_interval[ipull] = np.nan
			try:
				pull2_to_pull1_interval[ipull] = abs(np.nanmax(aa[aa<0]))
			except:
				pull2_to_pull1_interval[ipull] = np.nan
		try:
		    pull_to_pull_interval[ipull] = np.nanmin(abs(aa))
		except:
		    pull_to_pull_interval[ipull] = np.nan

		               
	# pull1 -> gaze1
	# gaze1 -> pull1
	pull1_to_gaze1_interval = np.zeros((1,pull1_num))[0]
	pull1_to_gaze1_interval[:] = np.nan
	gaze1_to_pull1_interval = np.zeros((1,pull1_num))[0]
	gaze1_to_pull1_interval[:] = np.nan
	for ipull in np.arange(0,pull1_num,1):
		aa = np.array(time_point_pull1_round)[ipull]-time_point_onewaygaze1_round
		try:
		    gaze1_to_pull1_interval[ipull] = np.nanmin(aa[aa>0])
		except:
		    gaze1_to_pull1_interval[ipull] = np.nan
		try:
		    pull1_to_gaze1_interval[ipull] = abs(np.nanmax(aa[aa<0]))
		except:
		    pull1_to_gaze1_interval[ipull] = np.nan
		
	# pull2 -> gaze2
	# gaze2 -> pull2
	pull2_to_gaze2_interval = np.zeros((1,pull2_num))[0]
	pull2_to_gaze2_interval[:] = np.nan
	gaze2_to_pull2_interval = np.zeros((1,pull2_num))[0]
	gaze2_to_pull2_interval[:] = np.nan
	for ipull in np.arange(0,pull2_num,1):
		aa = np.array(time_point_pull2_round)[ipull]-time_point_onewaygaze2_round
		try:
		    gaze2_to_pull2_interval[ipull] = np.nanmin(aa[aa>0])
		except:
		    gaze2_to_pull2_interval[ipull] = np.nan
		try:
		    pull2_to_gaze2_interval[ipull] = abs(np.nanmax(aa[aa<0]))
		except:
		    pull2_to_gaze2_interval[ipull] = np.nan    
		    
	# pull1 -> gaze2
	# gaze2 -> pull1
	pull1_to_gaze2_interval = np.zeros((1,pull1_num))[0]
	pull1_to_gaze2_interval[:] = np.nan
	gaze2_to_pull1_interval = np.zeros((1,pull1_num))[0]
	gaze2_to_pull1_interval[:] = np.nan
	for ipull in np.arange(0,pull1_num,1):
		aa = np.array(time_point_pull1_round)[ipull]-time_point_onewaygaze2_round
		try:
		    gaze2_to_pull1_interval[ipull] = np.nanmin(aa[aa>0])
		except:
		    gaze2_to_pull1_interval[ipull] = np.nan
		try:
		    pull1_to_gaze2_interval[ipull] = abs(np.nanmax(aa[aa<0]))
		except:
		    pull1_to_gaze2_interval[ipull] = np.nan

	# pull2 -> gaze1
	# gaze1 -> pull2
	pull2_to_gaze1_interval = np.zeros((1,pull2_num))[0]
	pull2_to_gaze1_interval[:] = np.nan
	gaze1_to_pull2_interval = np.zeros((1,pull2_num))[0]
	gaze1_to_pull2_interval[:] = np.nan
	for ipull in np.arange(0,pull2_num,1):
		aa = np.array(time_point_pull2_round)[ipull]-time_point_onewaygaze1_round
		try:
		    gaze1_to_pull2_interval[ipull] = np.nanmin(aa[aa>0])
		except:
		    gaze1_to_pull2_interval[ipull] = np.nan
		try:
		    pull2_to_gaze1_interval[ipull] = abs(np.nanmax(aa[aa<0]))
		except:
		    pull2_to_gaze1_interval[ipull] = np.nan        
		    
	#        
	all_pull_edges_intervals = {'pull_to_pull_interval':pull_to_pull_interval,
		                          'pull1_to_pull2_interval':pull1_to_pull2_interval,
		                          'pull2_to_pull1_interval':pull2_to_pull1_interval,
		                          'pull1_to_gaze1_interval':pull1_to_gaze1_interval,
		                          'gaze1_to_pull1_interval':gaze1_to_pull1_interval,
		                          'pull2_to_gaze2_interval':pull2_to_gaze2_interval,
		                          'gaze2_to_pull2_interval':gaze2_to_pull2_interval,
		                          'pull1_to_gaze2_interval':pull1_to_gaze2_interval,
		                          'gaze2_to_pull1_interval':gaze2_to_pull1_interval,
		                          'pull2_to_gaze1_interval':pull2_to_gaze1_interval,
		                          'gaze1_to_pull2_interval':gaze1_to_pull2_interval,
		                         }
	
	return all_pull_edges_intervals
