#  function - plot behavioral events

def plot_bhv_events_gaze_start_stop(gaze_thresold, date_tgt, animal1, animal2, session_start_time, session_plot_time, time_point_pull1, time_point_pull2, oneway_gaze1, oneway_gaze2, mutual_gaze1, mutual_gaze2):
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy
    import string
    import warnings
    import pickle    

    animal1_gaze = np.concatenate([oneway_gaze1, mutual_gaze1])
    animal1_gaze = np.sort(np.unique(animal1_gaze))
    animal1_gaze_stop = animal1_gaze[np.concatenate(((animal1_gaze[1:]-animal1_gaze[0:-1]>gaze_thresold)*1,[1]))==1]
    animal1_gaze_start = np.concatenate(([animal1_gaze[0]],animal1_gaze[np.where(animal1_gaze[1:]-animal1_gaze[0:-1]>gaze_thresold)[0]+1]))
    animal1_gaze_flash = np.intersect1d(animal1_gaze_start, animal1_gaze_stop)
    animal1_gaze_start = animal1_gaze_start[~np.isin(animal1_gaze_start,animal1_gaze_flash)]
    animal1_gaze_stop = animal1_gaze_stop[~np.isin(animal1_gaze_stop,animal1_gaze_flash)]
    #
    animal2_gaze = np.concatenate([oneway_gaze2, mutual_gaze2])
    animal2_gaze = np.sort(np.unique(animal2_gaze))
    animal2_gaze_stop = animal2_gaze[np.concatenate(((animal2_gaze[1:]-animal2_gaze[0:-1]>gaze_thresold)*1,[1]))==1]
    animal2_gaze_start = np.concatenate(([animal2_gaze[0]],animal2_gaze[np.where(animal2_gaze[1:]-animal2_gaze[0:-1]>gaze_thresold)[0]+1]))
    animal2_gaze_flash = np.intersect1d(animal2_gaze_start, animal2_gaze_stop)
    animal2_gaze_start = animal2_gaze_start[~np.isin(animal2_gaze_start,animal2_gaze_flash)]
    animal2_gaze_stop = animal2_gaze_stop[~np.isin(animal2_gaze_stop,animal2_gaze_flash)] 

    fig, axs = plt.subplots(2,1)
    fig.set_figheight(5)
    fig.set_figwidth(25)
    # plot for animal 1
    ind_plot = time_point_pull1 < (session_plot_time - session_start_time)
    #for itime in np.arange(0,720,1):
    #    plt.plot([itime,itime],[0,1],linewidth = 2.0,color=(0.5,0.5,0.5))
    for itime in time_point_pull1[ind_plot]:
        line1, = axs[0].plot([itime,itime],[0,1],linewidth = 2.0,color=(0.2,0.2,0.2),label = 'lever pull')
    try:
        for itime in animal1_gaze_start:
            line2, = axs[0].plot([itime,itime],[0,1],linewidth = 2.0,color=(0.0,0.0,0.7),label = 'gaze start')  
    except:
        print("no gaze start")
    try:
        for itime in animal1_gaze_stop:
            line3, = axs[0].plot([itime,itime],[0,1],linewidth = 2.0,color=(0.0,0.7,0.0),label = 'gaze stop')  
    except:
        print("no gaze stop")
    try:
        for itime in animal1_gaze_flash:
            line4, = axs[0].plot([itime,itime],[0,1],linewidth = 2.0,color=(0.7,0.0,0.0),label = 'gaze flash')  
    except:
        print("no gaze flash")
    axs[0].set_title(date_tgt+' '+animal1,fontsize = 18)
    axs[0].set_xlim([-10,session_plot_time+10])
    axs[0].set_xlabel("")
    axs[0].set_xticklabels("")
    axs[0].set_yticklabels("")
    plt.rc('legend', fontsize = 13)
    try:
        axs[0].legend(handles=[line1,line2,line3,line4], fontsize = 13)
    except:
        try: 
            axs[0].legend(handles=[line1,line2,line4], fontsize = 13)
        except:
            axs[0].legend(handles=[line1,line4], fontsize = 13)    

    # plot for animal 2
    ind_plot = time_point_pull2 < (session_plot_time - session_start_time)
    #for itime in np.arange(0,720,1):
    #    plt.plot([itime,itime],[0,1],linewidth = 2.0,color=(0.5,0.5,0.5))
    for itime in time_point_pull2[ind_plot]:
        line1, = axs[1].plot([itime,itime],[0,1],linewidth = 2.0,color=(0.2,0.2,0.2))
    try:
        for itime in animal2_gaze_start:
            line2, = axs[1].plot([itime,itime],[0,1],linewidth = 2.0,color=(0.0,0.0,0.7))    
    except:
        print("no gaze start")
    try:
        for itime in animal2_gaze_stop:
            line3, = axs[1].plot([itime,itime],[0,1],linewidth = 2.0,color=(0.0,0.7,0.0))    
    except:
        print("no gaze stop")
    try:
        for itime in animal2_gaze_flash:
            line4, = axs[1].plot([itime,itime],[0,1],linewidth = 2.0,color=(0.7,0.0,0.0))  
    except:
        print("no gaze flash")
    axs[1].set_title(date_tgt+' '+animal2,fontsize = 18)
    axs[1].set_xlim([-10,session_plot_time+10])
    axs[1].set_xlabel("time/s",fontsize = 19)
    axs[1].set_yticklabels("")
    axs[1].tick_params(labelsize = 15)

