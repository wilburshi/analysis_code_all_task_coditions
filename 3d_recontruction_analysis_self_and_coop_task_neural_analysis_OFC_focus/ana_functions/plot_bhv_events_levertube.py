#  function - plot behavioral events

def plot_bhv_events_levertube(date_tgt, animal1, animal2, session_start_time, session_plot_time, time_point_pull1, time_point_pull2, oneway_gaze1, oneway_gaze2, mutual_gaze1, mutual_gaze2, timepoint_lever1, timepoint_lever2, timepoint_tube1, timepoint_tube2):
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy
    import string
    import warnings
    import pickle    

    fig, axs = plt.subplots(2,1)
    fig.set_figheight(5)
    fig.set_figwidth(25)
    # plot for animal 1
    ind_plot = time_point_pull1 < (session_plot_time-session_start_time)
    #for itime in np.arange(0,720,1):
    #    plt.plot([itime,itime],[0,1],linewidth = 2.0,color=(0.5,0.5,0.5))
    for itime in time_point_pull1[ind_plot]:
        line1, = axs[0].plot([itime,itime],[0,1],linewidth = 2.0,color=(0.2,0.2,0.2),label = 'lever pull')
    try:
        for itime in oneway_gaze1:
            line2, = axs[0].plot([itime,itime],[0,1],linewidth = 2.0,color='r',label = 'one-way gaze')  
    except:
        print("no oneway gaze "+animal1)
    try:
        for itime in mutual_gaze1:
            line3, = axs[0].plot([itime,itime],[0,1],linewidth = 2.0,color='b',label = 'mutual gaze')  
    except:
        print("no mutual gaze "+animal1)
    try:
        for itime in timepoint_lever1:
            line4, = axs[0].plot([itime,itime],[0,1],linewidth = 2.0,color='g',label = 'look at self lever')  
    except:
        print("no self lever gaze "+animal1)
    try:
        for itime in timepoint_tube1:
            line5, = axs[0].plot([itime,itime],[0,1],linewidth = 2.0,color='y',label = 'look at self tube')  
    except:
        print("no self tube gaze "+animal1)
    
    axs[0].set_title(date_tgt+' '+animal1,fontsize = 18)
    axs[0].set_xlim([-10,session_plot_time+10])
    axs[0].set_xlabel("")
    axs[0].set_xticklabels("")
    axs[0].set_yticklabels("")
    plt.rc('legend', fontsize = 13)
    try:
        axs[0].legend(handles=[line1,line2,line3,line4,line5], fontsize = 13)
    except:
        try: 
            axs[0].legend(handles=[line1,line2,line4,line5], fontsize = 13)
        except:
            axs[0].legend(handles=[line1,line4,line5], fontsize = 13)    

    # plot for animal 2
    ind_plot = time_point_pull2 < (session_plot_time-session_start_time)
    #for itime in np.arange(0,720,1):
    #    plt.plot([itime,itime],[0,1],linewidth = 2.0,color=(0.5,0.5,0.5))
    for itime in time_point_pull2[ind_plot]:
        line1, = axs[1].plot([itime,itime],[0,1],linewidth = 2.0,color=(0.2,0.2,0.2))
    try:
        for itime in oneway_gaze2:
            line2, = axs[1].plot([itime,itime],[0,1],linewidth = 2.0,color='r')    
    except:
        print("no oneway gaze"+animal2)
    try:
        for itime in mutual_gaze2:
            line3, = axs[1].plot([itime,itime],[0,1],linewidth = 2.0,color='b')    
    except:
        print("no mutual gaze"+animal2)
    try:
        for itime in timepoint_lever2:
            line4, = axs[1].plot([itime,itime],[0,1],linewidth = 2.0,color='g',label = 'look at self lever')  
    except:
        print("no self lever gaze "+animal2)
    try:
        for itime in timepoint_tube2:
            line5, = axs[1].plot([itime,itime],[0,1],linewidth = 2.0,color='y',label = 'look at self tube')  
    except:
        print("no self tube gaze "+animal2)
    axs[1].set_title(date_tgt+' '+animal2,fontsize = 18)
    axs[1].set_xlim([-10,session_plot_time+10])
    axs[1].set_xlabel("time/s",fontsize = 19)
    axs[1].set_yticklabels("")
    axs[1].tick_params(labelsize = 15)

