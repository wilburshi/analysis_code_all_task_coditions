# function - plot inter-pull interval

def plot_interpull_interval(animal1, animal2, time_point_pull1, time_point_pull2):

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy
    import string
    import warnings
    import pickle

    time_point_pull1_n0 = time_point_pull1.reset_index(drop = True)[0:time_point_pull1.shape[0]-1]
    time_point_pull1_n1 = time_point_pull1.reset_index(drop = True)[1:time_point_pull1.shape[0]]
    time_point_pull2_n0 = time_point_pull2.reset_index(drop = True)[0:time_point_pull2.shape[0]-1]
    time_point_pull2_n1 = time_point_pull2.reset_index(drop = True)[1:time_point_pull2.shape[0]]
    ipi_1 = time_point_pull1_n1.reset_index(drop = True) - time_point_pull1_n0.reset_index(drop = True)
    ipi_2 = time_point_pull2_n1.reset_index(drop = True) - time_point_pull2_n0.reset_index(drop = True)
    #
    fig2, axs2 = plt.subplots(1,2)
    fig2.set_figheight(5)
    fig2.set_figwidth(20)
    axs2[0].hist(ipi_1, alpha=0.5, bins=np.arange(0,25,1))
    axs2[0].set_title(animal1)
    axs2[1].hist(ipi_2, alpha=0.5, bins=np.arange(0,25,1))
    axs2[1].set_title(animal2)
    print('animal1 median',int(np.median(ipi_1)),'s')
    print('animal2 median',int(np.median(ipi_2)),'s')
