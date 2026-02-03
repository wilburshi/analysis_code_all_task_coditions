#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 23:00:13 2024

@author: jadi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def simulateCoopDyad(noise):
    # Define the dynamical system 
    A = np.random.randn(6,6)
    T = .1
    A = np.array([[-T,0,0,1,0,1], #Gaze_1
                  [0,-T,1,0,0,0], #Gaze_2
                  [1,0,-T,0,1,0], #Pull_1
                  [0,1,0,-T,0,0], #Pull_2
                  [0,1,0,-4,-T,0], #Chirp_1
                  [0,0,0,1,0,-T]]) #Chirp_2
    
    
    # Initialize the system
    Y0 = np.random.randint(0,2,6)  # Initial value for x
    num_steps = 10000  # Number of time steps to simulate
    
    # Create arrays to store the results
    Y_values = np.zeros([6,num_steps + 1])
    
    # Set initial values
    Y_values[:,0] = Y0
    
    # Simulate a noisy system
    for t in range(num_steps):
        Th = 1
        y = np.matmul(A,Y_values[:,t])
        y[y>=Th] = 1
        y[y<Th] = 0
        
        #randomly flip a few bits
        # Define the fraction of bits to flip
        fraction_to_flip = noise  # For example, 30% of the bits
    
        # Calculate the number of bits to flip
        num_bits = len(y)
        num_flips = int(np.ceil(fraction_to_flip * num_bits))
        
        # Randomly select indices to flip
        indices_to_flip = np.random.choice(num_bits, num_flips, replace=False)
        
        # Flip the selected bits
        flipped_y = np.copy(y)
        flipped_y[indices_to_flip] = 1 - flipped_y[indices_to_flip]
            
        Y_values[:,t + 1] = flipped_y
         
    # Create a DataFrame to store the results
    df = pd.DataFrame({
        'Time': np.arange(num_steps),
        'G1_0': Y_values[0,:-1],
        'P1_0': Y_values[2,:-1],
        'Ch1_0': Y_values[4,:-1],
        'G2_0': Y_values[1,:-1],
        'P2_0': Y_values[3,:-1],
        'Ch2_0': Y_values[5,:-1],
        'G1_1': Y_values[0,1:],
        'P1_1': Y_values[2,1:],
        'Ch1_1': Y_values[4,1:],
        'G2_1': Y_values[1,1:],
        'P2_1': Y_values[3,1:],
        'Ch2_1': Y_values[5,1:]
    })
    
    # Save the DataFrame to a CSV file
    df.to_csv('simulated_dyad.csv', index=False)
    
    var_names = df.columns
    behVarName = ['Gaze_1', 'Pull_1', 'Chirp_1', 'Gaze_2', 'Pull_2', 'Chirp_2']
    behVarClr = ['red', 'green', 'blue', 'orange', 'cyan', 'magenta']
    # plot
    fig,ax = plt.subplots(1)
    
    for b in range(6):
        var = var_names[7+b]
        viewdata = df[var].values[80:100]
        ax.vlines(np.where(viewdata ==1)[0], b+0.7, b+1.1, colors=behVarClr[b], label=behVarName[b])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_yticks(np.arange(0,6,1) + 1)
    ax.set_yticklabels(behVarName)

    # Optionally, you can also move the left and bottom spines
    # to make the plot look like a traditional "box off" plot in MATLAB
    ax.spines['left'].set_position(('outward', 5))
    ax.spines['bottom'].set_position(('outward', 5))
    
    return