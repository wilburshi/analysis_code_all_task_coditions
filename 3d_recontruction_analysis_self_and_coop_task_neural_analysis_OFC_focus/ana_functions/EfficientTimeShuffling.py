#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 21:33:44 2020

@author: ad2422
"""
import random
import re
import os
import sys
import pandas as pd
import numpy as np
import math
import itertools
from itertools import groupby

def GroupColumns(data, seed=1990):
    '''
    Assumes last two characters of the column name give the time stamp
    https://stackoverflow.com/questions/37568763/python-group-a-list-into-sublists-by-a-equality-of-projected-value
    data: dataframe
    returns:
        list1 , a list of lists, each sublist consists of columns which are the same neural populations recorded at different time points
    '''
    def projection(val):
        return val[:-2]
    if 'Unnamed: 0' in data.columns :
        data.drop('Unnamed: 0', axis=1,inplace= True)
    l1=list(data.columns)
    x_sorted = sorted(l1, key=projection)
    x_grouped = [list(it) for k, it in groupby(x_sorted, projection)] 
    return x_grouped


def EfficientShuffle(df, seed=1990):
    '''
    input:
        df; the neural data binned and organized into time slices
        https://stackoverflow.com/questions/10483377/time-complexity-of-random-sample
        The algorithm given is select, and it takes O(N) time. https://www.tandfonline.com/doi/pdf/10.1080/00207168408803438?needAccess=true 
    returns:
        Groups the columns according to variables which are the same population in different time slices, then time shuffles.
        returns  a pair of dataframes, first gives the time shuffled dataset, the second gives the keys on how the time shuffling occurs
    
    '''
    numrows=df.shape[0]
    dfs,dfkeys=pd.DataFrame(columns=df.columns, index= list(range(df.shape[0]))), pd.DataFrame(columns=df.columns, index= list(range(df.shape[0])))
    listgrp=GroupColumns(df)
    for listgr in listgrp:
        frames, indkey=list(),list()
        for v in listgr:
            frames.append(df[v])
            indkey.append(v)
        result = pd.concat(frames, keys=indkey)
        ser1= result.sample(frac=1,replace=False)
        ser2=[u for u in ser1.index]
        for i,v in enumerate(listgr):
            ser1n= ser1[(i)*numrows:(i+1)*numrows]
            ser1n.reset_index(inplace=True, drop=True)
            dfs[v]=ser1n
            ser2n= pd.Series(ser2[(i)*numrows:(i+1)*numrows])
            dfkeys[v]=ser2n
    return(dfs,dfkeys)
        

