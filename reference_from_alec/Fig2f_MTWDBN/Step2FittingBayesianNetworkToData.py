#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 18:46:50 2020
@author: ad2422
python codes/Python Codes/FloatingBins is where the TestFittingADifferentBasisTimeBinForEachConn.py tests this code 
"""


from itertools import permutations
import random
import itertools
import networkx as nx
import pandas as pd
import numpy as np
from functools import reduce
from networkx.algorithms import bipartite
from math import log
from pgmpy.estimators import StructureEstimator, K2Score
from pgmpy.base import DAG
from pgmpy.estimators import HillClimbSearch, BicScore

class HillClimbSearch5(HillClimbSearch):
    def __init__(self, data, scoring_method=None, hardtabu=[], **kwargs):
        """
        Class for heuristic hill climb searches for DAGs, to learn
        network structure from data. `estimate` attempts to find a model with optimal score.
        Differences of scores less than 10^-8 are ignored, since computers can make such errors
        Parameters
        ----------
        data: pandas DataFrame object
            datafame object where each column represents one variable.
            (If some values in the data are missing the data cells should be set to `numpy.NaN`.
            Note that pandas converts each column containing `numpy.NaN`s to dtype `float`.)

        scoring_method: Instance of a `StructureScore`-subclass (`K2Score` is used as default)
            An instance of `K2Score`, `BdeuScore`, or `BicScore`.
            This score is optimized during structure estimation by the `estimate`-method.
        
        hardtabu= a set of directed edges which cannot be added to the estimated DAG
        
        state_names: dict (optional)
            A dict indicating, for each variable, the discrete set of states (or values)
            that the variable can take. If unspecified, the observed values in the data set
            are taken to be the only possible states.

        complete_samples_only: bool (optional, default `True`)
            Specifies how to deal with missing data, if present. If set to `True` all rows
            that contain `np.Nan` somewhere are ignored. If `False` then, for each variable,
            every row where neither the variable nor its parents are `np.NaN` is used.
            This sets the behavior of the `state_count`-method.
        """
        self.hardtabu=  hardtabu
        self.scoring_method = scoring_method
        super(HillClimbSearch5, self).__init__(data,scoring_method, **kwargs)


    def get_layer(self,timelayers, node):
        '''
        Helper function called by RestrictedEstimatetl
        Parameters
        ----------
        node: A string which is the name of the node
        timelayers: int
            The number of time slices that is being considered
        Returns
        -------
        the time layer to which node belongs to
        '''        
        for i in range(1, timelayers +1 ):
            reqsuf ='t'+str(i)
            statement="nodes"+str(i)+"= list()"
            exec(statement)
            if node.endswith(reqsuf):
                return i
    
    def get_all_layers(self,timelayers, nodes):
        '''
        Helper function called by RestrictedEstimatetl
        Parameters
        ----------
        nodes: object of class 'dict_keys' obtained from self.state_names.keys() which gives the names of the nodes.
                For example dict_keys(['V1_t1', 'V2_t1', 'V3_t1', 'V4_t1', 'V5_t1', 'V1_t2', 'V2_t2', 'V3_t2', 'V4_t2', 
                'V5_t2', 'V1_t3', 'V2_t3', 'V3_t3', 'V4_t3', 'V5_t3', 'V1_t4', 'V2_t4', 'V3_t4', 'V4_t4', 'V5_t4'])
                Other iterables like simple lists also work fine.
        timelayers: int
            The number of time slices that is being considered            
        Returns
        -------
        a list of lists of lngth timelayers , the i-th lists (whose  python list index is i-1) contains the names of the neurons\
        in timelayer i.
        '''
        final_result=list()
        for i in range(1, timelayers +1 ):
            reqsuf ='t'+str(i)
            statement="nodes"+str(i)+"= list()"
            exec(statement)
            for node in nodes:
                if node.endswith(reqsuf):
                    statement="nodes"+str(i)+".append(node)"
                    exec(statement)
            statement="final_result.append("+"nodes"+str(i)+")"
            exec(statement)
        return(final_result)


    def IssametypeofNode(self, node1, node2):
        '''
        Helper function called by RestrictedEstimatetl
        Parameters
        ----------
        node1, node2 are strings which give the names of vertexes, viz. node1= 'V_lyr1_spk2_t3'
        Returns
        -------
        True if the two nodes are of the same type , i.e. both excitatory. so node1= 'V_lyr1_spk2_t3' and node2= 'V_lyr2_spk2_t3' gives true, but
             node1= 'V_lyr1_spk2_t3' and node2= 'V_lyr1_spk1_t3' gives False
        '''
        if "spk1" in node1:
            if "spk1" in node2:
                return True
        if "spk2" in node1:
            if "spk2" in node2:
                return True
        return False

    def IsInSameCorticalLayer(self, node1, node2):
        '''
        Helper function called by RestrictedEstimatetl
        Parameters
        ----------
        node1, node2 are strings which give the names of vertexes, viz. node1= 'V_lyr1_spk2_t3'
        Returns
        -------
        True iff the two nodes are in the same cortical. so node1= 'V_lyr1_spk2_t3' and node2= 'V_lyr2_spk2_t3' gives False, but
             node1= 'V_lyr1_spk2_t3' and node2= 'V_lyr1_spk1_t3' gives True
        '''
        if "lyr1" in node1:
            if "lyr1" in node2:
                return True
        if "lyr2" in node1:
            if "lyr2" in node2:
                return True
        if "lyr3" in node1:
            if "lyr3" in node2:
                return True
        return False 

    
 
    def legal_operations2(self, model, timelayers, tabu_list=[], max_indegree=None):
        """
        Helper function called by RestrictedEstimatetl
        Parameters
        ----------
        model: A legal DAG (G). This means inter time-layer neurons are only connected from earlier time periods to the last time layer.
                Also intra-layer connections are only present in the last timelayer.
        timelayers: int
            The number of time slices that is being considered
        tabu_list: a list. Operations which are prohibted, or cannot be made
        Returns
        -------
        Basic operations are node additions, node deletions and node reverses. All basic operations that 
        a. turn G into another legal DAG g', AND
        b. Are not on the tabu list
        are returned, along with returned operation the score change resulting from its application is also returned.
        
        model given must be a legal DAG for this approach to work.
        Useful
        -------
        https://stackoverflow.com/questions/30773911/union-of-multiple-sets-in-python
        https://www.geeksforgeeks.org/python-map-function/
        https://stackoverflow.com/questions/12935194/combinations-between-two-lists
        set ops - : https://www.geeksforgeeks.org/python-set-operations-union-intersection-difference-symmetric-difference/
        """

        local_score = self.scoring_method.local_score
        nodes = self.state_names.keys()
        '''
        adding edges within layers, no such edges unless we are deaLing with the last time layer
        '''
        all_layers=self.get_all_layers(timelayers, nodes)
        potential_new_edges1= set(permutations(all_layers[timelayers-1], 2))
#        print(potential_new_edges1)
        '''
        adding edges between layers, only edges going forward in time to the final time layer are acceptable
        '''    
        def condition(s1,s2): # computes the union of the sets
            return s1.union(s2)
        someothersets=list()
        def crosslayeredges(x,y):
            return set(itertools.product(x, y))
        for j in range(0, timelayers-1):
            someothersets.append(crosslayeredges(all_layers[j],all_layers[timelayers-1]))
        # UNION of all sets in the list someothersets        
        potential_new_edges2= reduce( condition,someothersets)
        
        '''
        Adding all the potential edges which may be added and puting them in a set, same for reversible edges
        '''
        potential_new_edges= potential_new_edges1.union(potential_new_edges2)
        potential_new_edges = ( potential_new_edges -
                               set(model.edges()) -
                               set([(Y, X) for (X, Y) in model.edges()]))
        reversibleedges= list(set(model.edges()) - potential_new_edges2 )
        
        
        for (X, Y) in potential_new_edges:  # (1) add single edge
            if nx.is_directed_acyclic_graph(nx.DiGraph(list(model.edges()) + [(X, Y)])):
                operation = ('+', (X, Y))
                if operation not in tabu_list:
                    old_parents = model.get_parents(Y)
                    new_parents = old_parents + [X]
                    if max_indegree is None or len(new_parents) <= max_indegree:
                        score_delta = local_score(Y, new_parents) - local_score(Y, old_parents)
                        yield(operation, score_delta)
#
        for (X, Y) in model.edges():  # (2) remove single edge
            operation = ('-', (X, Y))
            if operation not in tabu_list:
                old_parents = model.get_parents(Y)
                new_parents = old_parents[:]
                new_parents.remove(X)
                score_delta = local_score(Y, new_parents) - local_score(Y, old_parents)
                yield(operation, score_delta)

        for (X, Y) in reversibleedges:  # (3) flip single edge
            new_edges = list(model.edges()) + [(Y, X)]
            new_edges.remove((X, Y))
            if nx.is_directed_acyclic_graph(nx.DiGraph(new_edges)):
                operation = ('flip', (X, Y))
                if operation not in tabu_list and ('flip', (Y, X)) not in tabu_list:
                    old_X_parents = model.get_parents(X)
                    old_Y_parents = model.get_parents(Y)
                    new_X_parents = old_X_parents + [Y]
                    new_Y_parents = old_Y_parents[:]
                    new_Y_parents.remove(X)
                    if max_indegree is None or len(new_X_parents) <= max_indegree:
                        score_delta = (local_score(X, new_X_parents) +
                                       local_score(Y, new_Y_parents) -
                                       local_score(X, old_X_parents) -
                                       local_score(Y, old_Y_parents))
                        yield(operation, score_delta)
                        
    
    def createRandLegalDag3(self, timelayers, maxdeg,seed=1990):
        '''
        Parameters
        ----------
        timelayers: int. The number of time layers that is being considered.
        maxdeg= the maximum indegree of any node in the final graph 
        Returns
        -------
        A legal (inter time-slicce neurons are only connected from a time slice to the last time slice. Also 
        intra-slice connections are only present in the last slice.) DAG on self.state_names.keys(). 
        https://stackoverflow.com/questions/33909968/extracting-a-random-sublist-from-a-list-in-python
        '''
        random.seed(seed)
        np.random.seed(seed)
        all_layers=self.get_all_layers(timelayers, self.state_names.keys())
        otherlayers, lastlayer, startr =list(), all_layers[timelayers-1], DAG()
        for i in range(0, timelayers-1):
           otherlayers.extend(all_layers[i]) 
           startr.add_nodes_from(all_layers[i]) 
        startr.add_nodes_from( lastlayer)
        for node in lastlayer:
            templist= lastlayer.copy()
            templist.remove(node)
            candidtaeparents, num = templist+ otherlayers , random.randint(0,maxdeg)
#            print('for node {0}, the indegree is {1}'.format(node, num))            
            parentlist= random.sample(candidtaeparents, num )
#            print('The list of parents is {0}'.format(parentlist))
            edges= [(x,node) for x in parentlist]
            startr.add_edges_from(edges)
        try:
            H= startr.subgraph(lastlayer)
            nx.find_cycle(H, source=None, orientation= 'original')
            random.seed(seed)
            np.random.seed(seed)
            seedn= random.randint(1, len(lastlayer)* timelayers)
            return(self.createRandLegalDag3(timelayers, maxdeg,seed=seedn) )
        except:
            return (startr, seed)

                    
    def RestrictedEstimatetl(self, timelayers, start=None, tabu_length=0, max_indegree=None, N=2):
        """
        Performs local hill climb search to estimates the `DAG` structure
        that has optimal score, according to the scoring method supplied in the constructor.
        Starts at model `start` and proceeds by step-by-step network modifications
        until a local maximum is reached. It only considers modifications which lead to  legal DAG (inter time-layer neurons 
        are only connected from earlier time layers to last time layer. Also intra-layer connections are only present 
        in the last layer). Only estimates network structure, no parametrization.
        N is the number of steps after which if there is yet no new best model the algo gives up.

        Parameters
        ----------
        start: DAG instance
            The starting point for the local search. Remember to pass a dag which has the same nodes as the variable names in self.data
        tabu_length: int.   If provided, the last `tabu_length` graph modifications cannot be reversed
            during the search procedure. This serves to enforce a wider exploration of the search space.             
        max_indegree: int or None. THIS IS YET IMPLEMENTED. BEST TO KEEP NONE.
            If provided and unequal None, the procedure only searches among models
            where all nodes have at most `max_indegree` parents. Defaults to None.
        N: int. If N successive iterations of finding best non-tabu operation leads to no improved structure then search is curtailed
        timelayers: int. number of time slices used   

        Returns
        -------
        model: `DAG` instance which is legal .  A `DAG` at a (local) score maximum.          
        """
        epsilon = 1e-8
        tabu_list = []
        current_model = start
        count=0
        while True:
            best_score_delta = 0
            best_operation = None

            for operation, score_delta in self.legal_operations2(current_model, timelayers, tabu_list, max_indegree):
                if score_delta > best_score_delta + epsilon:
                    best_operation = operation
                    best_score_delta = score_delta
            if best_operation is None:
                 FlagToChange=0
                 count=count+1
                 if len(tabu_list)<=0: 
                     break
                 '''trying to see if anything in the tabu list yields a suitable result.'''
                 local_score = self.scoring_method.local_score
                 G1= current_model
                 while (len(tabu_list)>0) and (count <= N):
                     temp1= tabu_list.pop(len(tabu_list)-1)
#                     print(temp1)
                     X, Y = temp1[1]
                     ''' when the bottom of temp is + '''
                     if temp1[0] == '+':
                         if ( nx.is_directed_acyclic_graph(nx.DiGraph(list(G1.edges()) + [(X, Y)])) ) and (X not in G1.get_parents(Y)):
                             old_parents = G1.get_parents(Y)
                             new_parents = old_parents + [X]
                             if ((local_score(Y, new_parents) - local_score(Y, old_parents))>epsilon):
                                 FlagToChange=1
                                 best_operation = temp1
                                 best_score_delta=local_score(Y, new_parents) - local_score(Y, old_parents)
                                 break
                             else:
                                 count=count+1                                 
                         else:
                             count=count+1
                     ''' when the bottom of temp is -'''
                     if temp1[0] == '-':
                         if X in G1.get_parents(Y):
                             old_parents = G1.get_parents(Y)
                             new_parents = old_parents[:]
                             new_parents.remove(X)
                             score_delta1 = local_score(Y, new_parents) - local_score(Y, old_parents)                         
                             if (score_delta1 > epsilon):
                                 FlagToChange=1
                                 best_operation = temp1
                                 best_score_delta=score_delta1
                                 break
                             else:
                                 count=count+1
                         else:
                             count=count+1                        
                     ''' when the bottom of temp is flip'''
                     if temp1[0] == 'flip':
                         new_edges = list(G1.edges()) + [(Y, X)]
                         if (X in G1.get_parents(Y)):
                             new_edges.remove((X, Y))
                         if (nx.is_directed_acyclic_graph(nx.DiGraph(new_edges))) and (X in G1.get_parents(Y)):
                             old_X_parents = G1.get_parents(X)
                             old_Y_parents = G1.get_parents(Y)
                             new_X_parents = old_X_parents + [Y]
                             new_Y_parents = old_Y_parents[:]
                             new_Y_parents.remove(X) 
                             score_delta1 = (local_score(X, new_X_parents) +
                                       local_score(Y, new_Y_parents) -
                                       local_score(X, old_X_parents) -
                                       local_score(Y, old_Y_parents))
                             if (score_delta1 > epsilon):
                                 FlagToChange=1
                                 best_operation = temp1
                                 best_score_delta=score_delta1
                                 break
                             else:
                                 count=count+1                                                                
                         else:
                             count=count+1                                                    
            else:  
                FlagToChange=1
            if count > N:
                break
            if FlagToChange==1:
                count =0
                if best_operation[0] == '+':
                    current_model.add_edge(*best_operation[1])
                    '''
                    The operation below just takes the full list if the list on the LHS has less than tabu_length length
                    '''
                    tabu_list = ([('-', best_operation[1])] + tabu_list)[:tabu_length]
                elif best_operation[0] == '-':
                    current_model.remove_edge(*best_operation[1])
                    tabu_list = ([('+', best_operation[1])] + tabu_list)[:tabu_length]
                elif best_operation[0] == 'flip':
                    X, Y = best_operation[1]
                    current_model.remove_edge(X, Y)
                    current_model.add_edge(Y, X)
                    tabu_list = ([best_operation] + tabu_list)[:tabu_length]

        return current_model
                
   
