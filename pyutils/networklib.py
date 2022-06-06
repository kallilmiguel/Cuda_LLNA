# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 16:49:06 2021

@author: kalli
"""

import networkx as nx
import numpy as np


#construct a regular network 
#with each node connected to his 4 closest neighbors
def create_regular(n=100):
    G = nx.Graph()
    
    for i in range(n):
        G.add_node(i)
        
    for i in range(n):
        if(i+1 >= n):
            G.add_edge(i, i+1-n)
        else:
            G.add_edge(i,i+1)
        if(i+2 >= n):
            G.add_edge(i, i+2-n)
        else:
            G.add_edge(i,i+2) 
    
    return G



#transform the regular network in a random network by 
# watts & Strogatz model using different p values
def reg2rand(G, p):
    
    n = len(G.nodes())
    
    for i in G.edges():
        r = np.random.uniform(0,1)
        if(r < p):
            G.remove_edge(i[0],i[1])
            v = np.random.randint(0,n)
            while(v == i[0]):
                v = np.random.randint(0,n)
            G.add_edge(i[0],v)
        
    return G

#save a graph in adjlist format
def save_graph(G, path):
    nx.write_adjlist(G, path)
    return 