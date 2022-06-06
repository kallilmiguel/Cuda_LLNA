#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 10:52:45 2022

@author: kallil
"""
#%%
import networklib as nl
import networkx as nx

DIR = "data/"

probs = [0.1]

number_of_nodes = [500,1000,1500,2000]

degree = [4, 6, 8, 10, 12, 14, 16]

iter = 1

for n in number_of_nodes:
    for k in degree:
        for p in probs:
            for i in range(1,iter+1):
                G = nx.watts_strogatz_graph(n,k,p)
                nl.save_graph(G, DIR + f"rulesel/watts_n={n}_k={k}_p={p}._i={i}.txt")
#%%
for n in number_of_nodes:
    for k in degree:
        for i in range(1,iter+1):
            G = nx.erdos_renyi_graph(n, p)
            nl.save_graph(G, DIR + f"rulesel/erdos_n={n}_p={p}.txt")

for n in number_of_nodes:
    for k in degree:
        for i in range(1, iter+1):
            G = nx.barabasi_albert_graph(n, k)
            nl.save_graph(G, DIR + f"rulesel/barabasi_n={n}_k={k}.txt")

