#%%
import networkx as nx
import igraph as ig
import os
import snap
import numpy as np

DATA_PATH = "../data/network/lit-fullLem-alt/"
OUT_PATH = "../data/network/lit-fullLem/"


#%%
splitter='.'
for file in os.listdir(DATA_PATH):
    G = nx.read_edgelist(DATA_PATH+file)

    nodes = sorted(G.nodes)
    s1, s2 = file.split(splitter)
    ordered_nodes = {}
    for i in range(len(nodes)):
        ordered_nodes[nodes[i]] = i

    G = nx.relabel_nodes(G, ordered_nodes)
    nx.write_adjlist(G, OUT_PATH+s1+'.txt')


#%%
for file in os.listdir(OUT_PATH):

    with open(OUT_PATH+file, "r+") as fp:
        lines = fp.readlines()

        fp.seek(0)

        fp.truncate()

        fp.writelines(lines[3:])
# %%

DATA_PATH = "../data/network/kingdom/"
#%%
for file in os.listdir(DATA_PATH):
    G = nx.read_adjlist(DATA_PATH+file)
    print(np.size(G.edges))
# %%

file = DATA_PATH+"animals_ptg_net.txt"

G = nx.read_adjlist(file)
print(np.size(G.nodes),np.size(G.edges))


#%%
splitter='.'
for file in os.listdir(DATA_PATH):
    G = nx.read_adjlist(DATA_PATH+file)

    nodes = sorted(G.nodes)
    s1, s2 = file.split(splitter)
    ordered_nodes = {}
    for i in range(len(nodes)):
        ordered_nodes[nodes[i]] = i

    G = nx.relabel_nodes(G, ordered_nodes)
    nx.write_adjlist(G, OUT_PATH+s1+'.txt')



for file in os.listdir(OUT_PATH):

    with open(OUT_PATH+file, "r+") as fp:
        lines = fp.readlines()

        fp.seek(0)

        fp.truncate()

        fp.writelines(lines[3:])
# %%
