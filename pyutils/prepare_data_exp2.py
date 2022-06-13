#%%
from fileinput import filename
import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# This file is responsible to pick ten examples of a network configuration with a fixed number of N nodes and
# K mean degree. This is done for all 4 types of networks
# %%
DATA_PATH = "../data/measures/experiment2/"
OUT_PATH = "../data/features/experiment2/"
SINGLE_CSV_PATH = "../data/features/"
number_of_rules = 2**18

N = [500]
K = [8]
networks = ["watts", "erdos", "barabasi", "geo"]
measures = ["shannon"]
folders = []

for folder in os.listdir(DATA_PATH):
    if folder in measures:
        folders.append(folder)

#%% 

filenames = []

for file in os.listdir(DATA_PATH+folders[0]):
    filenames.append(file)

#%% merge all measures into a single csv file for all networks in the directory
for net in networks:
    counter = 0
    for file in filenames: 
        if net in file:
            counter+=1
            df = pd.read_csv(DATA_PATH+folders[0]+'/'+file, header=None)
            for folder in folders[1:]:
                df = pd.concat([df, pd.read_csv(DATA_PATH+folder+'/'+file, header=None)], axis='columns')
            df.to_csv(OUT_PATH+net+ f'_i={counter}.csv')

#%% save every file in dataframe and write into csv file
for net in networks:
    DFlist = []
    for file in filenames:
        if net in file:
            df = pd.read_csv(DATA_PATH+folders[0]+'/'+file, header=None)
            for folder in folders[1:]:
                df = pd.concat([df, pd.read_csv(DATA_PATH+folder+'/'+file, header=None)], axis='columns')
            df.to_csv(OUT_PATH+file)
# %%
