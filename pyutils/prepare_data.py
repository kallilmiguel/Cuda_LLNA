#%%
from fileinput import filename
import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# %%
DATA_PATH = "data/measures/"
OUT_PATH = "data/features/net-split/"
SINGLE_CSV_PATH = "data/features/single/"
number_of_rules = 2**18

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

#%% save every network file in a single dataframe and write into a single csv file
for net in networks:
    DFlist = []
    for file in filenames:
        if net in file:
            df = pd.read_csv(DATA_PATH+folders[0]+'/'+file, header=None)
            for folder in folders[1:]:
                df = pd.concat([df, pd.read_csv(DATA_PATH+folder+'/'+file, header=None)], axis='columns')
            DFlist.append(df)
    dfmerge = DFlist[0]
    for df in DFlist[1:]:
        dfmerge = pd.concat([dfmerge, df])

    dfmerge.to_csv(OUT_PATH+net+'.csv')

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
