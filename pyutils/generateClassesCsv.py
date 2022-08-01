#%%
import os
import numpy as np
import pandas as pd

DATA_PATH = "/home/kallilzie/Cuda_LLNA/data/matrices/scalefree/"
OUT_PATH = "/home/kallilzie/Cuda_LLNA/data/train/scalefree/classes.csv"

#%%
file_classes = []
file_degree = []
classes = ['barabasi', 'mendes', 'BANL20', 'BANL05', 'BANL15']
splitter = "_net"
for file in os.listdir(DATA_PATH):
    if "density" in file:
        s1, s2 = file.split(splitter)
        class_index = 0
        for cl in classes:
            if cl in file:
                file_classes.append((DATA_PATH+file, DATA_PATH+s1+splitter+'_degree.csv', class_index))
                class_index=0
                break
            class_index+=1
        
            
# %%
file_classes = np.array(file_classes)
# %%
pd.DataFrame(file_classes).to_csv(OUT_PATH)
# %%
