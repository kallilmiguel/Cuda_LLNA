#%%
import os
import numpy as np
import pandas as pd

DATA_PATH = "/home/kallilzie/Cuda_LLNA/data/matrices/fungi/"
OUT_PATH = "/home/kallilzie/Cuda_LLNA/data/train/rnn/fungi/"

#%%
classes = ['basidio', 'sordario', 'eurotio', 'saccharo']
file_classes = []
file_degree = []
class_index = 0
splitter = '_net'
for cl in classes:
    for file in os.listdir(DATA_PATH):
        if cl in file and "density" in file:
            s1, s2 = file.split(splitter)
            file_classes.append((DATA_PATH+file, DATA_PATH+s1+splitter+'_degree.csv', class_index))
    class_index+=1
# %%
file_classes = np.array(file_classes)
# %%
pd.DataFrame(file_classes).to_csv(OUT_PATH+'classes.csv')

# %%
