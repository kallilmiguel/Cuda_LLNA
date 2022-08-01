
#%%
import numpy as np
import pandas as pd
import os
from measures import *

DATA_PATH = "/home/kallilzie/Cuda_LLNA/data/train/scalefree/statistical/"
CSV_PATH = "/home/kallilzie/Cuda_LLNA/data/train/scalefree/classes.csv"
rules = 1

attributes = 20
max_word_length = 40

df = pd.read_csv(CSV_PATH)
df = df.drop(df.columns[0], axis=1)
classes = df.iloc[:,2].values
paths = df.iloc[:,0].values
# %%

for rule in range(1,rules+1):
    y = []
    X = []
    for i in range(len(paths)):
        if f"rule_{rule}" in paths[i]:
            print(i)
            y.append(classes[i])
            df_graph = pd.read_csv(paths[i], header=None)
            TEP = df_graph.iloc[:,:].values
            TEP = np.array(255*TEP, dtype=np.uint8)
            binary_rep = np.array([], dtype=np.uint8)
            for c in range(TEP.shape[1]):
                evolution_array_dec = TEP[:,c]
                evolution_array_bit = np.array([], dtype=np.uint8)
                for byte in evolution_array_dec:
                    if(len(evolution_array_bit)==0):
                        evolution_array_bit = np.unpackbits(np.uint8(byte))
                    else:
                        evolution_array_bit = np.concatenate(
                        [evolution_array_bit,np.unpackbits(np.uint8(byte))]
                        )
                if(len(binary_rep) == 0):
                    binary_rep = np.copy(evolution_array_bit)
                else:
                    binary_rep = np.vstack([binary_rep, evolution_array_bit])
            binary_rep = binary_rep.transpose()
            
            shannon_array = []
            word_array = np.array([])

            for c in range(binary_rep.shape[1]):
                arr = binary_rep[:, c]
                shannon_array.append(shannon_ent(arr))
                if(c==0):
                    word_array = np.copy(word_ent(arr, max_word_length))
                else:
                    word_array = np.add(word_array, word_ent(arr, max_word_length))
                

            shannon_array = np.array(shannon_array)
            shannon_hist = shannon_ent_histogram(shannon_array, attributes=attributes)
            #shannon_hist = shannon_hist/np.sum(shannon_hist)
            word_hist = word_histogram(word_array, max_word_length, attributes)
            #word_hist = word_hist/np.sum(word_hist)

            X.append(np.concatenate([shannon_hist, word_hist]))

    feature_vector = np.column_stack([X,y])
    dfv = pd.DataFrame(feature_vector)
    dfv.to_csv(DATA_PATH+f'pyrule_{rule}.csv')
# %%
