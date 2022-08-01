
#%%
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Experimenter: compares models with different initialization")
    #data, paths, and other settings of general setup
    parser.add_argument('--dataset', type=str, default="fungi", help="Database Graph")

    parser.add_argument('--rules', type=int, default=1, help='Number of rules to be considered')
    parser.add_argument('--attributes', type=int, default=20, help='Number of attributes to be considered')

    parser.add_argument('--bset', type=list, default=[(60,)], help='Number of bins to be considered as features')
    parser.add_argument('--methods', type=list, default=["DLLNA"], help="List of methods to evaluate")

    return parser.parse_args()


import numpy as np
import pandas as pd
import os
from measures import *
import itertools

if __name__ == "__main__":

    args = parse_args()
    DATABASE = args.dataset
    methods = args.methods
    rules = args.rules
    attributes= args.attributes
    measures = ["global", "state"]

    BSET = []
    stuff = [20,40,60,80,100,120,140,160]
    for L in range(1, 3):
        for subset in itertools.combinations(stuff, L):
            BSET.append(subset)

    BSET = args.bset
    
    for method in methods:
        DATA_PATH = "/home/kallilzie/Cuda_LLNA/data/train/"+DATABASE+"/statistical/"
        CSV_PATH = "/home/kallilzie/Cuda_LLNA/data/train/"+DATABASE+"/"+ method + "_classes.csv"

        if method == "LLNA":
            df = pd.read_csv(CSV_PATH)
            df = df.drop(df.columns[0], axis=1)
            classes = df.iloc[:,-1].values
            shannon_paths = df.iloc[:,0].values
            word_paths = df.iloc[:,1].values
            lz_paths = df.iloc[:,2].values

            for rule in range(rules):
                y = []
                X = np.array([])
                for i in range(len(classes)):
                    y.append(classes[i])
                    shannon_hist = pd.read_csv(shannon_paths[i], header=None).iloc[rule,:].values
                    word_hist = pd.read_csv(word_paths[i], header=None).iloc[rule,:].values
                    lz_comp = pd.read_csv(lz_paths[i],header=None).iloc[rule,:].values
                    shannon_hist = shannon_hist/np.sum(shannon_hist)
                    word_hist = word_hist/np.sum(word_hist)
                    
                    lz_hist = lempel_ziv_histogram(lz_comp,attributes)

                    lz_hist = lz_hist/np.sum(lz_hist)

                    features = np.concatenate([shannon_hist, word_hist, lz_hist])
                    if(len(X)==0):
                        X = features
                    else:
                        X = np.vstack([X, features])
            
            feature_vector = np.column_stack([X,y])
            dfv = pd.DataFrame(feature_vector)
            dfv.to_csv(DATA_PATH+method+f'_rule_{rule}.csv')
        
        elif method == "DLLNA":
            df = pd.read_csv(CSV_PATH)
            df = df.drop(df.columns[0], axis=1)
            classes = df.iloc[:, -1].values
            gh_paths = df.iloc[:,0].values
            eh_paths = df.iloc[:, 1].values

            for bset in BSET:
                for rule in range(rules):
                    y = []
                    X = np.array([])
                    for i in range(len(classes)):
                        f = np.array([])
                        y.append(classes[i])
                        
                        
                        for j in range(len(measures)):
                            measure_path = df.iloc[:, j].values
                            measure_hist = pd.read_csv(measure_path[i], header=None)
                            bins = measure_hist.iloc[:,-1].values
                            for b in bset:
                                index = np.where(bins==b)[0]
                                global_values = measure_hist.iloc[index,:b].values.reshape(-1)
                                global_values = global_values/np.sum(global_values)
                            
                                if(len(f) == 0):
                                    f = global_values
                                else:
                                    f = np.concatenate([f, global_values])
            

                        if(len(X)== 0):
                            X = f
                        else: 
                            X = np.vstack([X, f])

                feature_vector = np.column_stack([X,y])
                dfv = pd.DataFrame(feature_vector)
                dfv.to_csv(DATA_PATH+method+f'_rule_{rule}_b={bset}.csv')
# %%
