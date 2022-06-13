
#%%
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

def calculate_label_acc(labels):
    total_samples = 2**18
    accuracies = []
    for i in range(len(labels)):
        for j in range(i+1,len(labels)):
            accuracies.append(accuracy_score(labels[i], labels[j]))
    
    return accuracies

# %%

DATA_PATH = "../data/features/experiment2/"

networks = ["erdos", "watts", "barabasi", "geo"]

KMAX = 20

# %%
for network in networks:

    files = [i for i in os.listdir(DATA_PATH) if network in i]
    accuracies = []
    for k in range(2, KMAX+1):
        labels = []
        for file in files:
            df = pd.read_csv(DATA_PATH+file)
            X = df.iloc[:,1:].values

            kmeans = KMeans(n_clusters=k, random_state=0)  
            kmeans.fit(X=X)
            labels.append(kmeans.labels_)


        accuracies.append(np.mean(calculate_label_acc(labels)))
    with open(DATA_PATH+'results.csv', 'a') as fout:
        fout.write(network)
        for accuracy in accuracies:
            fout.write(','+str(accuracy))
        fout.write('\n')

# %%
