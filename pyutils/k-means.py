
#%%
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def calculate_label_acc(labels):
    total_samples = 2**18
    accuracies = []
    for i in range(len(labels)):
        for j in range(i+1,len(labels)):
            accuracies.append(accuracy_score(labels[i], labels[j]))
    
    return accuracies

# %%

DATA_PATH = "../data/features/same_config_k=4/"

networks = ["erdos", "watts", "barabasi", "geo"]

KMAX = 10

# %%
for network in networks:

    files = [i for i in os.listdir(DATA_PATH) if network in i]
    accuracies = []
    for k in range(2, KMAX+1):
        labels = []
        for file in files:
            df = pd.read_csv(DATA_PATH+file)
            X = df.iloc[1:,1:].values
            
            scaler = StandardScaler()

            X = scaler.fit_transform(X)

            kmeans = KMeans(n_clusters=k)  
            clusters = kmeans.fit_predict(X=X)
            labels.append(clusters)


        accuracies.append(np.mean(calculate_label_acc(labels)))
    with open(DATA_PATH+'results.csv', 'a') as fout:
        fout.write(network)
        for accuracy in accuracies:
            fout.write(','+str(accuracy))
        fout.write('\n')

# %%
