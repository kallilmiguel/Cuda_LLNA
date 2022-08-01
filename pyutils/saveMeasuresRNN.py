#%%
import numpy as np
import pandas as pd
import os
from Random_Network import RNN
from sklearn.preprocessing import StandardScaler

DATA_PATH = "/home/kallilzie/Cuda_LLNA/data/train/scalefree/rnn/"
CLASSES_PATH = "/home/kallilzie/Cuda_LLNA/data/train/scalefree/rnn/classes.csv"

Q = 60
rules=7
iterations = 350

df = pd.read_csv(CLASSES_PATH)
df = df.drop(df.columns[0], axis=1)
classes = df.iloc[:,2].values
paths = df.iloc[:,0].values
degree_paths = df.iloc[:,1].values

#%%
for rule in range(1,rules+1):
    X = np.array([])
    y = []
    for i in range(len(paths)):
        if f"rule_{rule}" in paths[i]:
            df_TEP = pd.read_csv(paths[i], header=None)
            df_degree = pd.read_csv(degree_paths[i], header=None)

            y.append(classes[i])

            y_rnn = df_degree.iloc[:iterations,:].values
            X_rnn = df_TEP.iloc[:iterations,:].values

            std = StandardScaler()

            X_rnn = std.fit_transform(X_rnn)
            y_rnn = std.transform(y_rnn)

            num_samples = X_rnn.shape[1]
            num_attributes = X_rnn.shape[0]

            rnn = RNN(num_samples, num_attributes,
            num_hidden_neurons=Q)
            rnn.set_hidden_weights(method='lcg')

            f = rnn.get_output_weights(X_rnn.transpose(), y_rnn)

            if(len(X)==0):
                X = np.copy(f)
            else:
                X = np.vstack([X, f])
           
    y = np.array(y)
    feature_vector = np.column_stack([X,y])
    dfv = pd.DataFrame(feature_vector)
    dfv.to_csv(DATA_PATH+f'rule_{rule}.csv')
# %%
