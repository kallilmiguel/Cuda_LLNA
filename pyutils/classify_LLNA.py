#%%

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Experimenter: compares models with different initialization")
    #data, paths, and other settings of general setup
    parser.add_argument('--dataset', type=str, default="fungi", help="Database Graph")

    parser.add_argument('--rules', type=int, default=1, help='Number of rules to be considered')
    parser.add_argument('--repetitions', type=int, default=10, help='Number of repetitions for the classification algorithm')
    parser.add_argument('--iteration', type=int, default=1, help="Iteration number (for random seed evaluation)")
    parser.add_argument('--bset', type=list, default=[(20,)], help="number of bins")

    parser.add_argument('--methods', type=list, default=["DLLNA"], help="List of methods to evaluate")
    parser.add_argument('--savename', type=str, default='accuracies', help="name for accuracy save file")
    parser.add_argument('--folds', type=int, default=10, help="Number of folds for classification")
    return parser.parse_args()


import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score, KFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

if __name__ == "__main__":

    args = parse_args()
    rules = args.rules
    DATABASE = args.dataset
    methods = args.methods
    CSV_PATH = "/home/kallilzie/Cuda_LLNA/data/train/"+DATABASE+"/statistical/"
    repetitions = args.repetitions
    iteration = args.iteration
    bset = args.bset[0]
    savename = args.savename
    folds = args.folds
    measures = ["global", "state", "combined"]

    for method in methods:
        for rule in range(rules):
            accs_for_measures = np.zeros(len(measures))
            stds_for_measures = np.zeros(len(measures))
            measure_performance = np.zeros(shape=(repetitions,len(measures)))
            for k in range(repetitions):
                if method == "DLLNA":
                    df = pd.read_csv(CSV_PATH+method+f'_rule_{rule}_b={bset}.csv')
                else:
                    df = pd.read_csv(CSV_PATH+method+f'_rule_{rule}.csv')
                df = df.drop(df.columns[0], axis=1)

                X = df.iloc[:,:-1].values
                y = df.iloc[:, -1].values
                kfold = KFold(n_splits=folds, shuffle=True)

                n_features = int(X.shape[1]/(len(measures)-1))
                
                for i in range(len(measures)):
                    if measures[i] == "combined":
                        X_measure = np.copy(X)
                    else:
                        X_measure = df.iloc[:,i*n_features:(i+1)*n_features].values

                    local_accuracies = []
                    for train_index, test_index in kfold.split(X_measure):
                        X_train, X_test = X_measure[train_index], X_measure[test_index]
                        y_train, y_test = y[train_index], y[test_index]  

                        std = StandardScaler()
                        X_train = std.fit_transform(X_train)
                        X_test = std.transform(X_test)

                        svm = SVC(tol=1e-9)
                        svm.fit(X_train, y_train)
                        y_pred = svm.predict(X_test)

                        local_accuracies.append(accuracy_score(y_test, y_pred))
                    
                    local_accuracies = np.array(local_accuracies)
                    measure_performance[k, i] = np.mean(local_accuracies)
                
            for i in range(len(measures)):
                accs_for_measures[i] = np.mean(measure_performance[:,i])
                stds_for_measures[i] = np.std(measure_performance[:,i])
            file = CSV_PATH+method+f"_rule_{rule}_{savename}.csv"
            if(iteration == 0):
                f = open(file, "w")
                f.write("iter")
                for i in range(len(measures)):
                    f.write(f",acc_{measures[i]}, std_{measures[i]}")
                f.write("\n")
                f.close()
            else:
                f = open(file, "a")
                f.write(f"{iteration}")
                for i in range(len(measures)):
                    f.write(f",{accs_for_measures[i]}, {stds_for_measures[i]}")
                f.write("\n")
                f.close() 
                
                  
# %%
