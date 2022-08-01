#%%
import os
import numpy as np
import pandas as pd
import itertools

#%%
from sklearn.model_selection import train_test_split, cross_val_score, KFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline


rules = 1
DATABASE = "protist"
repetitions = 10
teps = ["sd", "d"]
CSV_PATH = "/home/kallilzie/Cuda_LLNA/data/train/"+DATABASE+"/statistical/"
measures = ["global", "degree", "temporal", "combined"]

BSET = []
stuff = [20,40,60,80,100,120,140,160]
for L in range(1, 3):
    for subset in itertools.combinations(stuff, L):
        BSET.append(subset)

savename = "measure_accs"
folds = 10
#%%
for tep in teps:
    
    accs_for_measures = np.zeros(shape=(len(BSET), len(measures)))
    stds_for_measures = np.zeros(shape=(len(BSET), len(measures)))
    for b in range(len(BSET)):
        measure_performance = np.zeros(shape=(repetitions,len(measures)))
        for k in range(repetitions):
            df = pd.read_csv(CSV_PATH+f'features_'+tep+f'={BSET[b]}.csv')
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

                    # std = StandardScaler()
                    # X_train = std.fit_transform(X_train)
                    # X_test = std.transform(X_test)

                    svm = SVC(tol=1e-9)
                    svm.fit(X_train, y_train)
                    y_pred = svm.predict(X_test)

                    local_accuracies.append(accuracy_score(y_test, y_pred))
                
                local_accuracies = np.array(local_accuracies)
                measure_performance[k, i] = np.mean(local_accuracies)
        
        for i in range(len(measures)):
            accs_for_measures[b, i] = np.mean(measure_performance[:,i])
            stds_for_measures[b, i] = np.std(measure_performance[:,i])
    file = CSV_PATH+f"{tep}_{savename}.csv"
    f = open(file, "w")
    f.write("Bset")
    for i in range(len(measures)):
        f.write(f",acc_{measures[i]},std_{measures[i]}")
    f.write("\n")
    for b in range(len(BSET)):
        f.write(f"{str(BSET[b]).replace(',',' ')}")
        for i in range(len(measures)):
            f.write(f",{accs_for_measures[b][i]},{stds_for_measures[b][i]}")
        f.write("\n")
    f.close() 
                
# %%
df_density = pd.read_csv(CSV_PATH+f"d_{savename}.csv")
df_statedensity = pd.read_csv(CSV_PATH+f"sd_{savename}.csv")
# %%
density_arr = df_density.iloc[:, :].values
statedensity_arr = df_statedensity.iloc[:,:].values


# %%
density_arr_best_indexes = density_arr[:,-2].argsort()[-3:]
statedensity_arr_best_indexes = statedensity_arr[:,-2].argsort()[-3:]
# %%
file = CSV_PATH+f"compare_{savename}.csv"
text= "bset_d, bset_sd, acc, sd\n"
for i in density_arr_best_indexes:
    for j in statedensity_arr_best_indexes:
        text+=f"{str(BSET[i]).replace(',',' ')}, {str(BSET[j]).replace(',', ' ')},"
        df = pd.read_csv(CSV_PATH+f'features_'+tep+f'={BSET[i]}.csv')
        a = df.iloc[:,1:-1].values
        y = df.iloc[:, -1].values
        b = pd.read_csv(CSV_PATH+f'features_'+tep+f'={BSET[j]}.csv').iloc[:,1:-1].values 
        X = np.column_stack([a,b])

        measure_performance = np.zeros(shape=(repetitions))

        for k in range(repetitions):

            kfold = KFold(n_splits=folds, shuffle=True)
            local_accuracies = []
            for train_index, test_index in kfold.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]  

                # std = StandardScaler()
                # X_train = std.fit_transform(X_train)
                # X_test = std.transform(X_test)

                svm = SVC(tol=1e-9)
                svm.fit(X_train, y_train)
                y_pred = svm.predict(X_test)

                local_accuracies.append(accuracy_score(y_test, y_pred))
            
            local_accuracies = np.array(local_accuracies)
            measure_performance[k] = np.mean(local_accuracies)

        text+=f"{np.mean(measure_performance)}, {np.std(measure_performance)}\n"
f = open(file, "w")
f.write(text)
f.close()


# %%
