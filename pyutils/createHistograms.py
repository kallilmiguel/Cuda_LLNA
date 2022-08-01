#%%
import numpy as np
import pandas as pd
import itertools
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Experimenter: compares models with different initialization")
    #data, paths, and other settings of general setup
    parser.add_argument('--dataset', type=str, default="fungi", help="Database Graph")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    dataset = args.dataset

    DATA_PATH = "/home/kallilzie/Cuda_LLNA/data/train/"+dataset+"/TEPs_classes.csv"
    OUT_PATH = "/home/kallilzie/Cuda_LLNA/data/train/"+dataset+"/statistical/"

    BSET = []
    stuff = [20,40,60,80,100,120,140,160]
    for L in range(1, 3):
        for subset in itertools.combinations(stuff, L):
            BSET.append(subset)


    df = pd.read_csv(DATA_PATH)
    df = df.drop(df.columns[0], axis=1)
    labels = df.iloc[:,-1].values
    paths = df.iloc[:,:-1].values

    for bset in BSET:
        y = []
        X = np.array([])
        for i in range(len(labels)):
            y.append(labels[i])

            df_dtep = pd.read_csv(paths[i,0],header=None)
            df_tep = pd.read_csv(paths[i,1])

            dtep_degrees = df_dtep.iloc[0,:].values.astype(np.int32)
            dtep_values = df_dtep.iloc[1:,:].values
            tep_values = df_tep.iloc[:,:].values

            gh = np.array([])
            gsh = np.array([])
            dh = np.array([])
            dsh = np.array([])
            th = np.array([])
            tsh = np.array([])
            ## GLOBAL HISTOGRAM
            for b in bset:
                
                global_histogram = np.histogram(dtep_values, bins=b, range=(0,1))[0]
                global_histogram = global_histogram/np.sum(global_histogram)
            
            if(len(gh)==0):
                gh = np.copy(global_histogram)
            else:
                gh = np.concatenate([gh, global_histogram])

            ## GLOBAL STATE HISTOGRAM
            for b in bset:
                gstate_values = dtep_values*(2*tep_values-1)
                gstate_histogram = np.histogram(gstate_values, bins=b, range=(-1,1))[0]
                gstate_histogram = gstate_histogram/np.sum(gstate_histogram)

            if(len(gsh)==0):
                gsh = np.copy(gstate_histogram)
            else:
                gsh = np.concatenate([gsh, gstate_histogram])


            ## DEGREE HISTOGRAM IMPLEMENTATION
            for b in bset:

                max_degree = np.max(dtep_degrees)
                degree_histogram = []
                gstate_dhistogram = []
                for i in range(1,max_degree+1):
                    indexes = np.where(dtep_degrees == i)
                    if(len(indexes[0])):
                        dtep_values_degree = dtep_values[:, indexes].reshape(-1)
                        degree_histogram.append(np.histogram(dtep_values_degree, bins=b, range=(0,1))[0])
                        gstate_values_degree = gstate_values[:,indexes].reshape(-1)
                        gstate_dhistogram.append(np.histogram(gstate_values_degree, bins=b, range=(-1,1))[0])
                degree_histogram = np.array(degree_histogram)
                degree_histogram = degree_histogram.mean(axis=0)
                degree_histogram = degree_histogram/np.sum(degree_histogram)
                gstate_dhistogram = np.array(gstate_dhistogram)
                gstate_dhistogram = gstate_dhistogram.mean(axis=0)
                gstate_dhistogram = gstate_dhistogram/np.sum(gstate_dhistogram)

                if(len(dh)==0):
                    dh = np.copy(degree_histogram)
                else:
                    dh = np.concatenate([dh, degree_histogram])

                if(len(dsh)==0):
                    dsh = np.copy(gstate_dhistogram)
                else:
                    dsh = np.concatenate([dsh, gstate_dhistogram])


            for b in bset:
                ##TEMPORAL HISTOGRAM IMPLEMENTATION
                max_it = gstate_values.shape[0]
                temporal_dtep_hist = []
                temporal_gstate_hist = []
                for i in range(max_it):
                    temporal_dtep_hist.append(np.histogram(dtep_values[i,:], bins=b, range=(0,1))[0])
                    temporal_gstate_hist.append(np.histogram(gstate_values[i,:], bins=b, range=(-1,1))[0])

                temporal_dtep_hist = np.array(temporal_dtep_hist)
                temporal_dtep_hist = temporal_dtep_hist.mean(axis=0)
                temporal_dtep_hist = temporal_dtep_hist/np.sum(temporal_dtep_hist)
                temporal_gstate_hist = np.array(temporal_gstate_hist)
                temporal_gstate_hist = temporal_gstate_hist.mean(axis=0)
                temporal_gstate_hist = temporal_gstate_hist/np.sum(temporal_gstate_hist)

                if(len(th)==0):
                    th = np.copy(temporal_dtep_hist)
                else:
                    th = np.concatenate([th, temporal_dtep_hist])

                if(len(tsh)==0):
                    tsh = np.copy(temporal_gstate_hist)
                else:
                    tsh = np.concatenate([tsh, temporal_gstate_hist])



                ### FAST FOURIER TRANSFORM IMPLEMENTATION
                # time_series = np.copy(dtep_values)

                # fourier = np.fft.fft(time_series, n=300, axis=0)
                # fourier_real = np.real(fourier)
                # fourier_imag = np.imag(fourier)
                # fourier_abs = np.absolute(fourier)
                        
                # fourier_real_hist = np.histogram(fourier_real, bins=b, range=(0,1))[0] 
                # fourier_imag_hist = np.histogram(fourier_imag, bins=b, range=(0,1))[0]  
                # fourier_abs_hist = np.histogram(fourier_abs, bins=b, range=(0,1))[0]  

            f = np.concatenate([gh, gsh, dh, dsh, th, tsh])
            
            if(len(X) == 0):
                X = np.copy(f)
            else:
                X = np.vstack([X, f])
        features = np.column_stack([X, y])
        dfv = pd.DataFrame(features)
        dfv.to_csv(OUT_PATH+f"DLLNA_rule_0_b={bset}.csv")

        

    

# %%
