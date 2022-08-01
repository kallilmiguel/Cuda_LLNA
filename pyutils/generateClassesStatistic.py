
#%%
import os
import numpy as np
import pandas as pd
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Experimenter: compares models with different initialization")
    #data, paths, and other settings of general setup
    parser.add_argument('--dataset', type=str, default="fungi", help="Database Graph")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    DATABASE = args.dataset

    DATA_PATH = "/home/kallilzie/Cuda_LLNA/data/matrices/"+DATABASE+"/"
    LLNA_OUT_PATH = "/home/kallilzie/Cuda_LLNA/data/train/"+DATABASE+"/LLNA_classes.csv"
    DLLNA_OUT_PATH = "/home/kallilzie/Cuda_LLNA/data/train/"+DATABASE+"/DLLNA_classes.csv"
    TEP_OUT_PATH = "/home/kallilzie/Cuda_LLNA/data/train/"+DATABASE+"/TEPs_classes.csv"


    if DATABASE == "fungi":
        classes = ['basidio', 'saccharo', 'eurotio', 'sordario']
    elif DATABASE == "scalefree":
        classes = ['barabasi', 'mendes', 'BANL15', 'BANL05', 'BANl20']
    elif DATABASE == "oneofakind" or DATABASE == '4models':
        classes = ['barabasi', 'erdos', 'geo', 'watts']
    elif DATABASE == "plant":
        classes = ['eudicots', 'greenalgae', 'monocots']
    elif DATABASE == "protist":
        classes = ['Alveolates', 'Amoebozoa', 'Euglenozoa','Stramenopiles']
    elif DATABASE == "kingdom":
        classes = ['animals', 'fungi', 'plants','protist']
    elif DATABASE == "animal":
        classes = ['birds', 'fishes', 'insects','mammals']
    elif DATABASE == "actinobacteria":
        classes = ['Coryne', 'Myco', 'Streptomyces','protist']
    elif DATABASE == "firmicutes":
        classes = ['Bacillus', 'Lactobacillus', 'Straphilococcus','Streptococcus']
    elif DATABASE == "social":
        classes = ['gplus', 'twitter']
    elif DATABASE == "ns10" or DATABASE == "ns20" or DATABASE == "ns30":
        classes = ['BA','BANL2','BANL5','BANL15','ER','GEO', 'MEN', 'WS']
    elif DATABASE == "lit-fullLem" or DATABASE == "lit-partialLem" or DATABASE == "lit-nullLem":
        classes = ['autor1','autor2','autor3','autor4','autor5','autor6', 'autor7', 'autor8']
    elif DATABASE == "stomatos":
        classes = ['4h', '24h', 'natural']

    #%%
    file_classes = []
    file_degree = []

    splitter = "_net_s"
    for file in os.listdir(DATA_PATH):
        if "shannon_tep" in file:
            s1, s2 = file.split(splitter)
            class_index = 0
            for cl in classes:
                if cl in file:
                    file_classes.append((DATA_PATH+file, DATA_PATH+s1+'_net'+'_word_tep.csv', DATA_PATH+s1+'_net'+'_lz_tep.csv', class_index))
                    class_index=0
                    break
                class_index+=1

    file_classes = np.array(file_classes)

    pd.DataFrame(file_classes).to_csv(LLNA_OUT_PATH)

    # %%
    file_classes = []
    file_degree = []
    splitter = "_net_den"
    for file in os.listdir(DATA_PATH):
        if "density_histogram" in file:
            class_index = 0
            s1,s2 = file.split(splitter)
            for cl in classes:
                if DATABASE == "stomatos":
                    if file.startswith(cl):
                        file_classes.append((DATA_PATH+file, DATA_PATH+s1+'_net'+'_density_state_histogram.csv',class_index))
                        class_index=0
                        break
                else:
                    if cl in file:
                        file_classes.append((DATA_PATH+file, DATA_PATH+s1+'_net'+'_density_state_histogram.csv',class_index))
                        class_index=0
                        break
                class_index+=1

    file_classes = np.array(file_classes)

    pd.DataFrame(file_classes).to_csv(DLLNA_OUT_PATH)

    file_classes = []
    file_degree = []
    splitter = "rule_1_"
    for file in os.listdir(DATA_PATH):
        if "rule_1_density" in file:
            class_index = 0
            s1,s2 = file.split(splitter)
            for cl in classes:
                if cl in file:
                    file_classes.append((DATA_PATH+file, DATA_PATH+s1+splitter+'binary.csv',class_index))
                    class_index=0
                    break
                class_index+=1

    file_classes = np.array(file_classes)

    pd.DataFrame(file_classes).to_csv(TEP_OUT_PATH)
# %%
