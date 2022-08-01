#%%
import argparse
import itertools

def parse_args():
    parser = argparse.ArgumentParser(description="Experimenter: compares models with different initialization")
    #data, paths, and other settings of general setup
    parser.add_argument('--dataset', type=str, help="Database Graph")

    parser.add_argument('--rules', type=int, default=1, help='Number of rules to be considered')
    parser.add_argument('--repetitions', type=int, default=100, help='Number of repetitions for the general algorithm')
    parser.add_argument('--classification_reps', type=int, default=10, help="Number of classification repetitions")
    parser.add_argument('--attributes', type=int, default=20, help="Number of attributes to be considered as features")
    parser.add_argument('--seed', type=int, default=4, help='define initial configuration')
    return parser.parse_args()

import os

if __name__ == "__main__":
    args = parse_args()
    DATASETS = ["actinobacteria","animal", "firmicutes", "fungi", "kingdom", "plant", "protist", "scalefree", "social", "ns10", "ns20", "ns30"]
    if(args.dataset):
        DATASETS = [args.dataset]
    REPETITIONS = args.repetitions
    number_of_rules = args.rules
    attributes=args.attributes
    classification_reps = args.classification_reps
    seed = args.seed

    for DATASET in DATASETS:
        
        first = True

        DATA_PATH = "data/network/"+DATASET

        for file in os.listdir(DATA_PATH):
            if DATASET == "social" or DATASET == "ns10" or DATASET == "ns20" or DATASET == "ns30":
                os.system("CUDA_VISIBLE_DEVICES=0 ./Cuda_social "+ f"{file} data/network/{DATASET}/ "+f"data/rules/{DATASET}.txt "+f"data/matrices/{DATASET}/ "+f"data/seeds/init_state_{seed}.csv")
            else:
                os.system("CUDA_VISIBLE_DEVICES=0 ./Cuda_test "+ f"{file} data/network/{DATASET}/ "+f"data/rules/{DATASET}.txt "+f"data/matrices/{DATASET}/ "+f"data/seeds/init_state_{seed}.csv")
        if first:
            os.system(f"python pyutils/generateClassesStatistic.py --dataset {DATASET}")
            first = False
        
        os.system(f"python pyutils/createHistograms.py --dataset {DATASET}")

        os.system(f"python pyutils/classify_LLNA_parameter_eval.py --dataset {DATASET} --rules {number_of_rules} --savename new_histogram_accuracies")


# %%
