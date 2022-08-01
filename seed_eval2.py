import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Experimenter: compares models with different initialization")
    #data, paths, and other settings of general setup
    parser.add_argument('--dataset', type=str, help="Database Graph")

    parser.add_argument('--rules', type=int, default=1, help='Number of rules to be considered')
    parser.add_argument('--repetitions', type=int, default=100, help='Number of repetitions for the general algorithm')
    parser.add_argument('--classification_reps', type=int, default=10, help="Number of classification repetitions")
    parser.add_argument('--attributes', type=int, default=20, help="Number of attributes to be considered as features")
    parser.add_argument('--gpu', type=int, default=1, help="GPU to use in the execution")
    parser.add_argument('--folds', type=int, default=10, help="Number of folds for classification")
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
    gpu = args.gpu
    single_labeled_classes = ["social", "ns10", "ns20", "ns30", "lit-fullLem", "lit-partialLem", "lit-nullLem", "stomatos"]
    folds = args.folds

    import itertools
    
    for DATASET in DATASETS:
        first = True
        DATA_PATH = "data/network/"+DATASET+"/"
        
        for r in range(REPETITIONS):
            for file in os.listdir(DATA_PATH):
                if DATASET in single_labeled_classes:
                    os.system("CUDA_VISIBLE_DEVICES=0 ./Cuda_social "+ f"{file} data/network/{DATASET}/ "+f"data/rules/{DATASET}.txt "+f"data/matrices/{DATASET}/ "+f"data/seeds/init_state_{r}.csv")
                else:
                    os.system("CUDA_VISIBLE_DEVICES=0 ./Cuda_test "+ f"{file} data/network/{DATASET}/ "+f"data/rules/{DATASET}.txt "+f"data/matrices/{DATASET}/ "+f"data/seeds/init_state_{r}.csv")

            if(first):
                os.system(f"python pyutils/generateClassesStatistic.py --dataset {DATASET}")
                first = False

            os.system(f"python pyutils/saveMeasuresStatistics.py --dataset {DATASET} --rules {number_of_rules} --attributes {attributes}")

            os.system(f"python pyutils/classify_LLNA.py --dataset {DATASET} --rules {number_of_rules} --repetitions {classification_reps} --iteration {r} --folds {folds}")

