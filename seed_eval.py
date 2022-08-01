import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Experimenter: compares models with different initialization")
    #data, paths, and other settings of general setup
    parser.add_argument('--dataset', type=str, default="fungi", help="Database Graph")

    parser.add_argument('--rules', type=int, default=1, help='Number of rules to be considered')
    parser.add_argument('--repetitions', type=int, default=100, help='Number of repetitions for the general algorithm')
    parser.add_argument('--classification_reps', type=int, default=10, help="Number of classification repetitions")
    parser.add_argument('--attributes', type=int, default=20, help="Number of attributes to be considered as features")
    parser.add_argument('--gpu', type=int, default=1, help="GPU to use in the execution")
    return parser.parse_args()

import os

if __name__ == "__main__":
    args = parse_args()
    DATASETS = [args.dataset]
    #DATASET = args.dataset
    REPETITIONS = args.repetitions
    number_of_rules = args.rules
    attributes=args.attributes
    classification_reps = args.classification_reps

    import itertools

    for DATASET in DATASETS:
        
        for r in range(REPETITIONS):

            os.system("CUDA_VISIBLE_DEVICES=1 ./Cuda_selected "+ f"data/network/{DATASET}/ "+f"data/rules/{DATASET}.txt "+f"data/matrices/{DATASET}/ "+f"data/seeds/init_state_{r}.csv")

            os.system(f"python pyutils/saveMeasuresStatistics.py --dataset {DATASET} --rules {number_of_rules} --attributes {attributes}")

            os.system(f"python pyutils/classify_LLNA.py --dataset {DATASET} --rules {number_of_rules} --repetitions {classification_reps} --iteration {r}")

