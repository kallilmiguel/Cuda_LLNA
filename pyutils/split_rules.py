
#%%
from threading import enumerate
import os
import shutil

DATA_PATH = "../data/network/full-dataset/"
OUT_PATH = "../data/rulesel/"

N = [500]
K = [8]
networks_per_config = 10

for n in N:
    for k in K:
        for folder in os.listdir(DATA_PATH):
            for file in os.listdir(DATA_PATH+folder):
                for i in range(1,networks_per_config+1):
                    if f"i={i}.txt" in file and f"n={n}" in file and f"k={k}" in file:
                        with open(DATA_PATH+folder+'/'+file, "r") as fp:
                            lines = fp.readlines()

                        with open(OUT_PATH+file, "w") as fp:
                            for line in lines[2:]:
                                fp.write(line)

#%%
for n in N:
    for k in K:
        for file in os.listdir(DATA_PATH+folder):
            for i in range(1,networks_per_config+1):
                if f"i={i}.txt" in file and f"n={n}" in file and f"k={k}" in file:
                    src = DATA_PATH+folder+'/'+file
                    dst = OUT_PATH+folder+'/'+file
                    print(f"Source: {src}\t Dest: {dst}")
                    shutil.copyfile(src, dst) 
# %%
