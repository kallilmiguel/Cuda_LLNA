
#%%
from threading import enumerate
import os
import shutil

DATA_PATH = "data/network/full-dataset/"
OUT_PATH = "data/rulesel/"

for folder in os.listdir(DATA_PATH):
    for file in os.listdir(DATA_PATH+folder):
        if "i=1.txt" in file:
            with open(DATA_PATH+folder+'/'+file, "r") as fp:
                lines = fp.readlines()

            with open(OUT_PATH+file, "w") as fp:
                for line in lines[2:]:
                    fp.write(line)

#%%
for folder in os.listdir(DATA_PATH):
    for file in os.listdir(DATA_PATH+folder):
       if "i=1.txt" in file:
           src = DATA_PATH+folder+'/'+file
           dst = OUT_PATH+folder+'/'+file
           print(f"Source: {src}\t Dest: {dst}")
           shutil.copyfile(src, dst) 
# %%
