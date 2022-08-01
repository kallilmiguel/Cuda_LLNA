#%%
import numpy as np
import pandas as pd

MAX_NODES = 12000
REPETITIONS = 100
SAVE_PATH = '../data/seeds/'

for i in range(REPETITIONS):
    array = np.random.randint(0, 2, size=MAX_NODES)
    df_array = pd.DataFrame(array)
    df_array.to_csv(SAVE_PATH+f'init_state_{i}.csv', header=False, index=False)
# %%
