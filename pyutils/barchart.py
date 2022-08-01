#%%
import pandas as pd
import numpy as np
import seaborn as sns

OUTPUT_PATH = '/home/kallilzie/data/images/'

Palette = ["#090364", "#091e75"]

df =  pd.read_csv('/home/kallilzie/Cuda_LLNA/data/train/oneofakind/statistical/DLLNA_rule_0_32its.csv')

arr = df.iloc[0,:].values
x = np.arange(len(arr))

#%%
sns.set_style('darkgrid')
sns.color_palette('rocket')
sns.set(rc={'figure.figsize': (16,8)})


ax = sns.barplot(x=x, y=arr, units=None, palette="rocket")
ax.set(xticklabels=[], yticklabels=[])

fig = ax.get_figure()

fig.savefig(OUTPUT_PATH+'barabasi32.png')

# %%
