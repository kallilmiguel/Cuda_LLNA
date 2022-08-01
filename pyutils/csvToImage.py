#%%
import matplotlib
import numpy as np
from numpy import genfromtxt
from matplotlib import pyplot
from matplotlib.image import imread
import pandas as pd
import os
from measures import *

DATABASE = "oneofakind"
DATA_PATH = '../data/matrices/'+DATABASE+'/'
OUT_PATH = '../data/images/'

files = []
for file in os.listdir(DATA_PATH):
    files.append(file)

#%%
splitter = "_net"
for file in files:
    if "density." in file or "binary." in file:
        image = pd.read_csv(DATA_PATH+file, header=None)
        arr_img = image.iloc[:,:].values
        s1,s2 = file.split(splitter)
        arr_degree = pd.read_csv(DATA_PATH+s1+splitter+"_degree.csv", header=None).iloc[:,:].values
        arr_degree_inds = arr_degree.argsort()

        arr_img_ordered = arr_img[:, arr_degree_inds].reshape(arr_img.shape)
        matplotlib.image.imsave(OUT_PATH+file[:-4]+'.png', arr_img_ordered, cmap='hotmap')
        image = imread(OUT_PATH+file[:-4]+'.png')

        pyplot.imshow(image)
        pyplot.title(file[:-4]+'.png')
        pyplot.show()
    


# %%

splitter = "rule"
for file in files:
    if  "binary." in file:
        image = pd.read_csv(DATA_PATH+file, header=None)
        arr_img = image.iloc[1:,:].values
        shannons = []
        for column in range(arr_img.shape[1]):
            node_values = pd.read_csv(DATA_PATH+file, header=None).iloc[1:,column].values
            shannons.append(shannon_ent(node_values))
        shannons = np.array(shannons)
        arr_degree_inds = shannons.argsort()

        arr_img_ordered = arr_img[:, arr_degree_inds]
        matplotlib.image.imsave(OUT_PATH+file[:-4]+'_entropy_ordered.png', arr_img_ordered, cmap='hotmap')
        image = imread(OUT_PATH+file[:-4]+'_entropy_ordered.png')

        pyplot.imshow(image)
        pyplot.title(file[:-4]+'_entropy_ordered.png')
        pyplot.show()

        s1,s2 = file.split(splitter)
        density_file = s1+splitter+"_1_density.csv"
        density_img = pd.read_csv(DATA_PATH+density_file, header=None).iloc[1:,:].values
        density_img_ordered = density_img[:,arr_degree_inds].reshape(density_img.shape)
        matplotlib.image.imsave(OUT_PATH+density_file[:-4]+'_entropy_ordered.png', density_img_ordered, cmap='hotmap')
        image = imread(OUT_PATH+density_file[:-4]+'_entropy_ordered.png')

        

        pyplot.imshow(image)
        pyplot.title(density_file[:-4]+'_entropy_ordered.png')
        pyplot.show()

# %%
