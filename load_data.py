import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from natsort import natsorted



#read s11 files

s11_filenames = []
mask_filenames = []

for file in os.listdir('../data/raw/'):
    if file.startswith('s11'):
        s11_filenames.append(file)

    if file.startswith('mask'):
        mask_filenames.append(file)


s11_filenames = natsorted(s11_filenames)
mask_filenames = natsorted(mask_filenames)

s_11_lists = []
mask_lists =[]
for file_s11, file_mask in zip(s11_filenames,mask_filenames):

    #read s11
    path_s11 = '../data/raw/'+file_s11
    complex_num = np.loadtxt(path_s11,dtype=np.complex_)

    s_11_list = complex_num.real.tolist()
    s_11_lists.append(s_11_list)


    #read mask
    path_mask = '../data/raw/'+file_mask

    with open(path_mask, 'r') as f:
        txt = f.read()
    nums = re.findall(r'\[([^][]+)\]', txt)
    arr = np.loadtxt(nums)
    mask_array = arr.flatten().astype(float).tolist()
    mask_lists.append(mask_array)




s11 = pd.DataFrame(s_11_lists)
s11 = s11.add_prefix('s11_')

mask = pd.DataFrame(mask_lists)
mask = mask.add_prefix('mask_')

data = pd.concat([mask,s11], axis = 1)

data.to_csv('../data/processed/data.csv', index = False)



