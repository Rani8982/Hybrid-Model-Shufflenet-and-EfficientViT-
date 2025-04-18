import csv
import os
import numpy as np
import h5py
import skimage.io
import torch

ck_path = '/kaggle/input/driver2x/nic'
#ck+dataset whaving 7 emotions categorized into seven folders

anger_path = os.path.join(ck_path, 'AN')
disgust_path = os.path.join(ck_path, 'DI')
fear_path = os.path.join(ck_path, 'FE')
happy_path = os.path.join(ck_path, 'HA')
sadness_path = os.path.join(ck_path, 'SA')
surprise_path = os.path.join(ck_path, 'SU')
#contempt_path = os.path.join(ck_path, 'contempt')


# # Creat the list to store the data and label information
data_x = []
data_y = []

datapath = os.path.join('KMUdata','mtcnnkmuneww.h5')
if not os.path.exists(os.path.dirname(datapath)):
    os.makedirs(os.path.dirname(datapath))

# order the file, so the training set will not contain the test set (don't random)
files = os.listdir(anger_path)
files.sort()
for filename in files:
    I = skimage.io.imread(os.path.join(anger_path,filename))
    data_x.append(I.tolist())
    data_y.append(0)

files = os.listdir(disgust_path)
files.sort()
for filename in files:
    I = skimage.io.imread(os.path.join(disgust_path,filename))
    data_x.append(I.tolist())
    data_y.append(1)

files = os.listdir(fear_path)
files.sort()
for filename in files:
    I = skimage.io.imread(os.path.join(fear_path,filename))
    data_x.append(I.tolist())
    data_y.append(2)

files = os.listdir(happy_path)
files.sort()
for filename in files:
    I = skimage.io.imread(os.path.join(happy_path,filename))
    data_x.append(I.tolist())
    data_y.append(3)

files = os.listdir(sadness_path)
files.sort()
for filename in files:
    I = skimage.io.imread(os.path.join(sadness_path,filename))
    data_x.append(I.tolist())
    data_y.append(4)

files = os.listdir(surprise_path)
files.sort()
for filename in files:
    I = skimage.io.imread(os.path.join(surprise_path,filename))
    data_x.append(I.tolist())
    data_y.append(5)

print(np.shape(data_x))
print(np.shape(data_y))

datafile = h5py.File(datapath, 'w')
datafile.create_dataset("data_pixel", dtype = 'uint8', data=data_x)
datafile.create_dataset("data_label", dtype = 'int64', data=data_y)
datafile.close()

print("Save data finish!!!")
