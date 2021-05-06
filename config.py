import numpy as np
import os
import time
from dataset import DataSet
from SR2CNN import getSR2CNN

#############
# training parameters
lam_center=0.03
lam_encoder=10
feature_dim=256
version='RELEASE'
epoch_num = 250
lr = 1e-3
batchsize = 256
#############

#############
# set GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
#############

#############
# dataset
__dataset_path = './RML2016.10a_dict.pkl'
dataset = DataSet(__dataset_path)
X, lbl, snrs, mods = dataset.get_meta()
X_train, Y_train, X_test, Y_test, classes = dataset.get_train_test()
num_class = len(classes)
#############

#############
model_path='./models/model_{}.pkl'.format(version)
model = getSR2CNN(num_class,feature_dim)
#############
