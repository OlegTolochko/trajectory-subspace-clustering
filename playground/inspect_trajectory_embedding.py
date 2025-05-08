# Created by zhicong.xian at 21:50 07.05.2025 using PyCharm
import numpy as np
import pickle
feature_path = "../out/results/trajectory_embedding.npy"
with open(feature_path, 'rb') as fp:
    data_dict = pickle.load(fp)
    print('Debugging')
#  shape [459,360]