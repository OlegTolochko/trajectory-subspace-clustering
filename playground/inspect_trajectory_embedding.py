# Created by zhicong.xian at 21:50 07.05.2025 using PyCharm
import numpy as np

feature_path = "../out/results/trajectory_embedding.npy"
data_dict = np.load(feature_path,allow_pickle=True)
print("data_dict")