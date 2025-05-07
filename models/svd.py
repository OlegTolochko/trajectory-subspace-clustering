# Created by zhicong.xian at 16:59 06.05.2025 using PyCharm
import numpy as np
a = [[2, -1, 1],[-1, 2, 1],[1, 1, 2]]
U, S, Vh = np.linalg.svd(a)
print("U: ", U)
print("S: ", S)
print("Vh: ", Vh)

eigenvalues, eigenvectors = np.linalg.eig(a)
print("rank: ", np.linalg.matrix_rank(a))
print("eigenvalues: ", eigenvalues)
print("eigenvectors: ", eigenvectors) # eigenvector is column wise