import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def PCA(data, k=2):
    X_bar = data[:, :-1] - data[:, :-1].mean(axis=0) # loại label
    X_bar = X_bar.T
    conv_mat = np.cov(X_bar)
    eigen_values, eigen_vectors = np.linalg.eig(conv_mat)
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalues = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[sorted_index]
    Utk = sorted_eigenvectors[0:k, :]
    new_data = Utk @ X_bar
    new_data = new_data.T
    data_ = np.vstack((new_data.T, data[:, 9].T))
    return data_.T

def visualize(data):
    k = len(data[0]) - 1
    # print(k)
    if k == 2:
        plt.figure(figsize=(8, 8))
        colors = data[:, -1:]
        colors.reshape(len(data))
        target_name = ['Class 2' if na == 2 else 'Class 4' for na in data[:, -1:]]
        plt.scatter(data[:, 0], data[:, 1], c=colors, lw=2, label=target_name)
        plt.show()
        pass
    elif k == 3:
        fig = plt.figure(1, figsize=(8, 6))
        ax = Axes3D(fig, elev=-150, azim=110)
        colors = data[:, -1:]
        colors.reshape(len(data))
        ax.scatter(
            data[:, 0],
            data[:, 1],
            data[:, 2],
            c=colors,
            cmap=plt.cm.Set1,
            edgecolor="k",
            s=40,
        )
        plt.show()
        pass
    else:
        print('Không thể biểu diễn với số chiều > 3')