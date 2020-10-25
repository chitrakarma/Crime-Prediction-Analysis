import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.vq import whiten
import skfuzzy as fuzz

colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen']

data_set = pd.read_csv('data set/Datasets for prediction/crime_head_wise.csv')
x_axis = data_set['State']
y_axis = data_set['Rate']

y = whiten(y_axis)
# print(y_axis)
# print(y)
df = pd.DataFrame({'x_axis': x_axis, 'y_axis': y})

# fig0, ax0 = plt.subplots()
# ax0.plot(x_axis, y_axis, '.')
# ax0.set_title('Test data: 200 points x3 clusters.')

fig1, axes1 = plt.subplots(3, 3, figsize=(8, 8))
alldata = np.vstack((x_axis, x_axis))

x_axis = list(x_axis)
y_axis = list(y_axis)
fpcs = []

for ncenters, ax in enumerate(axes1.reshape(-1), 2):
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        alldata, ncenters, 2, error=0.005, maxiter=1000, init=None)

    # Store fpc values for later
    fpcs.append(fpc)

    # Plot assigned clusters, for each data point in training set
    cluster_membership = np.argmax(u, axis=0)
    for j in range(ncenters):
        ax.plot(x_axis[cluster_membership == j],
                y_axis[cluster_membership == j], '.', color=colors[j])

    # Mark the center of each fuzzy cluster
    for pt in cntr:
        ax.plot(pt[0], pt[1], 'rs')

    ax.set_title('Centers = {0}; FPC = {1:.2f}'.format(ncenters, fpc))
    ax.axis('off')

fig1.tight_layout()

plt.show()

# print(type(df['x_axis']))
# print(type(x_axis))
# print(x_axis, y_axis)
