import pickle
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import sys
from skimage.segmentation import slic

# x = np.array([[1, 0, 0],[0, 1, 1]])
# i = np.nonzero(x == 1)
# print(i)
# # print(np.reshape(i, (len(i[0]), 2)))
# z = np.zeros((len(i[0]), 2), dtype="int32")
# z[:,0] = i[0]
# z[:,1] = i[1]
# print(z)
# sys.exit(0)

with open('file.pickle','rb') as file:
     plt.subplot(1,2,1)
     segmentation_mask = pickle.load(file)
     plt.imshow(segmentation_mask)
     # labels = np.zeros(segmentation_mask.shape)
     # coord = np.nonzero(segmentation_mask == 1)
     # z = np.zeros((len(coord[0]), 2), dtype="int32")
     # z[:, 0] = coord[0]
     # z[:, 1] = coord[1]
     # dbscan = DBSCAN(eps=2, min_samples=10)
     # label_1d = dbscan.fit_predict(z)
     labels = slic(segmentation_mask, n_segments=10, compactness=20)
     # labels[coord] = label_1d + 1
     plt.subplot(1,2,2)
     plt.imshow(labels)
     plt.show()
