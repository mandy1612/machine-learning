from sklearn.cluster import KMeans
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.image import imread
from skimage.transform import resize
from matplotlib.colors import to_hex

img = imread("ss1.png")
# print(img.shape)
img = resize(img,(360,480))
# print(img.shape)

img_flatten = np.reshape(img,(img.shape[0]*img.shape[1],img.shape[2]))
img_flatten = img_flatten[:,:-1]
# print(img_flatten.shape)

kmeans = KMeans(n_clusters=3,random_state=1)
kmeans.fit(img_flatten)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_
print(centroids)


# plt.figure(figsize=(9,7))
# axs = plt.axes(projection="3d")
# axs.scatter3D(img_flatten[:,0],img_flatten[:,1],img_flatten[:,2],c=labels.astype(float),edgecolor="k")
# axs.scatter3D(centroids[0,:],centroids[1,:],centroids[2,:],color="red",marker="^")
# plt.show()

# plt.scatter(img_flatten[:,0],img_flatten[:,1],c=labels.astype(float),edgecolor="k")
# plt.scatter(centroids[:,0],centroids[:,1],color="red",marker="^")
# plt.show()

for color in centroids:
    print(to_hex(color))
    plt.figure(figsize=(1,1))
    plt.imshow([[color]])
    plt.axis("off")
    plt.show()

