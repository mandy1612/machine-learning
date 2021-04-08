import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt
from matplotlib.image import imread
from sklearn import cluster


# img_flatten = img.reshape((img.shape[0]*img.shape[1],img.shape[2]))
# img_flatten = img_flatten[:,:-1]

# fourier transform for compression of image
class KMeans:
  def __init__(self,img,k=3):
    self.img_flatten = img.reshape((img.shape[0]*img.shape[1],img.shape[2]))
    self.img_flatten = self.img_flatten[:,:-1]
    self.k = k
  
  def compress(self,img):
    im_gray = np.mean(img,-1)
    imhat = np.fft.fft2(im_gray)
    imhatsort = np.sort(np.abs(imhat.reshape(-1)))

    keep = 0.05
    thresh = imhatsort[int(np.floor((1-keep)*len(imhatsort)))]
    ind = np.abs(imhat)>thresh          # Find small indices
    Atlow = imhat * ind                 # Threshold small indices
    compressed_img = np.fft.ifft2(Atlow).real  # Compressed image

    cmap = plt.get_cmap("jet")
    rgba_img = cmap(compressed_img)
    rgb_img = np.delete(rgba_img,3,2)

    plt.figure(figsize=(9,7))
    plt.imshow(compressed_img)
    plt.axis('off')
    plt.title('Compressed image: keep = ' + str(keep))
    plt.show()
    return rgb_img

  def plot_graph3D(self,centroid):
    xmean = centroid[:,0]
    ymean = centroid[:,1]
    zmean = centroid[:,2]
    plt.figure(figsize=(5,4))
    ax = plt.axes(projection="3d")
    ax.scatter3D(self.img_flatten[:,0],self.img_flatten[:,1],self.img_flatten[:,2],color="blue",marker="o")
    ax.scatter3D(xmean,ymean,zmean,color="red",marker="^")
    plt.show()

  def plot_graph2D(self,centroid):
    xmean = centroid[:,0]
    ymean = centroid[:,1]
    plt.figure(figsize=(5,4))
    plt.scatter(self.img_flatten[:,0],self.img_flatten[:,1],color="blue",marker="o")
    plt.scatter(xmean,ymean,color="red",marker="^")
    plt.show()


  def search(self,mat,val):
    for i in range(len(mat)):
      if mat[i] == val:
        return i


  def k_mean(self,centroid):
    prev_centroid = deepcopy(centroid)
    comp = False
    clusters = {}
    while not comp:
      dist = np.zeros((self.img_flatten.shape[0],self.k))
      # calculate distance
      for i in range(self.k):
          dist[:,i] = np.sum(np.power((self.img_flatten - prev_centroid[i]),2),axis=1)

      # dict for clustering data points
      clusters = {0:(),1:(),2:()}
      for i in range(dist.shape[0]):
        r_min = np.min(dist[i])
        clusters[self.search(dist[i],r_min)] += (self.img_flatten[i],)
      for i in clusters.keys():
        tmp = np.array(clusters[i])
        centroid[i] = (1/tmp.shape[0])*np.sum(tmp,axis = 0,keepdims=True)
      comp = (prev_centroid == centroid).all()
      prev_centroid = deepcopy(centroid)

    return centroid,clusters

  def get_labels(self,clusters):
    labels = []
  #   for i in range(self.k):
  #     labels.append(len(clusters[i])*str(i))
    return labels


if __name__ == "__main__":
  img = imread("JoinNow.png")
  initcentroids = np.array([[0.84590641 ,0.47304893 ,0.59792215],[0.01947734 ,0.3077252 ,0.78533317],
                              [0.69290852 ,0.72691926 ,0.16252225]])
  kmeans = KMeans(img)
  # k = int(input("Enter k:"))
  # initcentroids = np.random.rand(k,img_flatten.shape[1])
  print("Initial clustering:\n",initcentroids)
  # plot_graph2D(initcentroids)
  # plot_graph3D(initcentroids)
  finalcentroids,clusters = kmeans.k_mean(initcentroids)
  print("Clustering:\n",finalcentroids)
  labels = kmeans.get_labels(clusters)
  # plot_graph2D(finalcentroids)
  # plot_graph3D(finalcentroids)