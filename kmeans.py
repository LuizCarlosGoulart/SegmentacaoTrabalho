import numpy as np
import cv2
import sys
from random import randint as randi
from tqdm import tqdm
from math import sqrt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def bgr2hex(bgr):
    return "#%02x%02x%02x" % (int(bgr[2]), int(bgr[1]), int(bgr[0]))

def ScatterPlot(img, centroids, clusterLabels, plotNameOut="scatterPlot.png", sample_fraction=0.1):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sample_size = int(img.shape[0] * img.shape[1] * sample_fraction)
    sampled_indices = np.random.choice(img.shape[0] * img.shape[1], sample_size, replace=False)
    
    xs = []
    ys = []
    zs = []
    colors = []
    
    for index in sampled_indices:
        x = index // img.shape[1]
        y = index % img.shape[1]
        xs.append(img[x, y, 2])
        ys.append(img[x, y, 1])
        zs.append(img[x, y, 0])
        colors.append(bgr2hex(centroids[clusterLabels[x, y]]))
    
    ax.scatter(xs, ys, zs, color=colors, s=1)
    plt.savefig(plotNameOut)
    plt.show()

def ShowCluster(img, centroids, clusterLabels, imgNameOut="out.png"):
    result = np.zeros((img.shape), dtype=np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            bgr = centroids[clusterLabels[i, j]]
            result[i, j, 0] = np.uint8(bgr[0])
            result[i, j, 1] = np.uint8(bgr[1])
            result[i, j, 2] = np.uint8(bgr[2])
    cv2.imwrite(imgNameOut, result)
    cv2.imshow("K-Mean Cluster", result)
    cv2.waitKey(0)

def GetEuclideanDistance(Cbgr, Ibgr):
    b = float(Cbgr[0]) - float(Ibgr[0])
    g = float(Cbgr[1]) - float(Ibgr[1])
    r = float(Cbgr[2]) - float(Ibgr[2])
    return sqrt(b*b + g*g + r*r)

def KMeans3D(img, k=2, max_iterations=100, imgNameOut="out.png"):
    Clusters = k
    centroids = np.zeros((k, 3), dtype=np.float64)
    for i in range(Clusters):
        x = randi(0, img.shape[0]-1)
        y = randi(0, img.shape[1]-1)
        b = float(img[x, y, 0])
        g = float(img[x, y, 1])
        r = float(img[x, y, 2])
        centroids[i, 0] = b
        centroids[i, 1] = g
        centroids[i, 2] = r

    print("Centroids:\n", centroids)
    ClusterLabels = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for _ in tqdm(range(max_iterations)):
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                MinDist = sys.float_info.max
                for c in range(Clusters):
                    dist = GetEuclideanDistance(centroids[c], img[x, y])
                    if dist <= MinDist:
                        MinDist = dist
                        ClusterLabels[x, y] = c

        MeanCluster = np.zeros((Clusters, 4), dtype=np.float64)
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                clusterNumber = ClusterLabels[x, y]
                MeanCluster[clusterNumber, 0] += 1
                MeanCluster[clusterNumber, 1] += float(img[x, y, 0])
                MeanCluster[clusterNumber, 2] += float(img[x, y, 1])
                MeanCluster[clusterNumber, 3] += float(img[x, y, 2])

        copy = np.copy(centroids)
        for c in range(Clusters):
            centroids[c, 0] = MeanCluster[c, 1] / MeanCluster[c, 0]
            centroids[c, 1] = MeanCluster[c, 2] / MeanCluster[c, 0]
            centroids[c, 2] = MeanCluster[c, 3] / MeanCluster[c, 0]

        Same = True
        for i in range(centroids.shape[0]):
            for j in range(centroids.shape[1]):
                if copy[i, j] != centroids[i, j]:
                    Same = False
                    break
            if not Same:
                break
        if Same:
            break
    ShowCluster(img, centroids, ClusterLabels, imgNameOut)
    ScatterPlot(img, centroids, ClusterLabels, plotNameOut="scatterPlot.png")

def GetEuclideanDistance5D(Cbgrxy, Ibgrxy):
    b = float(Cbgrxy[0]) - float(Ibgrxy[0])
    g = float(Cbgrxy[1]) - float(Ibgrxy[1])
    r = float(Cbgrxy[2]) - float(Ibgrxy[2])
    x = float(Cbgrxy[3]) - float(Ibgrxy[3])
    y = float(Cbgrxy[4]) - float(Ibgrxy[4])
    return sqrt(b*b + g*g + r*r + x*x + y*y)

def KMeans5D(img, k=2, max_iterations=100, imgNameOut="out5d.png"):
    Clusters = k
    centroids = np.zeros((k, 5), dtype=np.float64)
    for i in range(Clusters):
        x = randi(0, img.shape[0]-1)
        y = randi(0, img.shape[1]-1)
        b = float(img[x, y, 0])
        g = float(img[x, y, 1])
        r = float(img[x, y, 2])
        centroids[i, 0] = b
        centroids[i, 1] = g
        centroids[i, 2] = r
        centroids[i, 3] = float(x)
        centroids[i, 4] = float(y)

    print("Centroids:\n", centroids)
    ClusterLabels = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for _ in tqdm(range(max_iterations)):
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                MinDist = sys.float_info.max
                for c in range(Clusters):
                    dist = GetEuclideanDistance5D(centroids[c], [img[x, y, 0], img[x, y, 1], img[x, y, 2], x, y])
                    if dist <= MinDist:
                        MinDist = dist
                        ClusterLabels[x, y] = c

        MeanCluster = np.zeros((Clusters, 6), dtype=np.float64)
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                clusterNumber = ClusterLabels[x, y]
                MeanCluster[clusterNumber, 0] += 1
                MeanCluster[clusterNumber, 1] += float(img[x, y, 0])
                MeanCluster[clusterNumber, 2] += float(img[x, y, 1])
                MeanCluster[clusterNumber, 3] += float(img[x, y, 2])
                MeanCluster[clusterNumber, 4] += float(x)
                MeanCluster[clusterNumber, 5] += float(y)

        copy = np.copy(centroids)
        for c in range(Clusters):
            centroids[c, 0] = MeanCluster[c, 1] / MeanCluster[c, 0]
            centroids[c, 1] = MeanCluster[c, 2] / MeanCluster[c, 0]
            centroids[c, 2] = MeanCluster[c, 3] / MeanCluster[c, 0]
            centroids[c, 3] = MeanCluster[c, 4] / MeanCluster[c, 0]
            centroids[c, 4] = MeanCluster[c, 5] / MeanCluster[c, 0]

        Same = True
        for i in range(centroids.shape[0]):
            for j in range(centroids.shape[1]):
                if copy[i, j] != centroids[i, j]:
                    Same = False
                    break
            if not Same:
                break
        if Same:
            break
    ShowCluster(img, centroids, ClusterLabels, imgNameOut)

ImageNames = ["2apples.jpg", "2or4objects.jpg", "7apples.jpg", "variableObjects.jpeg"]
No = 3
Image = cv2.imread(ImageNames[No])
Image = cv2.resize(Image, None, fx=0.5, fy=0.5)
print("Image Size:", Image.shape)

KMeans3D(Image, k=6, max_iterations=15, imgNameOut="img_out_3d.png")
KMeans5D(Image, k=6, max_iterations=15, imgNameOut="img_out_5d.png")