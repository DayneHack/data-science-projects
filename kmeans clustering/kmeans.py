from asyncio.windows_events import NULL
from dataclasses import replace
import numpy as np
import random
import math
from numpy.random import choice

class KMeans():

    def __init__(self, n_clusters: int, init: str='random', max_iter = 300):
        """

        :param n_clusters: number of clusters
        :param init: centroid initialization method. Should be either 'random' or 'kmeans++'
        :param max_iter: maximum number of iterations
        """
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.centroids = None

    def fit(self, X: np.ndarray):
        self.initialize_centroids(X)
        iteration = 0
        index = 0
        clustering = np.zeros(X.shape[0])
        clusteringCompare = None
        while iteration < self.max_iter:

            for i, x in enumerate(X):
                distList = self.euclidean_distance(x, self.centroids)
                distList = distList[distList != 0]
                minVal = 999

                for j, y in enumerate(distList):
                    if y < minVal:
                        minVal = y
                        index = j

                clustering[i] = index
                
            if  np.array_equal(clustering, clusteringCompare):
                return clustering

            clusteringCompare = clustering[:]
            iteration = iteration + 1
            self.update_centroids(clustering, X)

        return clustering

    def update_centroids(self, clustering: np.ndarray, X: np.ndarray):

        for i in range(self.n_clusters):
            self.centroids[i] = X[clustering==i].mean(axis=0)


    def initialize_centroids(self, X: np.ndarray):
        """
        Initialize centroids either randomly or using kmeans++ method of initialization.
        :param X:
        :return:
        """

        newList = []

        if self.init == 'random':

            for i in range(self.n_clusters):
                randNum = random.randint(0, len(X))
                newList.append(X[randNum])
            self.centroids = newList

        elif self.init == 'kmeans++':

            randNum = random.randint(0, len(X))
            newList.append(X[randNum])
            distList = self.euclidean_distance(newList[0], X)
            distList = distList[distList != 0]

            distSum = sum(distList)
            distListWeights = [weight/distSum for weight in distList]

            for i in range (self.n_clusters-1):
                selection = choice(distList, p=distListWeights, replace=False)
                for ji, j in enumerate(distList):
                    if j == selection:
                        newList.append(X[ji])
                distList = self.euclidean_distance(newList[i+1], X)
                distList = distList[distList != 0]
                distSum = sum(distList)
                distListWeights = [weight/distSum for weight in distList]

            self.centroids = newList

        else:
            raise ValueError('Centroid initialization method should either be "random" or "k-means++"')

        return self.centroids

    def euclidean_distance(self, X1:np.ndarray, X2:np.ndarray):
        """
        Computes the euclidean distance between all pairs (x,y) where x is a row in X1 and y is a row in X2.
        Tip: Using vectorized operations can hugely improve the efficiency here.
        :param X1:
        :param X2:
        :return: Returns a matrix `dist` where `dist_ij` is the distance between row i in X1 and row j in X2.
        """

        val = 0

        dist = np.zeros(shape= (1, len(X2)))
    
        for yi, y in enumerate(X2):
            for i in range(2):            #dimension
                val = val + pow(X1[i]-y[i], 2)
            val = math.sqrt(val)
            dist[0][yi] = val
            val = 0
            
        return dist

    def silhouette(self, clustering: np.ndarray, X: np.ndarray):

        silhouetteList = []

        for index, x in enumerate(clustering):

            distances = self.euclidean_distance(X[index], self.centroids)
            a = np.amin(distances)
            distances = distances[distances != a]
            b = np.amin(distances)

            silhouetteList.append((b-a)/b)
        
        return sum(silhouetteList)/len(silhouetteList)
