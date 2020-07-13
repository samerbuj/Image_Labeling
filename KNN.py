__authors__ = ['1494758', '1494603', '1490885']
__group__ = 'DL.15'

import numpy as np
import math
import operator
from scipy.spatial.distance import cdist


class KNN:
    def __init__(self, train_data, labels):

        self._init_train(train_data)
        self.labels = np.array(labels)

    def _init_train(self, train_data):
        """
        initializes the train data
        :param train_data: PxMxNx3 matrix corresponding to P color images
        :return: assigns the train set to the matrix self.train_data shaped as PxD (P points in a D dimensional space)
        """
        if isinstance(train_data, float) is False:
            self.train_data = train_data.astype(np.float)

        if len(train_data.shape) > 2:
            self.train_data = np.reshape(self.train_data,
                                         (len(train_data), int(np.size(self.train_data) / len(train_data))))

    def get_k_neighbours(self, test_data, k):
        """
        given a test_data matrix calculates de k nearest neighbours at each point (row) of test_data on self.neighbors
        :param test_data:   array that has to be shaped to a NxD matrix ( N points in a D dimensional space)
        :param k:  the number of neighbors to look at
        :return: the matrix self.neighbors is created (NxK)
                 the ij-th entry is the j-th nearest train point to the i-th test point
        """
        test_data = test_data.reshape(len(test_data), test_data[0].size)
        self.neighbors = self.labels[np.argsort(cdist(test_data, self.train_data), axis=1)[:, :k]]

    def get_class(self):
        """
        Get the class by maximum voting
        :return: 2 numpy array of Nx1 elements.
                1st array For each of the rows in self.neighbors gets the most voted value
                            (i.e. the class at which that row belongs)
                2nd array For each of the rows in self.neighbors gets the % of votes for the winning class
        """
        final = np.empty([1, 0], dtype=object)
        for i, line in enumerate(self.neighbors):
            clases = {}
            for element in line:
                if element not in clases:
                    clases[element] = 1
                else:
                    clases[element] += 1
            final = np.insert(final, i, max(clases, key=clases.get))
        return final

    def predict(self, test_data, k):
        """
        predicts the class at which each element in test_data belongs to
        :param test_data: array that has to be shaped to a NxD matrix ( N points in a D dimensional space)
        :param k:         :param k:  the number of neighbors to look at
        :return: the output form get_class (2 Nx1 vector, 1st the classm 2nd the  % of votes it got
        """
        self.get_k_neighbours(test_data, k)
        return self.get_class()
