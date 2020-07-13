__authors__ = ['1494758', '1494603', '1490885']
__group__ = 'DL.15'

import numpy as np
import utils
import time


class KMeans:

    def __init__(self, X, K=1, options=None):
        """
         Constructor of KMeans class
             Args:
                 K (int): Number of cluster
                 options (dict): dictionary with options
            """
        self.num_iter = 0
        self.K = K
        self._init_X(X)
        self._init_options(options)  # DICT options

    def _init_X(self, X):
        """Initialization of all pixels, sets X as an array of data in vector form (PxD)
            Args:
                X (list or np.array): list(matrix) of all pixel values
                    if matrix has more than 2 dimensions, the dimensionality of the smaple space is the length of
                    the last dimension
        """
        if isinstance(X, float) is False:
            X = X.astype(np.float)

        if len(X.shape) > 2:
            X = X.reshape([X.shape[0] * X.shape[1], X.shape[2]])

        self.X = X

    def _init_options(self, options=None):
        """
        Initialization of options in case some fields are left undefined
        Args:
            options (dict): dictionary with options
        """
        if options is None:
            options = {}
        if not 'km_init' in options:
            options['km_init'] = 'random'
        if not 'verbose' in options:
            options['verbose'] = False
        if not 'tolerance' in options:
            options['tolerance'] = 0
        if not 'max_iter' in options:
            options['max_iter'] = np.inf
        if not 'fitting' in options:
            options['fitting'] = 'WCD'  # within class distance.

        # If your methods need any other parameter you can add it to the options dictionary
        self.options = options

    def _init_centroids(self):
        """
        Initialization of centroids
        """
        n_k = 0
        self.centroids = np.zeros([self.K, self.X.shape[1]])
        self.old_centroids = np.zeros([self.K, self.X.shape[1]])
        lista_puntos = []

        if self.options['km_init'].lower() == 'first':
            for fila in self.X:
                if n_k < self.K and fila.tolist() not in lista_puntos:
                    self.centroids[n_k] = fila
                    lista_puntos.append(fila.tolist())
                    n_k += 1

                if n_k >= self.K:
                    break

        elif self.options['km_init'].lower() == 'random':
            while n_k < self.K:
                np.random.seed()
                n_aleatorio = np.random.randint(0, len(self.X))

                if self.X[n_aleatorio].tolist() not in lista_puntos:
                    self.centroids[n_k] = self.X[n_aleatorio]
                    lista_puntos.append(self.X[n_aleatorio].tolist())
                    n_k += 1

        elif self.options['km_init'].lower() == 'custom':
            awesomeness_factor = 10
            centroid_selector = []

            while n_k < (self.K * awesomeness_factor):
                np.random.seed()
                n_aleatorio = np.random.randint(0, len(self.X))

                if self.X[n_aleatorio].tolist() not in lista_puntos:
                    centroid_selector.append(self.X[n_aleatorio])
                    lista_puntos.append(self.X[n_aleatorio].tolist())
                    n_k += 1

            centroid_selector = np.array(centroid_selector)
            dist_centroids = distance(centroid_selector, centroid_selector)
            dist_centroids = np.array([np.mean(dist_centroids[i][np.nonzero(dist_centroids[i])])
                                       for i in range(len(dist_centroids))])
            max_args = np.argsort(dist_centroids)[-self.K:]
            self.centroids = centroid_selector[max_args]
            # print("Esto son los k definitivos max", self.centroids)
            # print("Esto son las medias de las distancias de cada uno", dist_centroids)
            # print("Esto los argumentos de los máximos", np.argsort(dist_centroids)[-self.K:])

        elif self.options['km_init'].lower() == 'hyper':
            dict_colors = {'Red': [255, 0, 0], 'Orange': [255, 127, 80], 'Brown': [255, 228, 196],
                           'Yellow': [255, 255, 0], 'Green': [0, 255, 0], 'Blue': [0, 0, 255],
                           'Purple': [156, 0, 156], 'Pink': [255, 20, 147], 'Black': [0, 0, 0],
                           'Grey': [192, 192, 192], 'White': [255, 255, 255]}
            centroid_selector = []
            awesomeness_factor = 100

            for pixel in range(len(self.X), -1, awesomeness_factor):
                centroid = [utils.colors[element] for element in
                            np.argmax(utils.get_color_prob(np.array([self.X[pixel]])), axis=1)]

                if n_k >= self.K:
                    break

                if n_k < self.K and centroid not in lista_puntos:
                    centroid_selector.append(dict_colors[centroid[0]])
                    lista_puntos.append(centroid)
                    n_k += 1

            self.K = len(centroid_selector)
            self.centroids = np.zeros([self.K, self.X.shape[1]])
            self.centroids = np.array(centroid_selector)
            centroid_selector.clear()

        elif self.options['km_init'].lower() == 'order':
            dict_colors = {'Red': [255, 0, 0], 'Orange': [255, 127, 80], 'Brown': [255, 228, 196],
                           'Yellow': [255, 255, 0], 'Green': [0, 255, 0], 'Blue': [0, 0, 255],
                           'Purple': [156, 0, 156], 'Pink': [255, 20, 147], 'Black': [0, 0, 0],
                           'Grey': [192, 192, 192], 'White': [255, 255, 255]}
            nombre_color = ['White', 'Black', 'Blue', 'Grey', 'Green', 'Red', 'Orange', 'Brown', 'Purple', 'Pink',
                            'Yellow']

            for color in range(self.K):
                self.centroids[color] = dict_colors[nombre_color[color % len(nombre_color)]]

        lista_puntos.clear()

    def get_labels(self):
        """        Calculates the closest centroid of all points in X
        and assigns each point to the closest centroid
        """
        self.dist_matrix = distance(self.X, self.centroids)
        self.labels = np.argmin(self.dist_matrix, axis=1)  # Devuelve el índice del valor mínimo de la FILA

    def get_centroids(self):
        """
        Calculates coordinates of centroids based on the coordinates of all the points assigned to the centroid
        """
        self.old_centroids = np.copy(self.centroids)
        self.centroids = np.array([(np.mean(self.X[self.labels == k], axis=0)) for k in range(self.K)]).reshape(-1, 3)
        # self.centroids = [(self.X[self.labels == k].sum(0)) for k in range(self.K)] / np.bincount(
        # self.labels).reshape(-1, 1)

    def converges(self):
        """
        Checks if there is a difference between current and old centroids
        """
        return np.array_equal(self.centroids, self.old_centroids)

    def fit(self):
        """
        Runs K-Means algorithm until it converges or until the number
        of iterations is smaller than the maximum number of iterations.
        """
        self._init_centroids()

        while not self.converges() and self.num_iter < self.options['max_iter']:
            self.get_labels()
            self.get_centroids()
            self.num_iter += 1

    def withinClassDistance(self):
        """
         returns the within class distance of the current clustering
        """
        # distances = distance(self.X, self.centroids)
        # min_dist = np.amin(self.dist_matrix, axis=1)
        return np.sum(np.square(np.amin(self.dist_matrix, axis=1))) / len(self.dist_matrix)

    def interClassDistance(self):
        """ El objetivo es sacar la distancia de cada punto con la de todos los puntos que no son de su clase.
        Para ello lo que haremos es hacer la media de distancia de todos los puntos de un cluster referente todos
        los otros puntos, y luego dividir entre el número total de distancias (número de puntos """
        num_total_dist = 0
        total = 0
        dist_classes = []

        for k in range(self.K):
            # Con esta línea de abajo conseguiremos sacar las distancias de los puntos del cluster K con el resto
            dist_classes.append(np.min(np.array(distance(self.X[self.labels == k], self.X[self.labels != k])) ** 2))
            # Necesitamos saber el número total de distancias para calcular la media
            num_total_dist += (len(self.X[self.labels == k]) * len(self.X[self.labels != k]))
            # print(num_total_dist)

        # Hacemos el cuadrado de todas las distancias y los sumamos, como en WCD
        # dist_classes = np.sum(np.square(np.array(dist_classes)))  # crashea por algun motivo desconocido
        # print(dist_classes)
        '''for elem in dist_classes:
            total += np.sum(np.array(elem) ** 2)'''
        # print(total / num_total_dist)
        # Retornamos la media -> suma de los cuadrados de las distancias / número total de distancias calculadas
        return np.min(np.array(dist_classes))

    def heuristic_kmeans(self, fitting):

        if fitting == 'WCD':  # Within Class Distance
            return self.withinClassDistance()

        elif fitting == 'ICD':  # Inter Class Distance (MINIMUM VALUE)
            dist_centroids = distance(self.centroids, self.centroids)
            return np.min(np.square(dist_centroids[np.nonzero(dist_centroids)]))

        elif fitting == 'ICD_AVG':  # Inter Class Distance (AVERAGE VALUE)
            dist_centroids = distance(self.centroids, self.centroids)
            return np.mean(np.square(dist_centroids[np.nonzero(dist_centroids)]))

        elif fitting == 'ICD_REAL':  # Inter Class Distance (PIXELS TO PIXELS) (REAL VALUE)
            return self.interClassDistance()

        elif fitting == 'ICD_REAL_CENT':  # Inter Class Distance (PIXELS TO CENTROIDS) (REAL VALUE)
            max_dist = np.array(self.dist_matrix)
            max_dist = max_dist[max_dist != np.amin(max_dist, axis=1)]
            return np.sum(np.square(max_dist)) / len(max_dist)

        elif fitting == 'FISHER':  # WCD/ICD
            dist_centroids = distance(self.centroids, self.centroids)
            return self.withinClassDistance() / np.min(np.square(dist_centroids[np.nonzero(dist_centroids)]))

        elif fitting == 'FISHER_AVG':  # WCD/ICD_AVG
            dist_centroids = distance(self.centroids, self.centroids)
            return self.withinClassDistance() / np.mean(np.square(dist_centroids[np.nonzero(dist_centroids)]))

        elif fitting == 'FISHER_REAL':  # WCD/ICD_REAL
            return self.withinClassDistance() / self.interClassDistance()

        elif fitting == 'FISHER_REAL_CENT':  # WCD/ICD_REAL_CENT
            max_dist = np.array(self.dist_matrix)
            max_dist = max_dist[max_dist != np.amin(max_dist, axis=1)]
            icd = np.sum(np.square(max_dist)) / len(max_dist)
            return self.withinClassDistance() / icd

    def find_bestK(self, max_K):
        """
         sets the best k analysing the results up to 'max_K' clusters
        """
        self.find_bestK_improved(max_K)  # Tenemos el nuevo código más abajo

        pass

    def find_bestK_improved(self, max_K, fitting='WCD', llindar=20):
        """
         sets the best k analysing the results up to 'max_K' clusters
        """
        self.K = 2
        self.fit()
        old_heuristic = self.heuristic_kmeans(fitting)

        best_k = 2
        stop = False
        k = 3
        while k < max_K and not stop:
            self.K = k
            self.fit()
            new_heuristic = self.heuristic_kmeans(fitting)

            if (100 - (100 * new_heuristic / old_heuristic)) > llindar:
                best_k = k
                old_heuristic = new_heuristic
            else:
                self.heuristic = old_heuristic  # Creamos este atributo para utilizarlo en my_labeling
                stop = True

            k += 1

        self.K = best_k
        self.fit()

        pass


def distance(X, C):
    """
    Calculates the distance between each pixel and each centroid
    Args:
        X (numpy array): PxD 1st set of data points (usually data points)
        C (numpy array): KxD 2nd set of data points (usually cluster centroids points)

    Returns:
        dist: PxK numpy array position ij is the distance between the
        i-th point of the first set an the j-th point of the second set
    """
    return np.sqrt((X[:, 0, np.newaxis] - C[:, 0]) ** 2 + (X[:, 1, np.newaxis] - C[:, 1]) ** 2 + (
            X[:, 2, np.newaxis] - C[:, 2]) ** 2)


def get_colors(centroids):
    """
    for each row of the numpy matrix 'centroids' returns the color laber following the 11 basic colors as a LIST
    Args:
        centroids (numpy array): KxD 1st set of data points (usually centroind points)
    Returns:
        lables: list of K labels corresponding to one of the 11 basic colors
    """
    # matriu_P = utils.get_color_prob(centroids)
    # matriu_max = np.argmax(matriu_P, axis=1)
    # print(centroids)
    return sorted([utils.colors[element] for element in np.argmax(utils.get_color_prob(centroids), axis=1)])
