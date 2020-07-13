__authors__ = ['1494758', '1494603', '1490885']
__group__ = 'DL.15'

import numpy as np
import Kmeans as km
import KNN
from utils_data import read_dataset, visualize_k_means, visualize_retrieval
import matplotlib.pyplot as plt
import cv2
import time
import timeit


def Retrieval_by_color(list_img, list_img_color, color_query):
    returner = []

    for index, image in enumerate(list_img_color):
        if color_query in image:
            returner.append(list_img[index])

    return returner


def Retrieval_by_shape(list_img, list_img_shape, shape_query):
    returner = []

    for index, image in enumerate(list_img_shape):
        if shape_query in image:
            returner.append(list_img[index])

    return returner


def Retrieval_combined(list_img, list_img_color, list_img_shape, color_query, shape_query):
    returner = []

    for index, (img_color, img_shape) in enumerate(zip(list_img_color, list_img_shape)):
        if color_query in img_color and shape_query in img_shape:
            returner.append(list_img[index])

    return returner


def Kmean_statistics(obj_kmeans, Kmax):
    for k in range(2, Kmax):
        obj_kmeans.K = k
        obj_kmeans.fit()
        obj_kmeans.heuristic = obj_kmeans.heuristic_kmeans('WCD')
        visualize_k_means(obj_kmeans, [80, 60, 3])


def Get_shape_accuracy(knn_labels, test_class_labels):

    return (np.sum(np.array(sorted(knn_labels)) == np.array(sorted(test_class_labels))) / len(test_class_labels)) * 100


def Get_color_accuracy(kmeans_colors, test_color_labels):
    count = 0

    for test, ground_truth in zip(kmeans_colors, test_color_labels):
        test = set(test)
        if test == ground_truth:
            count += 1
        else:
            count_aux = 0
            for test_color in test:
                if test_color in ground_truth:
                    count_aux += 1

            count += count_aux / len(ground_truth)

    return (count / len(test_color_labels)) * 100


if __name__ == '__main__':

    # Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, \
    test_imgs, test_class_labels, test_color_labels = read_dataset(ROOT_FOLDER='./images/', gt_json='./images/gt.json')

    # List with all the existant classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    # You can start coding your functions here
    start = time.time()

    # color_chosen = input('Choose a color')
    # query_color = Retrieval_by_color(test_imgs, test_color_labels, 'White')
    # visualize_retrieval(query_color, len(query_color))

    # query_shape = Retrieval_by_shape(test_imgs, test_class_labels, 'Dresses')
    # visualize_retrieval(query_shape, len(query_shape))

    # query_combined = Retrieval_combined(test_imgs, test_color_labels, test_class_labels, 'White', 'Dresses')
    # visualize_retrieval(query_combined, 20)

    elem_kmeans = []
    elem_colors_kmeans = []
    for img in test_imgs:
        elem_kmeans.append(km.KMeans(img))
        elem_kmeans[-1].find_bestK_improved(4, 'FISHER_REAL_CENT')
        elem_colors_kmeans.append(km.get_colors(elem_kmeans[-1].centroids))

    # Kmean_statistics(km.KMeans(test_imgs[0]), 4)
    '''

    elem_knn = KNN.KNN(train_imgs, train_class_labels)
    # start = time.time()
    porcentaje_shape = Get_shape_accuracy(elem_knn.predict(test_imgs, 4), test_class_labels)
    # print("tiempo: ", time.time() - start)
    print(porcentaje_shape)
    
    ''''''
    for propio, solucion in zip(sorted(elem_colors_kmeans), sorted(test_color_labels)):
        print(propio)
        print(solucion)
        '''
    # print("ahora empiezo la función")
    porcentaje_color = Get_color_accuracy(elem_colors_kmeans, test_color_labels)
    print(porcentaje_color)

    print("tiempo: ", time.time() - start)
pass
