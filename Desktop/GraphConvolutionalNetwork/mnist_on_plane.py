from __future__ import print_function, division
import cv2
import numpy as np
import mnist_utils
import graph_utils
import random
import os
from Graph import Graph


def grid_adj_matrix(grid_x,grid_y):
#https://stackoverflow.com/questions/16329403/how-can-you-make-an-adjacency-matrix-which-would-emulate-a-2d-grid
    n = grid_x * grid_y
    A = np.zeros((n, n))
    for r in range(0, grid_x):
        for c in range(0, grid_y):
            i = r * grid_y + c
            if c > 0: A[i - 1, i] = A[i, i - 1] = 1
            if r > 0: A[i - grid_y, i] = A[i, i - grid_y] = 1
    return A


def create_graph(filename, digit):

    img = cv2.imread(filename,0)
    img = np.divide(img, 255)
    rotated = mnist_utils.random_rotation(img)
    nodes = rotated.ravel()
    nodes = nodes.tolist()
    A = grid_adj_matrix(img.shape[0],img.shape[1])
    A, X = graph_utils.add_fictional_node(A,nodes,0)

    if digit == 9:
        digit = 6

    G = Graph(X,digit)
    return G


def generate_random_dataset(path,n):
    graph_list = []
    tot = 0
    for i in range(0, n):
        digit = random.randint(0, 9)
        if digit < 0 or digit > 9:
            exit(1)

        chosen = random.choice(os.listdir(path + '/' + str(digit)))
        G = create_graph(path + '/' + str(digit) + '/' + chosen, digit)
        graph_list.append(G)
        tot += 1
        print(' ' + str(i+1) + '/' + str(n) + "  label: " + str(digit))
    print("(MNIST on plane) Dataset created.")
    return graph_list, tot

def generate_balanced_dataset(path,n):
    graph_list=[]

    n_part1 = n // 9
    n_part2 = (n // 9) // 2
    tot = 0
    for digit in range(0,10):
        if digit == 6 or digit == 9:
            n_part = n_part2
        else:
            n_part = n_part1
        for i in range(0,n_part):

            chosen = random.choice(os.listdir(path + '/' + str(digit)))

            G = create_graph(path + '/' + str(digit) + '/' + chosen, digit)
            graph_list.append(G)
            tot += 1
            if digit == 9:
                digit = 6
            print(' ' + str(tot) + '/' + str(n) + "  label: " + str(digit))

    if n - tot != 0:
        missing = n - tot
        for i in range(0,missing):
            random_G, digit = mnist_utils.get_random_mnist(path)
            G = create_graph(random_G, digit)
            graph_list.append(G)
            tot += 1
            print(' ' + str(tot) + '/' + str(n) + "  label: " + str(digit))

    print("(MNIST on plane) Dataset created.")
    return graph_list, tot

