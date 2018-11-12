import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def visualize_graph(G):
    print (G.nodes(data=True))
    nx.draw_random(G, node_size=0.5, width=0.1)
    plt.show()

def print_graph(G):
    print ("Nodes:")
    print (G.nodes(data=True))
    print ("Edges:")
    for (u, v, d) in G.edges(data='weight'):
        print ('({}, {}, {})'.format(u, v, d))

def degree_matrix(A):
    d = np.zeros(A.shape[0])
    for i in range(0, A.shape[0]):
        d[i] = np.sum(A[i])
    D = np.diag(d)
    return D

def laplacian_mat(g):
    D = degree_matrix(g)
    A = nx.adjacency_matrix(g).toarray()
    L = D - A
    return L

def eigenvalues_matrix(g):
    L = laplacian_mat(g)
    eigval, eigvec = np.linalg.eig(L)

    mat = np.diag(eigval)
    return mat


#dati di prova (networkx)
def examples(g, number):

    if number == 1:
        g.add_nodes_from([1, 2, 3, 4, 5, 6])
        g.node[1]['v'] = np.array([1,2])
        g.node[2]['v'] = np.array([4,7])
        g.node[3]['v'] = np.array([6,8])
        g.node[4]['v'] = np.array([8,2])
        g.node[5]['v'] = np.array([4,5])
        g.node[6]['v'] = np.array([3,7])

        g.add_edges_from([(1, 2),
                                   (1, 3),
                                   (1, 5),
                                   (2, 3),
                                   (3, 4),
                                   (4, 5),
                                   (4, 6),
                                   (5, 6)])

    elif number == 2:
        g.add_nodes_from([1, 2, 3])
        g.node[1]['v'] = 34
        g.node[2]['v'] = 17
        g.node[3]['v'] = 9

        g.add_weighted_edges_from([(1, 2, 0.6),
                                   (2, 3, 0.4),
                                   (1, 3, 0.5)])

    elif number == 3:
        g.add_nodes_from([1, 2, 3, 4, 5, 6, 7, 8, 9])
        g.add_weighted_edges_from([(1, 2, 0.1),
                                   (1, 3, 0.8),
                                   (2, 3, 0.2),
                                   (3, 4, 0.5),
                                   (4, 5, 0.2),
                                   (4, 6, 0.2),
                                   (5, 6, 0.1),
                                   (6, 7, 0.7),
                                   (7, 8, 0.6),
                                   (8, 9, 0.4),
                                   (4, 8, 0.9)])
        g.node[1]['v'] = 31
        g.node[2]['v'] = 82
        g.node[3]['v'] = 32
        g.node[4]['v'] = 35
        g.node[5]['v'] = 62
        g.node[6]['v'] = 70
        g.node[7]['v'] = 55
        g.node[8]['v'] = 29
        g.node[9]['v'] = 40

    elif number == 4:
        fv_1 = np.array([1, 4, 7])
        fv_2 = np.array([8, 2, 6])
        fv_3 = np.array([3, 2, 3])
        fv_4 = np.array([5, 9, 9])
        fv_5 = np.array([2, 3, 4])
        fv_6 = np.array([7, 1, 8])

        g.add_node(1, v=fv_1)
        g.add_node(2, v=fv_2)
        g.add_node(3, v=fv_3)
        g.add_node(4, v=fv_4)
        g.add_node(5, v=fv_5)
        g.add_node(6, v=fv_6)

        g.add_weighted_edges_from([(1, 2, 0.8),
                                   (1, 3, 0.6),
                                   (1, 5, 0.1),
                                   (2, 3, 0.8),
                                   (3, 4, 0.2),
                                   (4, 5, 0.8),
                                   (4, 6, 0.7),
                                   (5, 6, 0.8)])


def add_fictional_node(A, X, dataset=0):
    A = add_fictional_node_to_A(A)
    X = add_fictional_node_to_X(X)

    #CITESEER
    '''
    for i in range(0,len(X)):
        X[i] = A[:,i]
    X[(len(X))] = A[:,A.shape[1]-1]
    '''

    return A, X

def add_fictional_node_to_X(X):
    X.append(float(0))
    return X

def add_fictional_node_to_A(A):
    position = A.shape[0] + 1
    column = np.ones((position - 1, 1))
    row = np.ones((1, position))
    A = np.hstack([A, column])
    row[0][position - 1] = 0
    A = np.vstack([A, row])

    return A

def data_to_networkx_graph(A,X):
    if len(A.shape) != 2:
        A = np.squeeze(A, axis=0)
    G = nx.from_numpy_matrix(A)
    for i in range(0,len(X)):
        G.node[i]['v'] = X[i]

    return G