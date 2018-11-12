from graph_utils import *
from scipy.linalg import fractional_matrix_power
import torch


#polinomiale di chebyshev
def T(k, x):
    if k == 0:
        return 1
    elif k == 1:
        return x
    elif k > 1:
        return 2 * x * T(k - 1, x) - T(k - 2, x)
    else:
        return -1

def k_hop_neighbors(G, node, k):
    return nx.single_source_shortest_path_length(G, node, cutoff=k)

def get_node_values(G):
    node_values = []
    for node in g.nodes():
        node_values.append(G.node[node]['v'])

    node_values = np.asarray(node_values)
    return node_values

#"Convolution on Graph: A High-Order and Adaptive Approach" (Zhenpeng Zhou, Xiaocheng Li)
# Ni -> neighbors ()
def spatial_conv(A, X, print_nodes = False):

    print ("Spatial convolution..."),

    G = data_to_networkx_graph(A,X)

    H = G.copy()

    #(pag 2/8)
    for node in G.nodes:
        sum = 0
        bias = 0

        for nbr in G.neighbors(node):
            w = G[node][nbr]
            sum = sum + (w['weight'] * G.node[nbr]['v'])
        H.node[node]['v'] = sum + bias

    print ("DONE")
    if print_nodes:
        print (H.nodes(data=True))
    print ("k_hop_convolution: fictional node (number " + str(H.number_of_nodes() - 1) + ")")
    print (H.nodes[H.number_of_nodes() - 1])

    return np.array(nx.get_node_attributes(H,'v').values())


#"Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering" (Michael Defferrard, Xavier Bresson, Pierre Vandergheynst)
def chebyshev_conv(A,X, K, print_nodes = False, verbose = False):
    if verbose:
        print ("Chebyshev convolution..."),
    G = data_to_networkx_graph(A,X)

    #(pag 3/9)
    L = laplacian_mat(G)

    eigval, U_t = np.linalg.eig(L)

    H = G.copy()

    #accumulatore
    filter = 0

    max_eig = np.amax(eigval)
    Lt = ((2 * L) / max_eig) - np.eye(L.shape[0])

    order = K + 1
    coef = np.random.rand(order) #parametri da far imparare alla rete

    for i in range(0,order):
        filter += coef[i]*T(i,Lt)

    node_values = get_node_values(G)

    y = filter.dot(node_values)

    i = 0
    for node in H.nodes():
        H.node[node]['v'] = y[i]
        i+=1
    print()

    if verbose:
        print (H.nodes(data=True))
        print ("chebyshev_convolution: fictional node (number " + str(H.number_of_nodes() - 1) + ")")
        print (H.nodes[H.number_of_nodes() - 1])

    return np.array(list(nx.get_node_attributes(H,'v').values()))


#"Convolution on Graph: A High-Order and Adaptive Approach" (Zhenpeng Zhou, Xiaocheng Li)
def k_hop_conv(A,X,k,Wk,print_nodes = False, verbose = False):
    if verbose:
        print ("K-hop convolution..."),

    '''
    if len(X.shape) != 2:
        X = np.squeeze(X, axis=1)
    '''
    #X = np.asmatrix(X)

    Ak = A.copy()

    for i in range(0,k-1):
        Ak = np.dot(Ak.copy(),A.copy())

    Akt = np.asarray(np.asmatrix(np.matrix.clip(Ak+np.eye(Ak.shape[0],Ak.shape[1]), None, 1)))

    #bias
    Bk = np.transpose(np.zeros(np.asmatrix(X).shape[1]))

    #matrice parametri (?) (pag.4/8)
    Q = np.random.rand(X.shape[0], Akt.shape[1])

    #filtro adattivo (pag 3/8) (dipende dalla matrice di adiacenza e dal feature vector)
    g = fadp(Akt,X,Q, False)

    #(pag 3/8)

    Wkt = np.multiply(g,Wk)
    #Wkt = np.ones_like(Ak) #se impostati ad 1, il risultato dovrebbe essere la media dei nodi nel neighborhood

    #(pag 4/8)
    Lconvt = ((np.multiply(Wkt,Akt)).dot(X.transpose()) + Bk).astype(float)

    H = data_to_networkx_graph(A, X)

    i = 0
    for node in H.nodes():
        H.node[node]['v'] = Lconvt[i]
        i+=1

    if verbose:
        print ("DONE")
    if print_nodes:
        print (H.nodes(data=True))
    if verbose:
        print ("k_hop_convolution: fictional node (number " + str(H.number_of_nodes() - 1) + ")")
        print (H.nodes[H.number_of_nodes() - 1])

    return np.array(list(nx.get_node_attributes(H,'v').values()))

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def fadp(Akt,X, Q, linear = True):
    #"Convolution on Graph: A High-Order and Adaptive Approach" (Zhenpeng Zhou, Xiaocheng Li)
    #(pag 3/8)
    if linear:
        #filtro adattivo lineare
        #problemi di dimensione (l'output deve avere la dimensione di Akt)
        #res = np.concatenate((Akt,np.asmatrix(X)),0)
        res = np.hstack((Akt,np.transpose(np.asmatrix(X))))
        res = Q.dot(res)
        res = sigmoid(res)
    else: #product
        res = sigmoid((Akt.dot(X.transpose())).dot(Q)) #MISMATCH DIMENSIONI

    return res



def kipf_welling(norm_A,X,W):
    return torch.matmul(torch.matmul(norm_A, X), W)

def kipf_welling_norm(D,A):

    At = A + np.eye(A.shape[0], A.shape[1])
    return fractional_matrix_power(D,(-1/2)).dot(At).dot(fractional_matrix_power(D,(-1/2)))
