import pandas as pd
from os.path import join
import numpy as np
import sys


def citeseer_to_graph(dirpath = 'citeseer_data'):

    cites = pd.read_csv(join(dirpath, "citeseer.cites"), header=None, sep="\t")
    content = pd.read_csv(join(dirpath, "citeseer.content"), header=None, sep="\t")
    key_docs = list(set(cites.iloc[:, 0]).union(set(cites.iloc[:, 1])))
    num_docs = len(key_docs)
    key_docs_map = {key_docs[i]: i for i in range(num_docs)}

    num_words = content.shape[1] - 2
    key_words = []
    for i in range(0, num_words):
        key_words.append('w'+str(i))
    key_words_map = {}

    j=0
    for i in range(num_docs,num_docs+num_words):
        key_words_map[(key_words[j])] = i
        j+=1

    nodes_map = dict(key_docs_map,**key_words_map)

    adj = np.zeros(shape=(num_docs+num_words, num_docs+num_words), dtype=np.uint8)

    n = 0
    for index, row in cites.iterrows():
        sys.stdout.write('\r' + "(Citeseer) Loading graph... " + str(int(n * 100 / (len(cites) + len(content)))) + '%')
        sys.stdout.flush()
        paper1 = key_docs_map[row[0]]
        paper2 = key_docs_map[row[1]]
        adj[paper2, paper1] = 1
        adj[paper1, paper2] = 1
        n+=1

    for index, row in content.iterrows():
        paper = row[0]
        sys.stdout.write('\r' + "(Citeseer) Loading graph... " + str(int(n * 100 / (len(cites) + len(content)))) + '%')
        sys.stdout.flush()
        for i in range(1,len(row)-2):

            if row[i] == 1:
                if str(paper) in key_docs_map:
                    adj[key_docs_map[str(paper)], num_docs+i] = 1
                    adj[num_docs+i, key_docs_map[str(paper)]] = 1

        n+=1
    sys.stdout.write('\r' + "Loading graph... " + str(100) + '%')
    sys.stdout.flush()
    sys.stdout.write('\r' + "Loading graph... DONE")
    print ('\n')

    nodes = {}
    for i in range(0,adj.shape[1]):
        nodes[(i)] = adj[:,i]

    return adj, nodes, nodes_map