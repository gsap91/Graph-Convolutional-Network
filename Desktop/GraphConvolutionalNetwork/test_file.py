#FILE DI TEST

import visdom
import numpy as np
import networkx as nx
import torch
import utils

'''
graphs = utils.load_data('./saved_datasets/mnist_on_sphere_training.dat')
pass
'''
resume = True

if not resume:
    vis = visdom.Visdom()
    prova = vis.line(Y=torch.Tensor([0,0]).unsqueeze(0),X=np.column_stack((np.arange(0,1), np.arange(0,1))), win="prova")

    for i in range(0,6):
        vis.line(Y=torch.randn(1,2), X=np.expand_dims(np.array([i,i]),0), win="prova", update='append')
else:
    vis = visdom.Visdom()
    for i in range(6,12):
        vis.line(Y=torch.randn(1,2), X=np.expand_dims(np.array([i,i]),0), win="prova", update='append')

