import torch
import os
import pickle
from torch.autograd import Variable
import mnist_on_plane
import numpy as np
from torch.utils.data.dataset import Dataset
import visdom


class MyDataset(Dataset):
    def __init__(self, dataset_path):
        with open(dataset_path, "rb") as f:
            graphs = pickle.load(f)

        self.x_data_nodes = torch.Tensor(list(g.nodes for g in graphs))
        self.y_data = torch.Tensor(list(g.label for g in graphs))
        self.len = len(graphs)

    def __getitem__(self, index):
        return self.x_data_nodes[index], self.y_data[index]

    def __len__(self):
        return self.len
class MyDataset_all(Dataset):
    # dataset dei grafi di tutte le immagini nella directory mnist_data (TROPPO PESANTE/LENTO)
    def __init__(self, dataset_path, type):

        self.graphs = []
        for root, dirnames, filenames in os.walk(dataset_path):
            for file_complete in filenames:
                filename = os.path.join(root, file_complete)
                with open(filename, "r") as f:
                    digit = os.path.basename(os.path.dirname(filename))

                    if type == 'mnist_on_plane':
                        self.graphs.append(mnist_on_plane.to_graph(filename, digit))
                    else:
                        # todo
                        pass

        self.x_data_nodes = torch.Tensor(list(g.nodes for g in self.graphs))
        self.y_data = torch.Tensor(list(g.label for g in self.graphs))
        self.len = len(self.graphs)

    def __getitem__(self, index):
        return self.x_data_nodes[index], self.y_data[index]

    def __len__(self):
        return self.len

def save_checkpoint(state, save_path, filename):
  filename = os.path.join(save_path, filename)
  torch.save(state, filename)
def load_checkpoint(model, optimizer, filename):

    if os.path.isfile(os.getcwd()+'\\' + filename):
        print("Loading checkpoint '{}'".format(os.getcwd()+'\\'+filename))
        checkpoint = torch.load(os.getcwd()+'\\'+filename)
        epoch = checkpoint['epoch']
        vis_i = checkpoint['vis_i']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("Loaded checkpoint '{}' (epoch {})".format(epoch, checkpoint['epoch']))
        return epoch, vis_i
    else:
        print("No checkpoint found at '{}'".format(os.getcwd()+'\\'+filename))
        return 0, 0
def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)
def adjust_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(net, dataloader, criterion, optimizer, epoch, save, vis_window=None, vis_i = 0):
    net.train()
    losses = []
    accuracies = []
    for i, data in enumerate(dataloader, 0):
        input, labels = data
        input, labels = Variable(input, requires_grad=True), Variable(labels.long())
        input = input.unsqueeze(dim=2)
        optimizer.zero_grad()
        output = net(input)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        acc = accuracy(output, labels.data)
        losses.append(loss.data)
        accuracies.append(acc)
        if save:
            save_checkpoint({'epoch': epoch + 1, 'vis_i': vis_i, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict(), }, '', "./checkpoints/checkpoint.pth")
        if vis_window is not None:
            vis = visdom.Visdom()
            vis.line(Y=torch.Tensor([loss]).unsqueeze(0),X=np.expand_dims(np.array([vis_i]), 0), win=vis_window, update='append')
            vis_i += 1
    return losses, accuracies, vis_i

def validate(net, dataloader, criterion):
    net.eval()
    losses = []
    accuracies = []
    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            input, labels = data
            input, labels = Variable(input), Variable(labels.long())
            input = input.unsqueeze(dim=2)
            output = net(input)
            loss = criterion(output, labels)
            losses.append(loss.data)
            acc = accuracy(output, labels.data)
            accuracies.append(acc)
    return losses, accuracies