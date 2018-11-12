from GCN import *
import citeseer
import nn_utils
from utils import *
import numpy as np
import os
import visdom
import time
from torch.utils.data import DataLoader

if __name__ == '__main__':
    print("Using Pytorch " + str(torch.__version__))

    mnist_dataset = True
    citeseer_dataset = False

    # MNIST
    if mnist_dataset:

        #Parametri-----------------------------------------------
        mnist_width = 28
        mnist_height = 28
        conv_channels = [1,16,64,128,256]
        linear_channels = [256,200,150,100,70,50,30,9]
        train = True
        validation = True
        test = False
        save = True
        load_optimizer = False
        visdom_plot = True
        criterion = nn.CrossEntropyLoss()
        start = 0
        learning_rate = 0.00555
        train_batch_size = 150
        val_batch_size = 50
        n_epochs = 1000
        optimizer_path = './checkpoints/checkpoint.pth'
        training_dataset_path = "./saved_datasets/mnist_on_plane/training/balanced_500.dat"
        test_dataset_path = "./saved_datasets/mnist_on_plane/testing/random_100.dat"
        #---------------------------------------------

        vis_i = 0

        #Inizializzazione rete-----------------------------
        net = GCN([mnist_width,mnist_height],conv_channels, linear_channels)
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

        if load_optimizer:
            if os.path.isfile(optimizer_path):
                start, vis_i = nn_utils.load_checkpoint(net, optimizer, optimizer_path)
                vis_i += 1
                for g in optimizer.param_groups:
                    g['lr'] = learning_rate
            else:
                print("No checkpoint found. Starting a new training.")
        else:
            print("Starting a new training.")
        #------------------------------------------------------------

        #Inizializzazione Visdom---------------------------------------
        vis = None
        loss_iter_graph = None
        loss_epoch_graph = None
        acc_epoch_graph = None

        if visdom_plot:
            vis = visdom.Visdom()
            win_name = "conv_ch: " + str(net.conv_channels) + " lin_ch: " + str(net.linear_channels) + " batch_size = " + str(train_batch_size)

            width = 500
            height = 500
            legend = ["train_loss"]

            layout = dict(title=win_name, width=width, height=height, legend=legend, showlegend=True)

            loss_iter_layout = layout.copy()
            loss_iter_layout['xlabel'] = "iteration"
            loss_iter_layout['ylabel'] = "loss"

            loss_epoch_layout = layout.copy()
            loss_epoch_layout['xlabel'] = "epoch"
            loss_epoch_layout['ylabel'] = "loss"

            acc_epoch_layout = layout.copy()
            acc_epoch_layout['xlabel'] = "epoch"
            acc_epoch_layout['legend'] = ["train_acc"]
            acc_epoch_layout['ylabel'] = "accuracy"

            if not load_optimizer:
                Y = torch.Tensor([0]).unsqueeze(0)
                X = np.column_stack(np.arange(0, 1))

                loss_iter_graph = vis.line(Y=Y, X=X, opts=loss_iter_layout, win=win_name + '_loss_iter')

                if validation:
                    Y = torch.Tensor([0, 0]).unsqueeze(0)
                    X = np.column_stack((np.arange(0, 1), np.arange(0, 1)))
                    loss_epoch_layout['legend'] = ["train_loss", "validation_loss"]
                    acc_epoch_layout['legend'] = ["train_acc", "validation_acc"]


                loss_epoch_graph = vis.line(Y=Y, X=X, opts=loss_epoch_layout, win=win_name + '_loss_epoch')
                acc_epoch_graph = vis.line(Y=Y,X=X, opts=acc_epoch_layout, win=win_name + '_acc_epoch')

        #--------------------------------------------------------------


        #Inizializzazione dataloaders----------
        training_dataset = nn_utils.MyDataset(training_dataset_path)
        training_loader = DataLoader(dataset=training_dataset, batch_size=train_batch_size, shuffle=True, num_workers=0)

        validation_loader = None
        if validation:
            validation_dataset = nn_utils.MyDataset(test_dataset_path)
            validation_loader = DataLoader(dataset=validation_dataset, batch_size=val_batch_size, shuffle=True, num_workers=0)
        #--------------------------------------


        val_losses = []
        val_accuracies = []

        for epoch in range(start, n_epochs):
            start_time = time.time()

            train_losses, train_accuracies, vis_i = nn_utils.train(net,training_loader,criterion,optimizer,epoch,save,win_name + '_loss_iter',vis_i)
            print('[%d] train_loss: %.3f' % (epoch + 1, np.asscalar(np.mean(train_losses))) + '  train_acc: %.3f' % (np.asscalar(np.mean(train_accuracies))), end=' ')

            if validation:
                val_losses, val_accuracies = nn_utils.validate(net,validation_loader,criterion)
                print(' val_loss: %.3f' % (np.asscalar(np.mean(val_losses))) + '  val_acc: %.3f' % (np.asscalar(np.mean(val_accuracies))), end=' ')

            elapsed_time = time.time() - start_time
            print(' elapsed_time: %.d' % (elapsed_time) + 's',end='\n')


            if visdom_plot:
                X = np.expand_dims(np.array([epoch]), 0)
                loss_Y = torch.Tensor([np.mean(train_losses)]).unsqueeze(0)
                acc_Y = torch.Tensor([np.mean(train_accuracies)]).unsqueeze(0)

                if validation:
                    X = np.expand_dims(np.array([epoch, epoch]), 0)
                    loss_Y = torch.Tensor([np.mean(train_losses), np.mean(val_losses)]).unsqueeze(0)
                    acc_Y = torch.Tensor([np.mean(train_accuracies), np.mean(val_accuracies)]).unsqueeze(0)

                vis.line(Y=loss_Y, X=X, win=win_name + '_loss_epoch', update='append')
                vis.line(Y=acc_Y, X=X, win=win_name + '_acc_epoch', update='append')



    # CITESEER #TODO
    '''
    if citeseer_dataset:
        adj, nodes, nodes_map = citeseer.citeseer_to_graph() # lento
        adj, nodes = add_fictional_node(adj,nodes,1)

        G = data_to_networkx_graph(adj,nodes)

        if plot_data:
            plt.figure("Networkx graph (Citeseer)")
            nx.draw_random(G,node_size=0.1,width=0.05)

        if plot_data:
            plt.show()
    '''
