import mnist_on_sphere
import mnist_on_plane
from utils import save_data
import os
import pathlib


def all_data_to_graphs():
    #genera una nuova struttura simile alla directory mnist_data, con tutte le immagini convertite in file.dat (grafi) (TROPPO PESANTE)
    input_path = './mnist_data/'
    output_path = './mnist_graphs/'
    for dirpath, dirnames, filenames in os.walk(input_path):
        structure = os.path.join(output_path, dirpath[len(input_path):])
        if not os.path.isdir(structure):
            os.mkdir(structure)
        else:
            print("Folder does already exits!")

    for root, dirnames, filenames in os.walk(input_path):
        for file_complete in filenames:
            filename = os.path.join(root,file_complete)
            with open(filename, "r") as f:
                digit = os.path.basename(os.path.dirname(filename))
                graph = mnist_on_plane.to_graph(filename, digit)
                parts = pathlib.Path(filename).parts[1:-1]
                file, file_extension = os.path.splitext(file_complete)
                new_filename = output_path + str(pathlib.Path(*parts)) + '/' + file + '.dat'
                save_data(graph, new_filename)

def random_graphs(path, n):
    dataset, tot = mnist_on_plane.generate_random_dataset(path,n)
    head, tail = os.path.split(path)
    save_data(dataset, './saved_datasets/mnist_on_plane/' + tail + '/random_' + str(tot) + '.dat')

def balanced_graphs(path, n):
    dataset, tot = mnist_on_plane.generate_balanced_dataset(path, n)
    head, tail = os.path.split(path)
    save_data(dataset, './saved_datasets/mnist_on_plane/' + tail + '/balanced_' + str(tot) + '.dat')

random_graphs('./mnist_data/testing',100)

