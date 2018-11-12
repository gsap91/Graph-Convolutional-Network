import pickle


def save_data(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)

def load_data(filename):

    with open(filename, "rb") as f:
        l = pickle.load(f)
    return l
