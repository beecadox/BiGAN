import csv
import pandas as pd


def load_csv(path, verbose=False):
    if verbose:
        print("Loading %s \n" % (path))
    return pd.read_csv(path, delim_whitespace=True, header=None).values

DATA_PATH = '/home/beecadox/Thesis/Dataset/dataset2019/shapenet'
classmap = load_csv(DATA_PATH + '/synsetoffset2category.txt')