import pandas as pd
from Clustering import Clustering
import sys


def kmeans(argv):
    if len(argv) == 1:
        data = pd.read_csv("./Datasets/datos_1.csv")
    else:
        data = pd.read_csv(argv[1])
    cluster = Clustering()
    cluster.KMeans(data)


if __name__ == "__main__":
    kmeans(sys.argv)

