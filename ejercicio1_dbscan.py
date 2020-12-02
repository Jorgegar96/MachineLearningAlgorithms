import pandas as pd
from Clustering import Clustering
import sys


def dbscan(argv):
    if len(argv) == 1:
        print("Executing script with default dataset ./Datasets/datos_1.csv")
        data = pd.read_csv("./Datasets/datos_1.csv")
    else:
        print(f"Executing script using specified dataset {argv[1]}")
        data = pd.read_csv(argv[1])
    cluster = Clustering()
    cluster.DBScan(data)


if __name__ == "__main__":
    dbscan(sys.argv)

