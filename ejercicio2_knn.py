import pandas as pd
from Classifiers import Classifiers
import sys


def knn(argv):
    if len(argv) == 1:
        training_data = pd.read_csv("./Datasets/os_training_data.csv")
        testing_data = pd.read_csv("./Datasets/os_testing_data.csv")
    elif len(argv) == 2:
        training_data = pd.read_csv(argv[1])
        testing_data = pd.read_csv("./Datasets/os_testing_data.csv")
    else:
        training_data = pd.read_csv(argv[1])
        testing_data = pd.read_csv(argv[2])

    knn = Classifiers()
    knn.KNN(training_data, testing_data)


if __name__ == "__main__":
    knn(sys.argv)
