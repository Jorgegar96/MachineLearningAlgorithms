import pandas as pd
from Classifiers import Classifiers
import sys


def knn(argv):
    if len(argv) == 1:
        print("Executing script with default datasets ./Datasets/os_training_data.csv and ./Datasets/os_testing_data.csv")
        training_data = pd.read_csv("./Datasets/os_training_data.csv")
        testing_data = pd.read_csv("./Datasets/os_testing_data.csv")
    elif len(argv) == 2:
        print(f"Executing script using specified training set {argv[1]} and the default testing set ./Datasets/os_testing_data.csv")
        training_data = pd.read_csv(argv[1])
        testing_data = pd.read_csv("./Datasets/os_testing_data.csv")
    else:
        print(f"Executing script using specified training set {argv[1]} and specified testing set {argv[2]}")
        training_data = pd.read_csv(argv[1])
        testing_data = pd.read_csv(argv[2])

    knn = Classifiers()
    knn.KNN(training_data, testing_data)


if __name__ == "__main__":
    knn(sys.argv)
