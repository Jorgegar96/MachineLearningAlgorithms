from sklearn import metrics, preprocessing
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import matplotlib.pyplot as plt


class Clustering:

    def run(self, dataset):
        self.KMeans(dataset)
        self.AggClustering(dataset)
        self.AggClusteringDist(dataset)
        self.DBScan(dataset)

    def KMeans(self, dataset):
        for i in range(1, 6):
            self.train(KMeans(n_clusters=i), dataset, f"K-Means {i} Cluster(s)")

    def AggClustering(self, dataset):
        for i in range(1, 6):
            self.train(AgglomerativeClustering(n_clusters=i), dataset, f"Agglomerative Clustering {i} Cluster(s)")

    def AggClusteringDist(self, dataset):
        thresholds = [0.25, 0.5, 0.75, 1.0, 1.5]
        for i in thresholds:
            self.train(AgglomerativeClustering(n_clusters=None, distance_threshold=i), dataset, f"Agglomerative Clustering by Distance {i}")

    def DBScan(self, dataset):
        eps = [0.25, 0.35, 0.5]
        min_samples = [5, 10, 15]
        cont = 0
        for i in eps:
            for j in min_samples:
                self.train(DBSCAN(eps=i, min_samples=j), dataset, f"DB Scan eps={eps} & min_samples={min_samples}")
                cont += 1

    def train(self, model, dataset, title):
        model.fit(dataset)
        self.plot(dataset, model.labels_, title)

    def plot(self, data, labels, title):
        plt.figure(figsize=(10, 7))
        plt.scatter(data["x"], data["y"], c=labels, cmap="viridis")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.title(title)
        plt.show()