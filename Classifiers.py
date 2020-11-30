from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import pandas as pd
import time


class Classifiers:

    def KNN(self, training_data, testing_data):
        k = [1, 3, 5, 7, 9, 11, 13, 15]
        training_data = self.preProcess(training_data)
        testing_data = self.preProcess(testing_data)
        for k_val in k:
            message = f"k = {k_val}"
            self.train(KNeighborsClassifier(n_neighbors=k_val, algorithm='ball_tree'), training_data, testing_data, message)

    def LogisticRegression(self, training_data, testing_data):
        training_data = self.preProcess(training_data)
        testing_data = self.preProcess(testing_data)
        message = f"Logistic Regression"
        self.train(LogisticRegression(), training_data, testing_data, message)

    def train(self, model, training_data, testing_data, message):
        model.fit(training_data.loc[:, training_data.columns != 'class'], training_data['class'])
        print(f"------------Training Set Performance: {message}------------")
        self.test(model, training_data)
        print(f"------------Testing Set Performance: {message}-------------")
        self.test(model, testing_data)

    def test(self, model, testing_data):
        start = time.time()
        predicted = model.predict(testing_data.loc[:, testing_data.columns != 'class'])
        end = time.time()
        print(f"Prediction Time on Test Set: {(end - start) * 1000}ms")
        print("Accuracy:", metrics.accuracy_score(predicted, testing_data['class']))
        print(metrics.classification_report(predicted, testing_data['class'], labels=['windows']))

    def preProcess(self, data):
        df_dict = {}
        size = data.shape[0]
        for col in data.columns.tolist():
            df_dict[col] = []
            for i in range(size):
                if col != 'class':
                    df_dict[col].append(1 if data.loc[i, col] == "Si" else 0)
                else:
                    df_dict[col].append(data.loc[i, col])
        return pd.DataFrame(df_dict)
