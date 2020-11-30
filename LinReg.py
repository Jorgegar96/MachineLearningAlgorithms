from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn import metrics


class LinReg:

    def run(self, training_data, testing_data):
        print("="*100)
        self.SimpleLinReg(training_data, testing_data)
        self.NormalizedLinReg(training_data, testing_data)
        self.RegularizedLinReg(training_data, testing_data)
        self.NormAndRegulLinReg(training_data, testing_data)
        print("="*100)

    def SimpleLinReg(self, training_data, testing_data):
        print("<-----Linear Regression without Normalization nor Lasso Regularization----->")
        self.train(LinearRegression(), training_data, testing_data)

    def NormalizedLinReg(self, training_data, testing_data):
        print("<-----Linear Regression with Normalization but without Lasso Regularization----->")
        training_data = self.normalize(training_data)
        testing_data = self.normalize(testing_data)
        self.train(LinearRegression(), training_data, testing_data)

    def RegularizedLinReg(self, training_data, testing_data):
        print("<-----Linear Regression without Normalization but with Lasso Regularization----->")
        self.train(Lasso(alpha=0.01), training_data, testing_data)

    def NormAndRegulLinReg(self, training_data, testing_data):
        print("<-----Linear Regression with Normalization but with Lasso Regularization----->")
        training_data = self.normalize(training_data)
        testing_data = self.normalize(testing_data)
        self.train(Lasso(alpha=0.01), training_data, testing_data)

    def train(self, model, training_data, testing_data):
        model.fit(training_data.loc[:, training_data.columns != "score"], training_data["score"])
        print(f"Regressors' Coefficients:")
        for i, reg in enumerate(training_data.columns):
            if reg != "score":
                print(f"{reg}-> {model.coef_[i]}")
        print(f"Bias Term: {model.intercept_}")
        predicted = model.predict(training_data.loc[:, training_data.columns != "score"])
        print(f"Training Data MSE: {metrics.mean_squared_error(training_data['score'], predicted)}")
        print(f"R^2 Score: {metrics.r2_score(training_data['score'], predicted)}")
        predicted = model.predict(testing_data.loc[:, testing_data.columns != "score"])
        print(f"Testing Data MSE: {metrics.mean_squared_error(testing_data['score'], predicted)}")

    def normalize(self, df):
        scaler = StandardScaler()
        scaler.fit(df.loc[:, df.columns != "score"])
        normal = scaler.transform(df.loc[:, df.columns != "score"])
        cols = [col for col in df.columns if col != 'score']
        normal = pd.DataFrame(data=normal, columns=cols)
        score = df["score"]
        normal = pd.concat([normal, score], axis=1)
        normal.head(10)
        return normal
