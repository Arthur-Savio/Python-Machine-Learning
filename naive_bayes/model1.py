import math
import random
from collections import defaultdict
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

"""This a basic implementation of Naive Bayes using sklearn. """


class Model:
    def __init__(self):
        self.data = load_breast_cancer()
        self.label_names = self.data['target_names']
        self.labels = self.data['target']
        self.feature_names = self.data['feature_names']
        self.feature = self.data['data']

        self.train = None
        self.test = None
        self.train_labels = None
        self.test_labels = None

    def split_data(self):
        self.train, self.test, self.train_labels, self.test_labels = train_test_split(self.feature, self.labels, test_size=0.33, random_state=42)

    def init_model(self):
        #Initialize the model
        gnb = GaussianNB()

        #Train the model
        model = gnb.fit(self.train, self.train_labels)

        #Make predictions
        preds = gnb.predict(self.test)

        print("Accuracy: ", accuracy_score(self.test_labels, preds))
