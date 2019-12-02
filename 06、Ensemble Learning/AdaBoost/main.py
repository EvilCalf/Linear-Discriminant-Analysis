import numpy as np


class ThreshClassifier:
    def __init__(self):
        self.v = 0
        self.direction = 0

    def train(self, x, y, w):
        loss = 0
        min_loss = 1
        for v in np.arange(0.5, 10, 1):
            for direction in [0, 1]:
                if direction == 0:
                    mis = ((x < v) - 0.5) * 2 != y
                else:
                    mis = ((x > v) - 0.5) * 2 != y
                loss = sum(mis * w)
                if loss < min_loss:
                    min_loss = loss
                    self.v = v
                    self.direction = direction

        return min_loss

    def predict(self, x):
        if self.direction == 0:
            return ((x < self.v) - 0.5) * 2
        else:
            return ((x > self.v) - 0.5) * 2


class AdaBoost:
    def __init__(self, classifier=ThreshClassifier):
        self.classifier = classifier
        self.classifiers = []
        self.alphas = []

    def train(self, x, y):
        n = x.shape[0]
        M = 3
        w_m = np.array([1 / n] * n)
        for m in range(M):
            classifier_m = self.classifier()
            e_m = classifier_m.train(x, y, w_m)
            print(e_m)
            alpha_m = 1 / 2 * np.log((1 - e_m) / e_m)
            w_m = w_m * np.exp(-alpha_m * y * classifier_m.predict(x))
            z_m = np.sum(w_m)
            w_m = w_m / z_m
            print(w_m)
            self.classifiers.append(classifier_m)
            self.alphas.append(alpha_m)

    def predict(self, x):
        n = x.shape[0]
        results = np.zeros(n)
        for alpha, classifier in zip(self.alphas, self.classifiers):
            results += alpha * classifier.predict(x)
        return ((results > 0) - 0.5) * 2


x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
y = [1, 1, 1, -1, -1, -1, 1, 1, 1, -1]
x = np.array(x)
y = np.array(y)
ab = AdaBoost()
ab.train(x, y)
ab.predict(x)
