import math
import numpy as np

def multivariateNormal(x, sigma, mu):
    tu = math.exp(((x - mu) @ np.linalg.inv(sigma) @ (x - mu).T) / (-2))
    mau = math.sqrt(2 * math.pi) ** x.shape[0] * math.sqrt(np.abs(np.linalg.det(sigma)))
    return tu / mau

class Bayes_classify:
    data = []
    sigma1 = None
    sigma2 = None
    u1 = None
    u2 = None

    def __init__(self, data):
        self.data = data

    def fit(self):
        data_class_2 = []
        data_class_4 = []

        for i in range(len(self.data)):
            if self.data[i, -1:] == 2:
                data_class_2.append(self.data[i, :-1])
            else:
                data_class_4.append(self.data[i, :-1])

        data_class_2 = np.copy(data_class_2)
        data_class_4 = np.copy(data_class_4)
        n_class_2 = len(data_class_2)
        n_class_4 = len(data_class_4)
        #class 2
        feature1 = data_class_2
        n_class_2 = len(data_class_2)
        prior1 = n_class_2 / len(self.data)
        u1 = feature1.mean(axis=0)
        Z_1 = feature1 - u1
        sigma_1 = (Z_1.T @ Z_1) / n_class_2
        #class 2:
        feature2 = data_class_4
        n_class_4 = len(data_class_4)
        prior2 = n_class_4 / len(self.data)
        u2 = feature2.mean(axis=0)
        Z_2 = feature2 - u2
        sigma_2 = (Z_2.T @ Z_2) / n_class_4
        self.sigma1 = sigma_1
        self.sigma2 = sigma_2
        self.u1 = u1
        self.u2 = u2
        pass

    def predict(self, x):
        b_1 = multivariateNormal(x, self.sigma1, self.u1)
        b_2 = multivariateNormal(x, self.sigma2, self.u2)
        if b_1 >= b_2:
            return 2
        else:
            return 4


