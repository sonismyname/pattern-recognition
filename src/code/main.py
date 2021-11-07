import load_data
import split_data
from PCA import PCA, visualize
from Bayes_classification import *
from Matrix import Evalution_value, Evalution, Timeline_accuracy

path = '../data/breast-cancer-wisconsin.data'
data = load_data.read_data_breast_cancer(path)
# print(data)

#Tính PCA cho data:
data_ = PCA(data, k=8)
# Split data:
Data_train, Data_test = split_data.split_dataset(data_, random_state=10)
# print(Data_test)
bayes = Bayes_classify(Data_train)
bayes.fit()
# Test data
pre = np.zeros((len(Data_test)))
for i in range(len(Data_test)):
    pre[i] = bayes.predict(Data_test[i, :-1])
visualize(data_)
# print(Data_test[:10, -1:])
# el = Evalution_value(Data_test[:, -1:].reshape(len(Data_test)), pre)
el = Evalution(Data_test[:, -1:], pre)
print(el)
# accurancies = np.zeros(1)
# accurancies[0] = el['accuracy']
# Timeline_accuracy(accurancies)


