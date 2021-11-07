import load_data
import split_data
from PCA import PCA, visualize
from Bayes_classification import *
from Matrix import *

path = '../data/breast-cancer-wisconsin.data'
data = load_data.read_data_breast_cancer(path)
# Accurancy_plot
# list_acc = []
# for i in range(2, 10):
#     data_ = PCA(data, k=i)
#     # Split data:
#     Data_train, Data_test = split_data.split_dataset(data_, random_state=10)
#     # print(Data_test)
#     bayes = Bayes_classify(Data_train)
#     bayes.fit()
#     # Test data
#     pre = np.zeros((len(Data_test)))
#     for j in range(len(Data_test)):
#         pre[j] = bayes.predict(Data_test[j, :-1])
#     el = Evalution_value(Data_test[:, -1:].reshape(len(Data_test)), pre)
#     list_acc.append([i, el['accuracy']])
#
# list_acc = np.copy(list_acc)
# print(list_acc)
# Timeline_accuracy(list_acc)

pass
# f1_score
# list_f1_2 = []
# list_f1_4 = []
# for i in range(2, 10):
#     data_ = PCA(data, k=i)
#     # Split data:
#     Data_train, Data_test = split_data.split_dataset(data_, random_state=10)
#     # print(Data_test)
#     bayes = Bayes_classify(Data_train)
#     bayes.fit()
#     # Test data
#     pre = np.zeros((len(Data_test)))
#     for j in range(len(Data_test)):
#         pre[j] = bayes.predict(Data_test[j, :-1])
#     el = Evalution_value(Data_test[:, -1:].reshape(len(Data_test)), pre)
#     list_f1_2.append([i, el['f1_score_benign']])
#     list_f1_4.append([i, el['f1_score_malignant']])
#
# Timeline_f1_score(list_f1_2, list_f1_4)
pass
# precisiom
# list_f1_2 = []
# list_f1_4 = []
# for i in range(2, 10):
#     data_ = PCA(data, k=i)
#     # Split data:
#     Data_train, Data_test = split_data.split_dataset(data_, random_state=10)
#     # print(Data_test)
#     bayes = Bayes_classify(Data_train)
#     bayes.fit()
#     # Test data
#     pre = np.zeros((len(Data_test)))
#     for j in range(len(Data_test)):
#         pre[j] = bayes.predict(Data_test[j, :-1])
#     el = Evalution_value(Data_test[:, -1:].reshape(len(Data_test)), pre)
#     list_f1_2.append([i, el['presision_benign']])
#     list_f1_4.append([i, el['presision_malignant']])
#
# Timeline_f1_score(list_f1_2, list_f1_4)
pass
# recall

# precisiom
list_f1_2 = []
list_f1_4 = []
for i in range(2, 10):
    data_ = PCA(data, k=i)
    # Split data:
    Data_train, Data_test = split_data.split_dataset(data_, random_state=10)
    # print(Data_test)
    bayes = Bayes_classify(Data_train)
    bayes.fit()
    # Test data
    pre = np.zeros((len(Data_test)))
    for j in range(len(Data_test)):
        pre[j] = bayes.predict(Data_test[j, :-1])
    el = Evalution_value(Data_test[:, -1:].reshape(len(Data_test)), pre)
    list_f1_2.append([i, el['recall_benign']])
    list_f1_4.append([i, el['recall_malignant']])

Timeline_f1_score(list_f1_2, list_f1_4)



