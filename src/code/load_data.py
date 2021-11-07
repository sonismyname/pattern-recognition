import numpy as np
'''Load data và xử lý một số kí tự đặc biệt'''
def processing(string):
    string = string.replace('\n', '')
    strings = string.split(',')
    arr = np.zeros(len(strings), dtype='int32')
    for i in range(len(strings)):
        if strings[i] != '?':
            arr[i] = int(strings[i])
        else:
            arr[i] = 0
    return arr[1:]

def read_data_breast_cancer(path):
    lines = None
    with open(path) as f:
        lines = f.readlines()
    data = []
    for i in range(len(lines)):
        line = processing(lines[i])
        data.append(line)
    return np.copy(data)
