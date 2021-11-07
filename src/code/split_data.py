from sklearn.model_selection import train_test_split
'''Chia dữ liệu ra tập train và test với scale 8:2'''
def split_dataset(data, random_state=0):
    X_train, X_test = train_test_split(data, test_size=0.2, random_state=random_state)
    return X_train, X_test

