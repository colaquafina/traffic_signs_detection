from pandas.io.parsers import read_csv
import data_process
import numpy as np
from pandas.io.parsers import read_csv
import pickle

signnames = read_csv("traffic_sign_classification\\signnames.csv").values[:, 1]
train_dataset_file = "traffic_sign_classification\\traffic-signs-data\\train.p"
test_dataset_file = "traffic_sign_classification\\traffic-signs-data\\test.p"


X_train, y_train = data_process.load_pickled_data(train_dataset_file, ['features', 'labels'])
print("Number of training examples in initial dataset =", X_train.shape[0])
X_test, y_test = data_process.load_pickled_data(test_dataset_file, ['features', 'labels'])
print("Number of testing examples in initial dataset =", X_test.shape[0])

train_preprocessed_dataset_file = "traffic_sign_classification\\traffic-signs-data\\train_preprocessed.p"
test_preprocessed_dataset_file = "traffic_sign_classification\\traffic-signs-data\\test_preprocessed.p"

X_train, y_train = data_process.load_and_process_data(train_dataset_file)
pickle.dump({
        "features" : X_train,
        "labels" : y_train
    }, open(train_preprocessed_dataset_file, "wb" ) )
print("Preprocessed training dataset saved in", train_preprocessed_dataset_file)


X_test, y_test = data_process.load_and_process_data(test_dataset_file)
pickle.dump({
        "features" : X_test,
        "labels" : y_test
    }, open(test_preprocessed_dataset_file, "wb" ) )
print("Preprocessed extended testing dataset saved in", test_preprocessed_dataset_file)

from sklearn.model_selection import train_test_split


import parameter
parameters = parameter.Parameters(
    # Data parameters
    num_classes = 43,
    image_size = (32, 32),
    # Training parameters
    batch_size = 64,
    max_epochs = 1001,
    log_epoch = 1,
    print_epoch = 1,
    # Optimisations
    learning_rate_decay = False,
    learning_rate = 0.0001,
    l2_reg_enabled = True,
    l2_lambda = 0.0001,
    early_stopping_enabled = True,
    early_stopping_patience = 100,
    resume_training = True,
    # Layers architecture
    conv1_k = 5, conv1_d = 32, conv1_p = 0.9,
    conv2_k = 5, conv2_d = 64, conv2_p = 0.8,
    conv3_k = 5, conv3_d = 128, conv3_p = 0.7,
    fc4_size = 1024, fc4_p = 0.5
)
import train

X_train, y_train = data_process.load_pickled_data(train_preprocessed_dataset_file, columns = ['features', 'labels'])
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.25)
X_test, y_test = data_process.load_pickled_data(test_preprocessed_dataset_file, columns = ['features', 'labels'])
train.train_model(parameters, X_train, y_train, X_valid, y_valid, X_test, y_test)
