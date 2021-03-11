import tensorflow as tf
import random as rn
import numpy as np
import os

os.environ['PYTHONHASHSEED'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# Setting the seed for numpy-generated random numbers
np.random.seed(45)

# Setting the graph-level random seed.
tf.set_random_seed(1337)

rn.seed(73)

from keras import backend as K

session_conf = tf.ConfigProto(
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1)

# Force Tensorflow to use a single thread
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)

K.set_session(sess)
import math
import pandas as pd

import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import InputLayer, Input
from keras.layers import Dropout
from keras.layers import Dense
from keras.callbacks import TensorBoard, EarlyStopping
from keras.optimizers import Adam, Adamax
from keras.models import load_model
from keras import regularizers

import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence
from skopt.plots import plot_objective, plot_evaluations
from skopt.utils import use_named_args

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report

import sys

dim_learning_rate = Real(low=1e-4, high=1e-2, prior='log-uniform', name='learning_rate')

dim_weight_decay = Real(low=1e-3, high=0.5, prior='log-uniform', name='weight_decay')

dim_num_dense_layers = Integer(low=0, high=5, name='num_dense_layers')

dim_num_dense_nodes = Integer(low=5, high=1024, name='num_dense_nodes')

dim_activation = Categorical(categories=['relu', 'softplus'], name='activation')

dim_dropout = Real(low=1e-6, high=0.5, prior='log-uniform', name='dropout')

dimensions = [dim_learning_rate, dim_weight_decay, dim_dropout, dim_num_dense_layers, dim_num_dense_nodes,
              dim_activation]

default_paramaters = [1e-4, 1e-3, 1e-6, 0, 100, 'relu']


def log_dir_name(learning_rate, weight_decay, num_dense_layers, num_dense_nodes, activation):
    log_dir = "/hpc/compgen/users/adanyi/LiquidB_data/temp/crossvalidation{}_logs/{}__lr_{}_wd_{}_layers_{}_nodes{}_{}/".format(fold, output_name, learning_rate, weight_decay, num_dense_layers, num_dense_nodes, activation)
    ## make sure that dir exists
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def create_model(learning_rate, weight_decay, dropout, num_dense_layers, num_dense_nodes, activation):
    ###Define model here
    model = Sequential()
    model.add(InputLayer(input_shape=(input_size,)))
    for i in range(num_dense_layers):
        name = 'layer_dense_{0}'.format(i + 1)
        model.add(
            Dense(num_dense_nodes, activation=activation, name=name, kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Dropout(dropout))
    model.add(Dense(num_classes, activation='softmax'))
    optimizer = Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


@use_named_args(dimensions=dimensions)
def fitness(learning_rate, weight_decay, dropout, num_dense_layers, num_dense_nodes, activation):
    global best_accuracy
    print('learning rate: ', learning_rate)
    print('weight_decay: ', weight_decay)
    print('dropout', dropout)
    print('num_dense_layers: ', num_dense_layers)
    print('num_dense_nodes: ', num_dense_nodes)
    print('activation: ', activation)
    model = create_model(learning_rate=learning_rate, weight_decay=weight_decay, dropout=dropout,
                         num_dense_layers=num_dense_layers, num_dense_nodes=num_dense_nodes, activation=activation)
    log_dir = log_dir_name(learning_rate, weight_decay, num_dense_layers,
                           num_dense_nodes, activation)
    callback_log = TensorBoard(
        log_dir=log_dir,
        histogram_freq=0,
        batch_size=32,
        write_graph=True,
        write_grads=True,
        write_images=False)
    callbacks = [callback_log]
    history = model.fit(x=x_train, y=y_train, epochs=50, batch_size=32, validation_data=validation_data,
                        callbacks=callbacks)
    accuracy = history.history['val_acc'][-1]
    print('Accuracy: {0:.2%}'.format(accuracy))
    if accuracy > best_accuracy:
        model.save(path_best_model)
        best_accuracy = accuracy
    del model
    K.clear_session()
    return -accuracy


def to_table(report):
    report = report.splitlines()
    res = []
    header = [''] + report[0].split()
    for row in report[2:-4]:
        res.append(np.array(row.split()))
    return np.array(res), header


if __name__ == '__main__':
    fold = int(sys.argv[1])
    input_data_filename = sys.argv[2]
    input_driver_data_filename = sys.argv[3]
    input_motif_data_filename = sys.argv[4]
    output_name = sys.argv[5]

    path_best_model = './{}__crossvalidation{}_best_model.keras'.format(output_name, fold)
    best_accuracy = 0.0
    data = pd.read_csv("./{}.csv".format(input_data_filename), index_col=[0])
    driver_data = pd.read_csv("./{}.csv".format(input_driver_data_filename),
                       index_col=[0])
    motif_data = pd.read_csv("./{}.csv".format(input_motif_data_filename),
                       index_col=[0])


    ### Making training, test, validation data
    training_samples = pd.read_csv('./training_idx_pcawg.csv', index_col=[0])
    training_samples.columns = ['guid', 'split']
    training_samples = training_samples[training_samples.split == fold]
    frames = []
    frames_driver = []
    frames_motif = []
    for guid_ in training_samples.guid:
        frames.append(data[data['guid'].str.contains(guid_)])
        frames_driver.append(driver_data[driver_data['guid'].str.contains(guid_)])
        frames_motif.append(motif_data[motif_data['guid'].str.contains(guid_)])
    training_data = pd.concat(frames)
    training_data = training_data.sort_values(by=['guid'])
    training_data_driver = pd.concat(frames_driver)
    training_data_driver = training_data_driver.sort_values(by=['guid'])
    training_data_motif = pd.concat(frames_motif)
    training_data_motif = training_data_motif.sort_values(by=['guid'])


    validation_samples = pd.read_csv('./validation_idx_pcawg.csv', index_col=[0])
    validation_samples.columns = ['guid', 'split']
    validation_samples = validation_samples[validation_samples.split == fold]
    validation_data = data[data['guid'].isin(validation_samples.guid)]
    validation_data = validation_data.sort_values(by=['guid'])
    validation_data_driver = driver_data[driver_data['guid'].isin(validation_samples.guid)]
    validation_data_driver = validation_data_driver.sort_values(by=['guid'])
    validation_data_motif = motif_data[motif_data['guid'].isin(validation_samples.guid)]
    validation_data_motif = validation_data_motif.sort_values(by=['guid'])


    test_samples = pd.read_csv('./test_idx_pcawg.csv', index_col=[0])
    test_samples.columns = ['guid', 'split']
    test_samples = test_samples[test_samples.split == fold]
    test_data = data[data['guid'].isin(test_samples.guid)]
    test_data = test_data.sort_values(by=['guid'])
    test_data_driver = driver_data[driver_data['guid'].isin(test_samples.guid)]
    test_data_driver = test_data_driver.sort_values(by=['guid'])
    test_data_motif = motif_data[motif_data['guid'].isin(test_samples.guid)]
    test_data_motif = test_data_motif.sort_values(by=['guid'])

    training_data = training_data.drop(['guid'], axis=1)
    validation_data = validation_data.drop(['guid'], axis=1)
    test_data = test_data.drop(['guid'], axis=1)

    training_data_driver = training_data_driver.drop(['guid'], axis=1)
    validation_data_driver = validation_data_driver.drop(['guid'], axis=1)
    test_data_driver = test_data_driver.drop(['guid'], axis=1)

    training_data_motif = training_data_motif.drop(['guid'], axis=1)
    validation_data_motif = validation_data_motif.drop(['guid'], axis=1)
    test_data_motif = test_data_motif.drop(['guid'], axis=1)

    x_train_bin = training_data.values
    y_train = training_data.index
    x_val_bin = validation_data.values
    y_val = validation_data.index
    x_test_bin = test_data.values
    y_test = test_data.index

    x_train_driver = training_data_driver.values
    x_val_driver = validation_data_driver.values
    x_test_driver = test_data_driver.values

    x_train_motif = training_data_motif.values
    x_val_motif = validation_data_motif.values
    x_test_motif = test_data_motif.values

    x_train = np.concatenate((x_train_bin, x_train_driver, x_train_motif), axis=1)
    x_test = np.concatenate((x_test_bin, x_test_driver, x_test_motif), axis=1)
    x_val = np.concatenate((x_val_bin, x_val_driver, x_val_motif), axis=1)

    encoder = LabelEncoder()
    test_labels_names = y_test
    y_test = encoder.fit_transform(y_test)
    test_labels = y_test

    num_of_cancers = len(encoder.classes_)
    print("Num of cancers: {}".format(num_of_cancers))
    y_test = keras.utils.to_categorical(y_test, num_of_cancers)
    y_train = encoder.fit_transform(y_train)
    y_train = keras.utils.to_categorical(y_train, num_of_cancers)
    y_val = encoder.fit_transform(y_val)
    y_val = keras.utils.to_categorical(y_val, num_of_cancers)

    validation_data = (x_val, y_val)

    input_size = x_train.shape[1]
    num_classes = num_of_cancers

    ### Run Bayesian optimization
    search_result = gp_minimize(func=fitness, dimensions=dimensions, acq_func='EI', n_calls=200, x0=default_paramaters,
                                random_state=7, n_jobs=-1)

    # Save Best Hyperparameters
    hyps = np.asarray(search_result.x)
    np.save('./crossvalidation_results/{}__fold{}_hyperparams'.format(output_name, fold), hyps, allow_pickle=False)
    model = load_model(path_best_model)
    # Evaluate best model on test data
    result = model.evaluate(x=x_test, y=y_test)
    # Save best model
    model.save(
        './crossvalidation_results/{}__fold_{}_model.keras'.format(output_name,
                                                                                                        fold))

    Y_pred = model.predict(x_test)
    y_pred = np.argmax(Y_pred, axis=1)

    a = pd.Series(test_labels_names)
    b = pd.Series(test_labels)
    d = pd.DataFrame({'Factor': b, 'Cancer': a})
    d = d.drop_duplicates()
    d = d.sort_values('Factor')

    ## Create array of prediction probabilities
    p_df = pd.DataFrame(data=Y_pred, columns=d.Cancer, index=test_labels_names)

    ## Generate Confusion Matrix
    c_matrix = confusion_matrix(test_labels, y_pred)
    c_df = pd.DataFrame(data=c_matrix, index=d.Cancer, columns=d.Cancer)

    ## Generate Class Report
    c_report = classification_report(test_labels, y_pred, digits=3)
    r, header_ = to_table(c_report)

    r = pd.DataFrame(data=r, columns=header_, index=d.Cancer)

    samples = []
    for i in r.index:
        l = len(data[data.index == i])
        samples.append(l)

    r['sample_size'] = samples
    r.columns = [x.replace('-', '_') for x in r.columns]
    r['f1_score'] = [float(x) for x in r.f1_score]
    r.to_csv('./report/{}_class_report_fold{}.csv'.format(output_name, fold))
    c_df.to_csv('./report/{}_confusion_matrix_fold{}.csv'.format(output_name, fold))
    p_df.to_csv('./report/{}_probability_classification_{}.csv'.format(output_name, fold))