import tensorflow as tf
import random as rn
import numpy as np
import os

os.environ['PYTHONHASHSEED'] = '0'
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
from keras.models import Model
from keras.layers import InputLayer, Input, Dropout, Dense, Embedding, LSTM
from keras.layers.merge import concatenate


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

### DRIVER
dim_driver_weight_decay = Real(low=1e-3, high=0.5, prior='log-uniform', name='driver_weight_decay')

dim_driver_num_dense_layers = Integer(low=0, high=5, name='driver_num_dense_layers')

dim_driver_num_dense_nodes = Integer(low=5, high=1024, name='driver_num_dense_nodes')

dim_driver_activation = Categorical(categories=['relu', 'softplus'], name='driver_activation')

dim_driver_dropout = Real(low=1e-6, high=0.5, prior='log-uniform', name='driver_dropout')

### MOTIF
dim_motif_weight_decay = Real(low=1e-3, high=0.5, prior='log-uniform', name='motif_weight_decay')

dim_motif_num_dense_layers = Integer(low=0, high=5, name='motif_num_dense_layers')

dim_motif_num_dense_nodes = Integer(low=5, high=1024, name='motif_num_dense_nodes')

dim_motif_activation = Categorical(categories=['relu', 'softplus'], name='motif_activation')

dim_motif_dropout = Real(low=1e-6, high=0.5, prior='log-uniform', name='motif_dropout')


dimensions = [dim_learning_rate, dim_weight_decay, dim_dropout, dim_num_dense_layers, dim_num_dense_nodes,
              dim_activation, dim_driver_weight_decay, dim_driver_dropout, dim_driver_num_dense_layers, dim_driver_num_dense_nodes,
              dim_driver_activation, dim_motif_weight_decay, dim_motif_dropout, dim_motif_num_dense_layers, dim_motif_num_dense_nodes,
              dim_motif_activation]

default_paramaters = [1e-4, 1e-3, 1e-6, 0, 100, 'relu', 1e-3, 1e-6, 0, 100, 'relu', 1e-3, 1e-6, 0, 100, 'relu']


def log_dir_name(learning_rate, weight_decay, num_dense_layers, num_dense_nodes, activation):
    log_dir = "./crossvalidation{}_logs/consecutive_{}__lr_{}_wd_{}_layers_{}_nodes{}_{}/".format(fold, output_name, learning_rate, weight_decay, num_dense_layers, num_dense_nodes, activation)
    ## make sure that dir exists
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def create_model(learning_rate, weight_decay, dropout, num_dense_layers, num_dense_nodes, activation,
                 driver_weight_decay, driver_dropout, driver_num_dense_layers, driver_num_dense_nodes, driver_activation,
                 motif_weight_decay, motif_dropout, motif_num_dense_layers, motif_num_dense_nodes, motif_activation):
    ### Define model here
    main_input = Input(shape=(input_size,), name='main_input')
    name = 'main_layer_dense_{0}'.format(1)
    main_branch = Dense(num_dense_nodes, activation=activation, name=name,
                        kernel_regularizer=regularizers.l2(weight_decay))(main_input)
    main_branch = Dropout(dropout)(main_branch)
    for i in range(1,num_dense_layers):
        name = 'main_layer_dense_{0}'.format(i + 1)
        main_branch = Dense(num_dense_nodes, activation=activation, name=name, kernel_regularizer=regularizers.l2(weight_decay))(main_branch)
        main_branch = Dropout(dropout)(main_branch)

    driver_input = Input(shape=(input_driver_size,), name='driver_input')
    main_driver_branch = keras.layers.concatenate([main_branch, driver_input])
    name = 'driver_layer_dense_{0}'.format(1)
    main_driver_branch = Dense(driver_num_dense_nodes, activation=driver_activation, name=name,
                        kernel_regularizer=regularizers.l2(driver_weight_decay))(main_driver_branch)
    main_driver_branch = Dropout(driver_dropout)(main_driver_branch)
    for i in range(1,driver_num_dense_layers):
        name = 'driver_layer_dense_{0}'.format(i + 1)
        main_driver_branch = Dense(driver_num_dense_nodes, activation=driver_activation, name=name, kernel_regularizer=regularizers.l2(driver_weight_decay))(main_driver_branch)
        main_driver_branch = Dropout(driver_dropout)(main_driver_branch)

    motif_input = Input(shape=(input_motif_size,), name='motif_input')
    main_driver_motif_branch = keras.layers.concatenate([main_driver_branch, motif_input])
    name = 'motif_layer_dense_{0}'.format(1)
    main_driver_motif_branch = Dense(motif_num_dense_nodes, activation=motif_activation, name=name,
                        kernel_regularizer=regularizers.l2(motif_weight_decay))(main_driver_motif_branch)
    main_driver_motif_branch = Dropout(motif_dropout)(main_driver_motif_branch)
    for i in range(1,motif_num_dense_layers):
        name = 'motif_layer_dense_{0}'.format(i + 1)
        main_driver_motif_branch = Dense(motif_num_dense_nodes, activation=motif_activation, name=name, kernel_regularizer=regularizers.l2(motif_weight_decay))(main_driver_motif_branch)
        main_driver_motif_branch = Dropout(motif_dropout)(main_driver_motif_branch)

    predictions = Dense(num_classes, activation='softmax', name='output')(main_driver_motif_branch)
    optimizer = Adam(lr=learning_rate)
    model = Model(inputs=[main_input, driver_input, motif_input], outputs=predictions)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


@use_named_args(dimensions=dimensions)
def fitness(learning_rate, weight_decay, dropout, num_dense_layers, num_dense_nodes, activation, driver_weight_decay, driver_dropout, driver_num_dense_layers, driver_num_dense_nodes, driver_activation,
            motif_weight_decay, motif_dropout, motif_num_dense_layers, motif_num_dense_nodes, motif_activation):
    global best_accuracy
    # best_accuracy = 0.0
    print('learning rate: ', learning_rate)
    print('weight_decay: ', weight_decay)
    print('dropout', dropout)
    print('num_dense_layers: ', num_dense_layers)
    print('num_dense_nodes: ', num_dense_nodes)
    print('activation: ', activation)

    print('driver_weight_decay: ', driver_weight_decay)
    print('driver_dropout', driver_dropout)
    print('driver_num_dense_layers: ', driver_num_dense_layers)
    print('driver_num_dense_nodes: ', driver_num_dense_nodes)
    print('driver_activation: ', driver_activation)

    print('motif_weight_decay: ', motif_weight_decay)
    print('motif_dropout', motif_dropout)
    print('motif_num_dense_layers: ', motif_num_dense_layers)
    print('motif_num_dense_nodes: ', motif_num_dense_nodes)
    print('motif_activation: ', motif_activation)

    model = create_model(learning_rate=learning_rate, weight_decay=weight_decay, dropout=dropout,
                         num_dense_layers=num_dense_layers, num_dense_nodes=num_dense_nodes, activation=activation,
                         driver_weight_decay=driver_weight_decay, driver_dropout=driver_dropout,
                         driver_num_dense_layers=driver_num_dense_layers, driver_num_dense_nodes=driver_num_dense_nodes, driver_activation=driver_activation,
                         motif_weight_decay=motif_weight_decay, motif_dropout=motif_dropout,
                         motif_num_dense_layers=motif_num_dense_layers, motif_num_dense_nodes=motif_num_dense_nodes, motif_activation=motif_activation)
    log_dir = log_dir_name(learning_rate, weight_decay, num_dense_layers, num_dense_nodes, activation)
    callback_log = TensorBoard(
        log_dir=log_dir,
        histogram_freq=0,
        batch_size=32,
        write_graph=True,
        write_grads=True,
        write_images=False)
    callbacks = [callback_log]
    history = model.fit(x=[x_train, x_train_driver, x_train_motif], y=y_train, epochs=50, batch_size=32, validation_data=validation_data,
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
    for guid_ in training_samples.guid:
        frames.append(data[data['guid'].str.contains(guid_)])
    training_data = pd.concat(frames)
    training_data = training_data.sort_values(by=['guid'])

    validation_samples = pd.read_csv('./validation_idx_pcawg.csv', index_col=[0])
    validation_samples.columns = ['guid', 'split']
    validation_samples = validation_samples[validation_samples.split == fold]
    validation_data = data[data['guid'].isin(validation_samples.guid)]
    validation_data = validation_data.sort_values(by=['guid'])

    test_samples = pd.read_csv('./test_idx_pcawg.csv', index_col=[0])
    test_samples.columns = ['guid', 'split']
    test_samples = test_samples[test_samples.split == fold]
    test_data = data[data['guid'].isin(test_samples.guid)]
    test_data = test_data.sort_values(by=['guid'])

    training_data = training_data.drop(['guid'], axis=1)
    validation_data = validation_data.drop(['guid'], axis=1)
    test_data = test_data.drop(['guid'], axis=1)

    x_train = training_data.values
    y_train = training_data.index
    x_val = validation_data.values
    y_val = validation_data.index
    x_test = test_data.values
    y_test = test_data.index

    ### DRIVER Making training, test, validation data
    frames = []
    for guid_ in training_samples.guid:
        frames.append(driver_data[driver_data['guid'].str.contains(guid_)])
    driver_training_data = pd.concat(frames)
    driver_training_data = driver_training_data.sort_values(by=['guid'])

    driver_validation_data = driver_data[driver_data['guid'].isin(validation_samples.guid)]
    driver_validation_data = driver_validation_data.sort_values(by=['guid'])
    driver_test_data = driver_data[driver_data['guid'].isin(test_samples.guid)]
    driver_test_data = driver_test_data.sort_values(by=['guid'])

    driver_training_data = driver_training_data.drop(['guid'], axis=1)
    driver_validation_data = driver_validation_data.drop(['guid'], axis=1)
    driver_test_data = driver_test_data.drop(['guid'], axis=1)

    x_train_driver = driver_training_data.values
    y_train_driver = driver_training_data.index
    x_val_driver = driver_validation_data.values
    y_val_driver = driver_validation_data.index
    x_test_driver = driver_test_data.values
    y_test_driver = driver_test_data.index

    ### MOTIF Making training, test, validation data
    frames = []
    for guid_ in training_samples.guid:
        frames.append(motif_data[motif_data['guid'].str.contains(guid_)])
    motif_training_data = pd.concat(frames)
    motif_training_data = motif_training_data.sort_values(by=['guid'])

    motif_validation_data = motif_data[motif_data['guid'].isin(validation_samples.guid)]
    motif_validation_data = motif_validation_data.sort_values(by=['guid'])
    motif_test_data = motif_data[motif_data['guid'].isin(test_samples.guid)]
    motif_test_data = motif_test_data.sort_values(by=['guid'])

    motif_training_data = motif_training_data.drop(['guid'], axis=1)
    motif_validation_data = motif_validation_data.drop(['guid'], axis=1)
    motif_test_data = motif_test_data.drop(['guid'], axis=1)

    x_train_motif = motif_training_data.values
    y_train_motif = motif_training_data.index
    x_val_motif = motif_validation_data.values
    y_val_motif = motif_validation_data.index
    x_test_motif = motif_test_data.values
    y_test_motif = motif_test_data.index

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

    y_train_driver = encoder.fit_transform(y_train_driver)
    y_train_driver = keras.utils.to_categorical(y_train_driver, num_of_cancers)
    y_val_driver = encoder.fit_transform(y_val_driver)
    y_val_driver = keras.utils.to_categorical(y_val_driver, num_of_cancers)
    y_test_driver = encoder.fit_transform(y_test_driver)
    y_test_driver = keras.utils.to_categorical(y_test_driver, num_of_cancers)

    y_train_motif = encoder.fit_transform(y_train_motif)
    y_train_motif = keras.utils.to_categorical(y_train_motif, num_of_cancers)
    y_val_motif = encoder.fit_transform(y_val_motif)
    y_val_motif = keras.utils.to_categorical(y_val_motif, num_of_cancers)
    y_test_motif = encoder.fit_transform(y_test_motif)
    y_test_motif = keras.utils.to_categorical(y_test_motif, num_of_cancers)

    validation_data = ([x_val, x_val_driver, x_val_motif], y_val)

    input_size = x_train.shape[1]
    input_driver_size = x_train_driver.shape[1]
    input_motif_size = x_train_motif.shape[1]
    num_classes = num_of_cancers

    ### Run Bayesian optimization
    search_result = gp_minimize(func=fitness, dimensions=dimensions, acq_func='EI', n_calls=200, x0=default_paramaters, random_state=7, n_jobs=-1)

    # Save Best Hyperparameters
    hyps = np.asarray(search_result.x)
    np.save('./crossvalidation_results/{}__fold{}_hyperparams'.format(output_name, fold), hyps, allow_pickle=False)
    model = load_model(path_best_model)
    # Evaluate best model on test data
    result = model.evaluate(x=[x_test, x_test_driver, x_test_motif], y=y_test)
    # Save best model
    model.save('./crossvalidation_results/{}__fold_{}_model.keras'.format(output_name, fold))

    Y_pred = model.predict([x_test, x_test_driver, x_test_motif])
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