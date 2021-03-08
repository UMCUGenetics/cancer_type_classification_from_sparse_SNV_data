from __future__ import division
import os
import sys
import numpy as np
import pandas as pd

import tensorflow as tf

from keras import backend as K
from keras.models import load_model


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.

    copied from: https://stackoverflow.com/questions/45466020/how-to-export-keras-h5-to-tensorflow-pb
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph

def load_pb_model(model_path):
    """Loads a the Protocol Buffers cancer type classification model and creates a TensorFlow session for it."""
    graph = tf.Graph()
    cfg = tf.ConfigProto(gpu_options={'allow_growth':True})
    sess = tf.InteractiveSession(graph=graph, config=cfg)
    graph_def = tf.GraphDef()
    with tf.gfile.Open(model_path, "rb") as f:
        data = f.read()
        graph_def.ParseFromString(data)
        tf.import_graph_def(graph_def)
        return sess, graph

def T(graph, layer):
  """Helper for getting layer output tensor"""
  return graph.get_tensor_by_name(layer)

def supplement_graph(graph):
  """Supplement the cancer type classification graph with a gradients operator to compute the
  gradients for the prediction at a particular label (specified by a placeholder)
  with respect to the input.
  """
  with graph.as_default():
    label_index = tf.placeholder(tf.int32, [])
    for i in graph.get_operations():
        print(i.name)
    inp = T(graph, 'import/input_1:0')
    label_prediction = T(graph, 'import/dense_1/Softmax:0')[:, label_index]
    return inp, label_index, T(graph, 'import/dense_1/Softmax:0'), tf.gradients(label_prediction, inp)[0]

def make_predictions_and_gradients(sess, graph):
    """Returns a function that can be used to obtain the predictions and gradients
    from the cancer type classification network for a set of inputs.

    The returned function is meant to be provided as an argument to the integrated_gradients
    method.
    """
    inp, label_index, predictions, grads = supplement_graph(graph)
    run_graph = sess.make_callable([predictions, grads], feed_list=[inp, label_index])

    def f(samples_to_test, target_label_index):
        return run_graph(samples_to_test, target_label_index)

    return f

def top_label_id_and_score(sample_, preds_and_grads_fn):
  """Returns the label id and corresponding value of the object class that receives the highest softmax value.
  """
  # Evaluate the SOFTMAX output layer for the sample and
  # determine the label for the highest-scoring class
  preds, _ = preds_and_grads_fn([sample_], 0)
  id = np.argmax(preds[0])
  return id, preds[0][id]


def integrated_gradients(
        inp,
        target_label_index,
        predictions_and_gradients,
        baseline,
        steps=50):
    """Computes integrated gradients for a given network and prediction label.
    Integrated gradients is a technique for attributing a deep network's
    prediction to its input features. It was introduced by:
    https://arxiv.org/abs/1703.01365
    In addition to the integrated gradients tensor, the method also
    returns some additional debugging information for sanity checking
    the computation. See sanity_check_integrated_gradients for how this
    information is used.

    This method only applies to classification networks, i.e., networks
    that predict a probability distribution across two or more class labels.

    Access to the specific network is provided to the method via a
    'predictions_and_gradients' function provided as argument to this method.
    The function takes a batch of inputs and a label, and returns the
    predicted probabilities of the label for the provided inputs, along with
    gradients of the prediction with respect to the input. Such a function
    should be easy to create in most deep learning frameworks.

    Args:
      inp: The specific input for which integrated gradients must be computed.
      target_label_index: Index of the target class for which integrated gradients
        must be computed.
      predictions_and_gradients: This is a function that provides access to the
        network's predictions and gradients. It takes the following
        arguments:
        - inputs: A batch of tensors of the same same shape as 'inp'. The first
            dimension is the batch dimension, and rest of the dimensions coincide
            with that of 'inp'.
        - target_label_index: The index of the target class for which gradients
          must be obtained.
        and returns:
        - predictions: Predicted probability distribution across all classes
            for each input. It has shape <batch, num_classes> where 'batch' is the
            number of inputs and num_classes is the number of classes for the model.
        - gradients: Gradients of the prediction for the target class (denoted by
            target_label_index) with respect to the inputs. It has the same shape
            as 'inputs'.
      baseline: [optional] The baseline input used in the integrated
        gradients computation. If None (default), the all zero tensor with
        the same shape as the input (i.e., 0*input) is used as the baseline.
        The provided baseline and input must have the same shape.
      steps: [optional] Number of intepolation steps between the baseline
        and the input used in the integrated gradients computation. These
        steps along determine the integral approximation error. By default,
        steps is set to 50.
    Returns:
      integrated_gradients: The integrated_gradients of the prediction for the
        provided prediction label to the input. It has the same shape as that of
        the input.

      The following output is meant to provide debug information for sanity
      checking the integrated gradients computation.
      See also: sanity_check_integrated_gradients
      prediction_trend: The predicted probability distribution across all classes
        for the various (scaled) inputs considered in computing integrated gradients.
        It has shape <steps, num_classes> where 'steps' is the number of integrated
        gradient steps and 'num_classes' is the number of target classes for the
        model.
    """
    if baseline is None:
        baseline = 0 * inp
    assert (baseline.shape == inp.shape)

    # Scale input and compute gradients.
    scaled_inputs = [baseline + (float(i) / steps) * (inp - baseline) for i in range(0, steps + 1)]
    predictions, grads = predictions_and_gradients(scaled_inputs,
                                                   target_label_index)  # shapes: <steps+1>, <steps+1, inp.shape>

    # Use trapezoidal rule to approximate the integral.
    # See Section 4 of the following paper for an accuracy comparison between
    # left, right, and trapezoidal IG approximations:
    # "Computing Linear Restrictions of Neural Networks", Matthew Sotoudeh, Aditya V. Thakur
    # https://arxiv.org/abs/1908.06214
    grads = (grads[:-1] + grads[1:]) / 2.0
    avg_grads = np.average(grads, axis=0)
    integrated_gradients = (inp - baseline) * avg_grads  # shape: <inp.shape>
    return integrated_gradients, predictions, grads


if __name__ == '__main__':
    fold = int(sys.argv[1])
    model_filepath = str(sys.argv[2])
    data_file = str(sys.argv[3])
    driver_data_file = str(sys.argv[4])
    motif_data_file = str(sys.argv[5])
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

    # Force Tensorflow to use a single thread
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)

    K.set_session(sess)
    model = load_model(model_filepath)
    frozen_graph = freeze_session(K.get_session(), output_names=[out.op.name for out in model.outputs])
    ### Save the frozen tf graph into a Protocol Buffers file
    model_folder, model_filename = os.path.split(model_filepath)

    model_basename, file_extension = os.path.splitext(model_filepath)
    model_filename = model_basename.split("/")[len(model_basename.split("/"))-1]
    print(model_filename)
    print(model.summary())
    pb_model_filename = str(model_filename + ".pb")
    tf.train.write_graph(frozen_graph, model_folder, pb_model_filename, as_text=False)
    sess, graph = load_pb_model(os.path.join(model_folder,pb_model_filename))

    # Load the cancer labels
    # format: cancer type names in order, 1 cancer type per line,
    # order should follow the LabelEncoder() class order used for training
    labels = np.array(open("./label_vocab_for_integrated_gradients_FINAL.txt").read().split('\n'))

    # Make the predictions_and_gradients function
    cancermodel_predictions_and_gradients = make_predictions_and_gradients(sess, graph)

    # Load data
    data = pd.read_csv(data_file, index_col=[0])
    driver_data = pd.read_csv(driver_data_file, index_col=[0])
    motif_data = pd.read_csv(motif_data_file, index_col=[0])

    test_samples = pd.read_csv('./test_stratified_idx_pcawg.csv', index_col=[0])
    test_samples.columns = ['guid', 'split']
    test_samples = test_samples[test_samples.split == fold]
    test_data = data[data['guid'].isin(test_samples.guid)]
    test_data = test_data.sort_values(by=['guid'])
    test_guid = list(test_data['guid'])

    test_data_driver = driver_data[driver_data['guid'].isin(test_samples.guid)]
    test_data_driver = test_data_driver.sort_values(by=['guid'])
    test_data_motif = motif_data[motif_data['guid'].isin(test_samples.guid)]
    test_data_motif = test_data_motif.sort_values(by=['guid'])

    test_data = test_data.drop(['guid'], axis=1)
    test_data_driver = test_data_driver.drop(['guid'], axis=1)
    test_data_motif = test_data_motif.drop(['guid'], axis=1)

    x_test_bin = test_data.values
    x_test_driver = test_data_driver.values
    x_test_motif = test_data_motif.values
    x_test = np.concatenate((x_test_bin, x_test_driver, x_test_motif), axis=1)
    y_test = test_data.index

    # Get feature list
    features_ = list(test_data.columns)
    features_.extend(list(test_data_driver.columns))
    features_.extend(list(test_data_motif.columns))

    # Get baseline
    averages_and_zeros = [0.0 for column_ in list(test_data.columns.values)]
    averages_and_zeros.extend([0.0 for column_ in list(test_data_driver.columns.values)])
    averages_and_zeros.extend([0.0 for column_ in list(test_data_motif.columns.values)])

    bline = np.reshape(np.asarray(averages_and_zeros), (x_test.shape[1],))

    sample_attributions = list()
    index_list = list()
    for cnc_ in labels:
        print(cnc_)
        for sdx, s_label in enumerate(y_test):
            if cnc_ != s_label:
                continue
            actual_sample = x_test[sdx]
            index_list.append(test_guid[sdx])
            # Determine top label and score.
            top_label_id, score = top_label_id_and_score(actual_sample, cancermodel_predictions_and_gradients)
            if labels[top_label_id] != s_label:
                print("Prediction mismatch: {}".format(test_guid[sdx]))
                print("correct label: {}".format(s_label))
                print("predicted label: {}".format(labels[top_label_id]))

            # Compute attributions based on the integrated gradients method
            attributions, predictions, prediction_trend = integrated_gradients(actual_sample,
                                                                 top_label_id,
                                                                 cancermodel_predictions_and_gradients,
                                                                 baseline=bline,
                                                                 steps=200)

            sample_attributions.append(attributions)

    ### Save attributions into file
    attribution_data = pd.DataFrame(sample_attributions, columns=features_, index=index_list)
    attribution_data.to_csv("./feature_attributions/{}_fold{}_0_baseline.txt".format(model_filename,fold), index=True)