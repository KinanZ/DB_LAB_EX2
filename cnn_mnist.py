from __future__ import print_function

import argparse
import gzip
import json
import os
import pickle

import numpy as np
import tensorflow as tf
import tempfile

count = 0

def one_hot(labels):
    """this creates a one hot encoding from a flat vector:
    i.e. given y = [0,2,1]
     it creates y_one_hot = [[1,0,0], [0,0,1], [0,1,0]]
    """
    classes = np.unique(labels)
    n_classes = classes.size
    one_hot_labels = np.zeros(labels.shape + (n_classes,))
    for c in classes:
        one_hot_labels[labels == c, c] = 1
    return one_hot_labels


def mnist(datasets_dir='./data'):
    if not os.path.exists(datasets_dir):
        os.mkdir(datasets_dir)
    data_file = os.path.join(datasets_dir, 'mnist.pkl.gz')
    if not os.path.exists(data_file):
        print('... downloading MNIST from the web')
        try:
            import urllib
            urllib.urlretrieve('http://google.com')
        except AttributeError:
            import urllib.request as urllib
        url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        urllib.urlretrieve(url, data_file)

    print('... loading data')
    # Load the dataset
    f = gzip.open(data_file, 'rb')
    try:
        train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
    except TypeError:
        train_set, valid_set, test_set = pickle.load(f)
    f.close()

    test_x, test_y = test_set
    test_x = test_x.astype('float32')
    test_x = test_x.astype('float32').reshape(test_x.shape[0], 28, 28, 1)
    test_y = test_y.astype('int32')
    valid_x, valid_y = valid_set
    valid_x = valid_x.astype('float32')
    valid_x = valid_x.astype('float32').reshape(valid_x.shape[0], 28, 28, 1)
    valid_y = valid_y.astype('int32')
    train_x, train_y = train_set
    train_x = train_x.astype('float32').reshape(train_x.shape[0], 28, 28, 1)
    train_y = train_y.astype('int32')
    print('... done loading data')
    return train_x, one_hot(train_y), valid_x, one_hot(valid_y), test_x, one_hot(test_y)


def LeNet(x, lr, num_filters, filter_size):
    #Our CNN consists of two convolutional layers (16 3 × 3 filters and a stride of
    #1), each followed by ReLU activation function and a max pooling layer (pool size
    #2). After the convolution layers we add a fully connected layer with 128 units
    #and a softmax layer for the classification.

    # With some help from the tutorial https://www.tensorflow.org/tutorials/estimators/cnn

    pool_size=[2, 2]
    pool_strides=2

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=x,
        filters=num_filters,
        kernel_size=[filter_size, filter_size],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size = pool_size, strides = pool_strides)

    # Convolutional Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=num_filters,
        kernel_size=[filter_size, filter_size],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #2
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size = pool_size, strides = pool_strides)

    # Flatten tensor into a batch of vectors
    #pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * num_filters])
    pool2_flat = tf.contrib.layers.flatten(pool2, outputs_collections = None, scope = None)

    # Fully connected Layer
    dense = tf.layers.dense(inputs=pool2_flat, units=128, activation=tf.nn.relu)

    # Output Layer
    y_conv = tf.layers.dense(inputs=dense, units=10)
    
    return y_conv


def train_and_validate(x_train, y_train, x_valid, y_valid, num_epochs, lr, num_filters, batch_size, filter_size):
    # TODO: train and validate your convolutional neural networks with the provided data and hyperparameters
    # With some help from the tutorial https://chromium.googlesource.com/external/github.com/tensorflow/tensorflow/+/r0.7/tensorflow/g3doc/tutorials/mnist/pros/index.md
    # Start TensorFlow InteractiveSession
    global count

    # Start building the computation graph by creating nodes for the input images and output classes.
    x_image = tf.placeholder(tf.float32, [None, 28,28,1], name='x')
    y_ = tf.placeholder(tf.float32, [None, 10], name='y_')

    num_samples = x_train.shape[0]
    num_batches = num_samples // batch_size

    # Forward pass
    y_conv = LeNet(x_image, lr, num_filters, filter_size)

    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_,logits=y_conv)
    cross_entropy = tf.reduce_mean(cross_entropy)*100
    train_step = tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    
    saver = tf.train.Saver()
        
    with tf.Session() as sess:
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name = "accuracy")
        sess.run(tf.global_variables_initializer())
        learning_curve = np.zeros(num_epochs)
        
        for i in range(num_epochs):
            for batch in range(num_batches):
                x_batch = x_train[batch*batch_size:(batch+1)*batch_size]
                y_batch = y_train[batch*batch_size:(batch+1)*batch_size]
                
                train_step.run(session=sess, feed_dict={x_image: x_batch, y_: y_batch})

		        #train_accuracy = accuracy.eval(session=sess, feed_dict={x_image: x_train, y_: y_train})
            learning_curve[i] = 1 - accuracy.eval(session=sess, feed_dict={x_image:x_valid, y_:y_valid})
            print("step %d, training error %g"%(i, learning_curve[i] ))
        
        model = saver.save(sess, './models/' + str(count) + '.ckpt')
        count += 1
        print("Model saved in path: %s" % model)
     
    return learning_curve, model  # TODO: Return the validation error after each epoch (i.e learning curve) and your model


def test(x_test, y_test, model):
    # TODO: test your network here by evaluating it on the test data
    tf.reset_default_graph()
    graph = tf.get_default_graph()

    with tf.Session(graph = graph) as sess:
        saver = tf.train.import_meta_graph(model + '.meta')
        saver.restore(sess, model)

        accuracy = graph.get_tensor_by_name("accuracy:0")
        x_image = graph.get_tensor_by_name("x:0")
        y_ = graph.get_tensor_by_name("y_:0")
        test_error = 1 - accuracy.eval(session=sess, feed_dict={x_image: x_test, y_: y_test})
    
    return test_error

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", default="./", type=str, nargs="?",
                        help="Path where the results will be stored")
    parser.add_argument("--input_path", default="./", type=str, nargs="?",
                        help="Path where the data is located. If the data is not available it will be downloaded first")
    parser.add_argument("--learning_rate", default=1e-3, type=float, nargs="?", help="Learning rate for SGD")
    parser.add_argument("--num_filters", default=16, type=int, nargs="?",
                        help="The number of filters for each convolution layer")
    parser.add_argument("--batch_size", default=128, type=int, nargs="?", help="Batch size for SGD")
    parser.add_argument("--epochs", default=12, type=int, nargs="?",
                        help="Determines how many epochs the network will be trained")
    parser.add_argument("--run_id", default=0, type=int, nargs="?",
                        help="Helps to identify different runs of an experiments")
    parser.add_argument("--filter_size", default=3, type=int, nargs="?",
                        help="Filter width and height")
    args = parser.parse_args()

    # hyperparameters
    lr = args.learning_rate
    num_filters = args.num_filters
    batch_size = args.batch_size
    epochs = args.epochs
    filter_size = args.filter_size

    # train and test convolutional neural network
    x_train, y_train, x_valid, y_valid, x_test, y_test = mnist(args.input_path)

    learning_curve, model = train_and_validate(x_train, y_train, x_valid, y_valid, epochs, lr, num_filters, batch_size, filter_size)

    test_error = test(x_test, y_test, model)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["lr"] = lr
    results["num_filters"] = num_filters
    results["batch_size"] = batch_size
    results["filter_size"] = filter_size
    results["learning_curve"] = learning_curve
    results["test_error"] = test_error

    path = os.path.join(args.output_path, "results")
    os.makedirs(path, exist_ok=True)

    fname = os.path.join(path, "results_run_%d.json" % args.run_id)

    fh = open(fname, "w")
    #json.dump(results, fh)
    fh.close()

    lrs = [0.1, 0.01, 0.001, 0.0001]
    for i in range(len(lrs)):
        x_train, y_train, x_valid, y_valid, x_test, y_test = mnist(args.input_path)

        learning_curve, model = train_and_validate(x_train, y_train, x_valid, y_valid, epochs, lrs[i], num_filters, batch_size, filter_size)

        test_error = test(x_test, y_test, model)

        # save results in a dictionary and write them into a .json file
        results = dict()
        results["lr"] = lrs[i]
        results["num_filters"] = num_filters
        results["batch_size"] = batch_size
        results["learning_curve"] = learning_curve.tolist()
        results["test_error"] = test_error.tolist()

        path = os.path.join(args.output_path, "results_learning_rates")
        os.makedirs(path, exist_ok=True)

        fname = os.path.join(path, "results_run_" + str(i+1) + ".json")

        fh = open(fname, "w")
        json.dump(results, fh)
        fh.close()

    filters = [1, 3, 5, 7]
    for i in range(len(filters)):
        x_train, y_train, x_valid, y_valid, x_test, y_test = mnist(args.input_path)

        learning_curve, model = train_and_validate(x_train, y_train, x_valid, y_valid, epochs, lr, num_filters, batch_size, filters[i])

        test_error = test(x_test, y_test, model)

        # save results in a dictionary and write them into a .json file
        results = dict()
        results["lr"] = lr
        results["num_filters"] = num_filters
        results["batch_size"] = batch_size
        results["learning_curve"] = learning_curve.tolist()
        results["test_error"] = test_error.tolist()

        path = os.path.join(args.output_path, "results_filters")
        os.makedirs(path, exist_ok=True)

        fname = os.path.join(path, "results_run_" + str(i+1) + ".json")

        fh = open(fname, "w")
        json.dump(results, fh)
        fh.close()
