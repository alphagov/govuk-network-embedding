import argparse
import logging.config
import os
from datetime import datetime
import keras
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import stellargraph as sg
from keras import backend as K
from stellargraph import globalvar
from stellargraph.data import EdgeSplitter
from stellargraph.layer import link_classification
from stellargraph.layer.graphsage import *
from stellargraph.mapper import GraphSAGELinkGenerator


def load_data(edgefile, embeddings_file):
    """

    :param edgefile:
    :param embeddings_file:
    :return:
    """
    edgelist = pd.read_csv(edgefile, compression="gzip")
    edgelist["label"] = "go_to"

    node_data = pd.read_csv(embeddings_file, compression="gzip", index_col=0)
    node_data.drop("content_id", axis=1, inplace=True)

    feature_names = node_data.columns
    node_features = node_data[feature_names].values

    graph = nx.from_pandas_edgelist(edgelist[['source', 'target', 'label']], edge_attr="label")

    removed_edges = 0
    for nid, f in zip(node_data.index, node_features):
        if nid in graph.node.keys():
            graph.node[nid][globalvar.TYPE_ATTR_NAME] = "page"  # specify node type
            graph.node[nid]["feature"] = f
        else:
            removed_edges += 1
    print(removed_edges)

    return graph


def init_generators(graph, batch_size, num_samples):
    """

    :param graph:
    :param batch_size:
    :param num_samples:
    :return:
    """
    # Define an edge splitter on the original graph
    edge_splitter_test = EdgeSplitter(graph)
    # Randomly sample a fraction p=0.1 of all positive links, and same number of negative links, from G,
    # and obtain the reduced graph_test with the sampled links removed
    graph_test, edge_ids_test, edge_labels_test = edge_splitter_test.train_test_split(p=0.1,
                                                                                      method="global",
                                                                                      keep_connected=True)

    # Define an edge splitter on the reduced graph graph_test
    edge_splitter_train = EdgeSplitter(graph_test)
    # Randomly sample a fraction p=0.1 of all positive links, and same number of negative links, from graph,
    # and obtain the reduced graph graph_train with the sampled links removed:
    graph_train, edge_ids_train, edge_labels_train = edge_splitter_train.train_test_split(p=0.1,
                                                                                          method="global",
                                                                                          keep_connected=True)

    graph_train = sg.StellarGraph(graph_train, node_features="feature")
    graph_test = sg.StellarGraph(graph_test, node_features="feature")

    train_gen = GraphSAGELinkGenerator(graph_train, batch_size, num_samples).flow(edge_ids_train,
                                                                                  edge_labels_train,
                                                                                  shuffle=True)
    test_gen = GraphSAGELinkGenerator(graph_test, batch_size, num_samples).flow(edge_ids_test,
                                                                                edge_labels_test)

    return train_gen, test_gen


def compile_model(train_gen, num_samples, layer_sizes, output_act, edge_embedding_method, dropout, aggregator):
    """

    :param train_gen:
    :param num_samples:
    :param layer_sizes:
    :param output_act:
    :param edge_embedding_method:
    :param dropout:
    :param aggregator:
    :return:
    """
    assert len(layer_sizes) == len(num_samples)

    graphsage = GraphSAGE(layer_sizes=layer_sizes,
                          generator=train_gen,
                          bias=True,
                          dropout=dropout,
                          #                       normalize = None,
                          aggregator=aggregator)

    # Expose input and output sockets of graphsage, for source and destination nodes:
    x_inp_src, x_out_src = graphsage.default_model(flatten_output=False)
    x_inp_dst, x_out_dst = graphsage.default_model(flatten_output=False)
    # re-pack into a list where (source, destination) inputs alternate, for link inputs:
    x_inp = [x for ab in zip(x_inp_src, x_inp_dst) for x in ab]
    # same for outputs:
    x_out = [x_out_src, x_out_dst]

    prediction = link_classification(output_dim=1,
                                     output_act=output_act,
                                     edge_embedding_method=edge_embedding_method)(x_out)

    model = keras.Model(inputs=x_inp, outputs=prediction)

    model.compile(optimizer=keras.optimizers.Adam(lr=1e-3),
                  loss=keras.losses.binary_crossentropy,
                  metrics=[keras.metrics.binary_accuracy, f1])

    return model


def evaluate_model(model, train_gen, test_gen):
    """

    :param model:
    :param train_gen:
    :param test_gen:
    :return:
    """
    init_train_metrics = model.evaluate_generator(train_gen, verbose=1)

    print("\nTrain Set Metrics of the model:")
    for name, val in zip(model.metrics_names, init_train_metrics):
        print("\t{}: {:0.4f}".format(name, val))

    init_test_metrics = model.evaluate_generator(test_gen, verbose=1)

    print("\nTest Set Metrics of the model:")
    for name, val in zip(model.metrics_names, init_test_metrics):
        print("\t{}: {:0.4f}".format(name, val))


def fit_model(model, train_gen, test_gen, epochs):
    """

    :param model:
    :param train_gen:
    :param test_gen:
    :param epochs:
    :return:
    """
    history = model.fit_generator(train_gen,
                                  epochs=epochs,
                                  validation_data=test_gen,
                                  verbose=1,
                                  shuffle=True)
    return history


def f1(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """

    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def plot_history(history):
    """

    :param history:
    :return:
    """
    metrics = sorted(history.history.keys())
    metrics = metrics[:len(metrics) // 2]

    f, axs = plt.subplots(1, len(metrics), figsize=(12, 4))

    for m, ax in zip(metrics, axs):
        # summarize history for metric m
        ax.plot(history.history[m])
        ax.plot(history.history['val_' + m])
        ax.set_title(m)
        ax.set_ylabel(m)
        ax.set_xlabel('epoch')
        ax.legend(['train', 'test'], loc='upper right')


def setup_filename(batch_size, num_samples, layer_sizes, output_act,
                   edge_embedding_method,
                   dropout,
                   aggregator, epochs):
    """
    Produce a meaningful filename for the trained model, includes values of graphSAGE parameters and training
    hyperparameters.
    """
    return "{}_{}_{}_d{}_b{}_ns{}-{}_l{}-{}_e{}_{}".format(aggregator.__name__,
                                                           output_act,
                                                           edge_embedding_method,
                                                           dropout,
                                                           batch_size,
                                                           num_samples[0], num_samples[1],
                                                           layer_sizes[0], layer_sizes[0],
                                                           epochs,
                                                           datetime.now().strftime("%d%m%y")) + ".h5"


def main(edgefile, embeddings_file, batch_size=64, num_samples=[20, 10], layer_sizes=[128, 128], output_act="relu",
         edge_embedding_method="ip",
         dropout=0.1,
         aggregator=AttentionalAggregator, epochs=20):
    """

    :param edgefile:
    :param embeddings_file:
    :param batch_size:
    :param num_samples:
    :param layer_sizes:
    :param output_act:
    :param edge_embedding_method:
    :param dropout:
    :param aggregator:
    :param epochs:
    :return:
    """

    logger.info("Loading data...")
    input_graph = load_data(edgefile, embeddings_file)
    logger.info("Initializing train and test data generators...")
    train, test = init_generators(input_graph, batch_size, num_samples)
    logger.info("Compiling model...")
    model = compile_model(train, num_samples, layer_sizes, output_act, edge_embedding_method,
                          dropout,
                          aggregator)
    logger.info("Evaluating untrained model...")
    evaluate_model(model, train, test)
    filename = setup_filename(batch_size, num_samples, layer_sizes, output_act,
                              edge_embedding_method,
                              dropout,
                              aggregator, epochs)
    logger.info("Training model...")
    hist = fit_model(model, train, test, epochs)
    logger.info("Evaluating trained model...")
    evaluate_model(model, train, test)
    logger.info("Saving model: {}...".filename)
    model.save(os.path.join(MODELS_DIR, filename))
    logger.info("Plotting performance metrics for training...")
    plot_history(hist)


def setup_aggregator(agg):
    if "maxpool" in agg:
        return MaxPoolingAggregator
    elif "meanpool" in agg:
        return MeanPoolingAggregator
    elif "attent" in agg:
        return AttentionalAggregator
    elif "meanagg" in agg:
        return MeanAggregator
    else:
        print("Invalid aggregator, returning MaxPool")
        return MaxPoolingAggregator


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run graphsage.')
    parser.add_argument('train_file', help='Name of edge file to train on.')
    parser.add_argument('emb_file', help='Name of embeddings file to train on.')
    parser.add_argument('-b', '--batch_size', type=int,
                        default=64)
    parser.add_argument('-ns', '--neighbor_samples', type=int, nargs="*",
                        default=[20, 10])
    parser.add_argument('-ls', '--layer_size', type=int, nargs="*",
                        default=[50, 50],
                        help="The number of hidden features at each GraphSAGE layer")
    parser.add_argument('-o', '--output_act', type=str,
                        default="relu")
    parser.add_argument('-ebm', '--edge_embedding_method', type=str,
                        default="ip",
                        help="The method for combining node embeddings into edge embeddings")
    parser.add_argument('-d', '--dropout', type=float,
                        default=0.0)
    parser.add_argument('-a', '--aggregator', type=str,
                        default="MaxPool")
    parser.add_argument('-e', '--epochs', type=int,
                        default=20)

    args = parser.parse_args()

    LOGGING_CONFIG = os.getenv("LOGGING_CONFIG")
    logging.config.fileConfig(LOGGING_CONFIG)
    logger = logging.getLogger('train_model')

    DATA_DIR = os.getenv("DATA_DIR")
    MODELS_DIR = os.getenv("MODELS_DIR")
    content_api = os.path.join(DATA_DIR, "content_api")

    edges = os.path.join(DATA_DIR, "processed_network", "edges_graphsagetest_feb_01_18_doo_min15weight_wtext.csv.gz")
    embeddings = os.path.join(content_api, "training_node_data_fixd.csv.gz")

    logger.info("\n\t{}".format("\n\t".join(["{}: {}".format(arg, getattr(args, arg)) for arg in vars(args)])))
    
    main(edges,
         embeddings,
         batch_size=args.batch_size,
         num_samples=args.num_samples,
         layer_sizes=args.neighbor_samples,
         output_act=args.output_act,
         # : 'concat', 'mul', 'ip', 'l1', 'l2', or 'avg'"
         edge_embedding_method=args.edge_embedding_method,
         dropout=args.dropout,
         aggregator=setup_aggregator(args.aggregator.lower()),
         epochs=args.epochs)
