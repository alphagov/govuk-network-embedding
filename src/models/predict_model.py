import logging.config
import os

import matplotlib
import networkx as nx
import pandas as pd

matplotlib.use('PS')
from stellargraph import StellarGraph
from stellargraph import globalvar
from stellargraph.layer.graphsage import MeanPoolingAggregator
from stellargraph.mapper import GraphSAGELinkGenerator
from keras.models import load_model
import gzip


def batched_predict(test_set, feature_set, node_feats, batch_size, num_samples, workers=8, use_multiprocessing=True,
                    max_queue_size=100):
    g = nx.from_pandas_edgelist(test_set, edge_attr="label")

    for nid, nfeat in zip(feature_set.index, node_feats):
        if nid in g.node.keys():
            g.node[nid][globalvar.TYPE_ATTR_NAME] = "page"  # specify node type
            g.node[nid]["feature"] = nfeat

    g_predict = StellarGraph(g, node_features="feature")

    edge_ids_test = [(e[0], e[1]) for e in g_predict.edges()]

    predict_gen = GraphSAGELinkGenerator(g_predict, batch_size, num_samples).flow(edge_ids_test)

    logger.info("Predicting links...")
    predicts = model.predict_generator(predict_gen, verbose=1, workers=workers, use_multiprocessing=use_multiprocessing,
                                       max_queue_size=max_queue_size)

    print("Got maximum prediction: {}".format(max(predicts)))
    return [p[0] for p in predicts], edge_ids_test


def predict(test_set, feature_set, filename, workers=8, use_multiprocessing=True,
            max_queue_size=100, batch_size=1, num_samples=[20, 10], chunk_size=10000):
    logger.debug("Setting up batched prediction...")
    node_features = feature_set[feature_set.columns].values
    logger.debug("# features: {}".format(feature_set.shape[1]))
    test_set = test_set[['source', 'target', 'label']]
    logger.debug("# test set samples: {}".format(test_set.shape[0]))

    predict_batches = compute_batches(test_set.shape[0], chunk_size)

    # predictions = []
    # test_ids = []
    for pt, indices in enumerate(predict_batches):
        logger.info("Working on indices: {}:{}".format(indices[0], indices[1]))
        pred_batch, id_batch = batched_predict(test_set[indices[0]:indices[1]], feature_set, node_features, batch_size,
                                               num_samples, workers,
                                               use_multiprocessing,
                                               max_queue_size)

        logging.debug("First 10 predictions: {}".format(pred_batch[0:10]))
        logging.debug("# predictions: {}\n# test_set ids: {}".format(len(pred_batch), len(id_batch)))

        results_file = filename.replace(".csv.gz", "_results_pt{:02d}of{}.csv.gz".format(pt + 1, len(predict_batches)))
        logging.debug("Writing predictions and test_set ids to file: {}".format(results_file))

        with open(results_file, "w") as writer:
            writer.write("prediction\tedge_1\tedge_2\n".encode())
            for pred_i, id_i in zip(pred_batch, id_batch):
                writer.write("{}\t{}\t{}\n".format(pred_i, id_i[0], id_i[1]).encode())

        # predictions.extend(pred_batch)
        # test_ids.extend(id_batch)

    # return predictions, test_ids


def compute_batches(length, chunksize):
    return [[i, i + chunksize] if i + chunksize < length else [i, length - 1] for i in range(0, length, chunksize)]


if __name__ == "__main__":

    DATA_DIR = os.getenv("DATA_DIR")
    PREDICT_DIR = os.path.join(DATA_DIR, "predict_network")
    MODELS_DIR = os.getenv("MODELS_DIR")

    LOGGING_CONFIG = os.getenv("LOGGING_CONFIG")
    logging.config.fileConfig(LOGGING_CONFIG)
    logger = logging.getLogger('predict_model')

    predict_file = os.path.join(PREDICT_DIR, "predict_top50_vs_all.csv.gz")
    node_data_file = os.path.join(PREDICT_DIR, "node_data_labelled_tfidf_2K.csv.gz")
    model_file = os.path.join(MODELS_DIR, "graphsage_attentional_20meanpool_20e.h5")

    logger.info("Reading edge pairs for prediction: {}".format(predict_file))
    predict_test = pd.read_csv(predict_file, compression="gzip")

    logger.info("Reading node data (text features) for prediction: {}".format(node_data_file))
    node_data = pd.read_csv(node_data_file, compression="gzip", index_col=0)

    logger.info("Loading model: {}".format(model_file))
    model = load_model(model_file, custom_objects={'MeanPoolingAggregator': MeanPoolingAggregator})
    logger.debug(model.summary())

    # workers = 8, use_multiprocessing = True,
    # max_queue_size = 100, batch_size = 1, num_samples = [20, 10]
    # all_predictions, all_test_ids =
    predict(predict_test, node_data, filename=predict_file)

    # logging.debug("First 10 predictions: {}".format(all_predictions[0:10]))
    # logging.debug("# predictions: {}\n# test_set ids: {}".format(len(all_predictions), len(all_test_ids)))

    # predict_test['predictions'] = all_predictions
    # predict_test['test_ids'] = all_test_ids
    #
    # predict_output = predict_file.replace(".csv.gz", "_results.csv.gz")
    # logger.info("Saving results: {}".format(predict_output))
    # predict_test.to_csv(predict_output, compression="gzip")
