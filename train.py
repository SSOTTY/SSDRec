# coding=utf-8
from __future__ import division
from __future__ import print_function

import os, sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse
import tensorflow.compat.v1 as tf
import numpy as np
import time

from utils import *
from minibatch import MinibatchIterator
from model import DGRec

tf.disable_eager_execution()
tf.logging.set_verbosity(tf.logging.ERROR)
# seed = 2
# np.random.seed(seed)
# tf.set_random_seed(seed)

def evaluate(sess, model, minibatch, saver, val_or_test='val'):
    epoch_val_cost = []
    epoch_val_recall = []
    epoch_val_ndcg = []
    epoch_val_ndcg_k = []
    epoch_val_point = []
    n = 0
    while not minibatch.end_val(val_or_test):
        feed_dict = minibatch.next_val_minibatch_feed_dict(val_or_test)
        outs = sess.run([model.loss, model.sum_recall_20, model.sum_ndcg, model.sum_ndcg_20, model.point_count],
                        feed_dict=feed_dict)
        epoch_val_cost.append(outs[0])
        epoch_val_recall.append(outs[1])
        epoch_val_ndcg.append(outs[2])
        epoch_val_ndcg_k.append(outs[3])
        epoch_val_point.append(outs[4])
    return np.mean(epoch_val_cost), np.sum(epoch_val_recall) / np.sum(epoch_val_point), np.sum(epoch_val_ndcg) / np.sum(
        epoch_val_point), np.sum(epoch_val_ndcg_k) / np.sum(epoch_val_point)


def construct_placeholders(args):
    # Define placeholders
    placeholders = {
        'input_x': tf.placeholder(tf.int32, shape=(args.batch_size, args.max_length), name='input_session'),
        'input_y': tf.placeholder(tf.int32, shape=(args.batch_size, args.max_length), name='output_session'),
        'mask_y': tf.placeholder(tf.float32, shape=(args.batch_size, args.max_length), name='mask_x'),
        'dependence': tf.placeholder(tf.int32, shape=(args.num_items, args.neighbors_1_dependence+1, args.neighbors_2_dependence+1),
                                     name='dependence_network'),
        'support_nodes_layer1': tf.placeholder(tf.int32, shape=(args.batch_size * args.samples_1_social * args.samples_2_social),
                                               name='support_nodes_layer1'),
        'support_nodes_layer2': tf.placeholder(tf.int32, shape=(args.batch_size * args.samples_2_social),
                                               name='support_nodes_layer2'),
        'support_sessions_layer1': tf.placeholder(tf.int32, shape=(args.batch_size * args.samples_1_social * args.samples_2_social, \
                                                                   args.max_length), name='support_sessions_layer1'),
        'support_sessions_layer2': tf.placeholder(tf.int32, shape=(args.batch_size * args.samples_2_social, \
                                                                   args.max_length), name='support_sessions_layer2'),
        'support_lengths_layer1': tf.placeholder(tf.int32, shape=(args.batch_size * args.samples_1_social * args.samples_2_social),
                                                 name='support_lengths_layer1'),
        'support_lengths_layer2': tf.placeholder(tf.int32, shape=(args.batch_size * args.samples_2_social),
                                                 name='support_lengths_layer2'),
    }
    return placeholders


def train(args, data):
    adj_info = data[0]
    latest_per_user_by_time = data[1]
    user_id_map = data[2]
    item_id_map = data[3]
    train_df = data[4]
    valid_df = data[5]
    test_df = data[6]
    dependence_info = data[7]

    args.num_items = len(item_id_map) + 1
    args.num_users = len(user_id_map)
    placeholders = construct_placeholders(args)
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    ckpt_path = os.path.join(args.ckpt_dir, 'model.ckpt')

    minibatch = MinibatchIterator(adj_info,
                                  dependence_info,
                                  latest_per_user_by_time,
                                  [train_df, valid_df, test_df],
                                  placeholders,
                                  batch_size=args.batch_size,
                                  max_degree=args.max_degree,
                                  num_nodes=len(user_id_map),
                                  num_dependence_nodes=args.num_items,
                                  max_length=args.max_length,
                                  samples_1_2_social=[args.samples_1_social, args.samples_2_social],
                                  neighbors_1_2_dependence=[args.neighbors_1_dependence, args.neighbors_2_dependence],
                                  dependence_network=args.dependence_network)

    dgrec = DGRec(args, minibatch.sizes, placeholders)

    config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=3)

    total_steps = 0
    avg_time = 0.

    patience = 1000
    inc = 0
    early_stopping = False

    lowest_val_loss = 100
    start_time = time.time()
    for epoch in range(args.epochs):
        minibatch.shuffle()

        iter_cn = 0
        print('Epoch: %04d' % (epoch + 1))
        epoch_val_cost = []
        epoch_val_recall = []
        epoch_val_ndcg = []
        epoch_val_ndcg_k = []
        epoch_train_cost = []
        epoch_train_recall = []
        epoch_train_ndcg = []
        epoch_train_ndcg_k = []
        epoch_train_point = []

        while not minibatch.end() and not early_stopping:
            t = time.time()
            feed_dict = minibatch.next_train_minibatch_feed_dict()
            outs = sess.run(
                [dgrec.opt_op, dgrec.loss, dgrec.sum_recall_20, dgrec.sum_ndcg, dgrec.sum_ndcg_20, dgrec.point_count],
                feed_dict=feed_dict)
            train_cost = outs[1]
            epoch_train_cost.append(train_cost)
            epoch_train_recall.append(outs[2])
            epoch_train_ndcg.append(outs[3])
            epoch_train_ndcg_k.append(outs[4])
            epoch_train_point.append(outs[5])
            # Print results
            avg_time = (avg_time * total_steps + time.time() - t) / (total_steps + 1)

            if iter_cn % args.val_every == 0:
                ret = evaluate(sess, dgrec, minibatch, 'val')
                epoch_val_cost.append(ret[0])
                epoch_val_recall.append(ret[1])
                epoch_val_ndcg.append(ret[2])
                epoch_val_ndcg_k.append(ret[3])
                if ret[0] <= lowest_val_loss:
                    saver.save(sess, ckpt_path, global_step=total_steps)
                    highest_val_recall = ret[1]
                    highest_val_ndcg_k = ret[3]
                    lowest_val_loss = ret[0]
                    inc = 0
                    print("Iter:", '%d' % iter_cn,
                          "val_loss=", "{:.5f}".format(epoch_val_cost[-1]),
                          "val_recall@20=", "{:.5f}".format(epoch_val_recall[-1]),
                          "val_ndcg=", "{:.5f}".format(epoch_val_ndcg[-1]),
                          "val_ndcg_k=", "{:.5f}".format(epoch_val_ndcg_k[-1]),
                          "dump model!"
                          )
                else:
                    inc += 1
                if inc >= patience:
                    early_stopping = True
                    break

            if total_steps % args.print_every == 0:
                print("Iter:", '%d' % iter_cn,
                      "train_loss=", "{:.5f}".format(np.mean(epoch_train_cost)),
                      "train_recall@20=", "{:.5f}".format(np.sum(epoch_train_recall) / np.sum(epoch_train_point)),
                      "train_ndcg=", "{:.5f}".format(np.sum(epoch_train_ndcg) / np.sum(epoch_train_point)),
                      "train_ndcg_k=", "{:.5f}".format(np.sum(epoch_train_ndcg_k) / np.sum(epoch_train_point)),
                      "val_loss=", "{:.5f}".format(epoch_val_cost[-1]),
                      "val_recall@20=", "{:.5f}".format(epoch_val_recall[-1]),
                      "val_ndcg=", "{:.5f}".format(epoch_val_ndcg[-1]),
                      "val_ndcg_k=", "{:.5f}".format(epoch_val_ndcg_k[-1]),
                      "time=", "{:.5f}s".format(avg_time))
                sys.stdout.flush()
            total_steps += 1
            iter_cn += 1
        if early_stopping:
            print('Early stop at epoch: {}, total training steps: {}'.format(epoch, total_steps))
            break
    end_time = time.time()
    print('-----------{} seconds per batch iteration-------------'.format((end_time - start_time) / total_steps))
    print('Parameter settings: {}'.format(args.ckpt_dir))
    print('Optimization finished!\tStart testing...')
    ret = evaluate(sess, dgrec, minibatch, 'test')
    print('Test results:',
          # '\tLoss:{}'.format(ret[0]),
          '\tRecall@20:{}'.format(highest_val_recall),
          '\tNDCG@20:{}'.format(highest_val_ndcg_k))


class Args():
    training = True
    global_only = False
    local_only = False
    without_social = False
    dynamic_social = False
    dependence_network = False
    epochs = 20
    aggregator_type = 'attn'
    act = 'relu'
    batch_size = 200
    max_degree = 50
    num_users = -1
    num_items = 100
    concat = False
    learning_rate = 0.002
    hidden_size = 100
    embedding_size = 100
    emb_user = 100
    max_length = 30
    samples_1_social = 5
    samples_2_social = 10
    neighbors_1_dependence = 10
    neighbors_2_dependence = 5
    dim1 = 100
    dim2 = 100
    model_size = 'small'
    dropout = 0.2
    weight_decay = 0.
    print_every = 30
    val_every = 30
    glb_weight = 0.5
    local_weight = 0.5
    ckpt_dir = 'save/'


def parseArgs():
    args = Args()
    parser = argparse.ArgumentParser(description='DGRec args')
    parser.add_argument('--batch', default=200, type=int)
    parser.add_argument('--model', default='attn', type=str)
    parser.add_argument('--act', default='relu', type=str)
    parser.add_argument('--degree', default=50, type=int)
    parser.add_argument('--lr', default=0.002, type=float)
    parser.add_argument('--hidden', default=100, type=int)
    parser.add_argument('--embi', default=100, type=int)
    parser.add_argument('--embu', default=100, type=int)
    parser.add_argument('--samples1_social', default=5, type=int)
    parser.add_argument('--samples2_social', default=10, type=int)
    parser.add_argument('--neighbors1_dependence', default=10, type=int)
    parser.add_argument('--neighbors2_dependence', default=5, type=int)
    parser.add_argument('--dim1', default=100, type=int)
    parser.add_argument('--dim2', default=100, type=int)
    parser.add_argument('--dropout', default=0.2, type=float)
    parser.add_argument('--l2', default=0., type=float)
    parser.add_argument('--decay_steps', default=400, type=int)
    parser.add_argument('--decay_rate', default=0.98, type=float)
    parser.add_argument('--local', default=0, type=int)
    parser.add_argument('--glb', default=0, type=int)
    parser.add_argument('--glb_weight', default=0.5, type=int)
    parser.add_argument('--local_weight', default=0.5, type=int)
    parser.add_argument('--without_social', default=0, type=int)
    parser.add_argument('--dynamic_social', default=0, type=int)
    parser.add_argument('--dependence_network', default=0, type=int)
    new_args = parser.parse_args()

    args.batch_size = new_args.batch
    args.aggregator_type = new_args.model
    args.act = new_args.act
    args.max_degree = new_args.degree
    args.learning_rate = new_args.lr
    args.hidden_size = new_args.hidden
    args.embedding_size = new_args.embi
    args.emb_user = new_args.embu
    args.samples_1_social = new_args.samples1_social
    args.samples_2_social = new_args.samples2_social
    args.neighbors_1_dependence = new_args.neighbors1_dependence
    args.neighbors_2_dependence = new_args.neighbors2_dependence
    args.dim1 = new_args.dim1
    args.dim2 = new_args.dim2
    args.dropout = new_args.dropout
    args.weight_decay = new_args.l2
    args.decay_steps = new_args.decay_steps
    args.decay_rate = new_args.decay_rate
    args.local_only = new_args.local
    args.global_only = new_args.glb
    args.local_weight = new_args.local_weight
    args.glb_weight = new_args.glb_weight
    args.without_social = new_args.without_social
    args.dynamic_social = new_args.dynamic_social
    args.dependence_network = new_args.dependence_network
    args.ckpt_dir = args.ckpt_dir + 'dgrec_batch{}'.format(args.batch_size)
    args.ckpt_dir = args.ckpt_dir + '_model{}'.format(args.aggregator_type)
    args.ckpt_dir = args.ckpt_dir + '_act{}'.format(args.act)
    args.ckpt_dir = args.ckpt_dir + '_maxdegree{}'.format(args.max_degree)
    args.ckpt_dir = args.ckpt_dir + '_hidden{}'.format(args.hidden_size)
    args.ckpt_dir = args.ckpt_dir + '_dropout{}'.format(args.dropout)
    args.ckpt_dir = args.ckpt_dir + '_l2reg{}'.format(args.weight_decay)
    args.ckpt_dir = args.ckpt_dir + '_global{}'.format(new_args.glb)
    args.ckpt_dir = args.ckpt_dir + '_local{}'.format(new_args.local)
    args.ckpt_dir = args.ckpt_dir + '_global_weight{}'.format(new_args.glb_weight)
    args.ckpt_dir = args.ckpt_dir + '_local_weight{}'.format(new_args.local_weight)
    args.ckpt_dir = args.ckpt_dir + '_without_social{}'.format(new_args.without_social)
    args.ckpt_dir = args.ckpt_dir + '_dynamic_social{}'.format(new_args.dynamic_social)
    args.ckpt_dir = args.ckpt_dir + '_ddependence_network{}'.format(new_args.dependence_network)
    return args


def main(argv=None):
    args = parseArgs()
    print('Loading training data..')
    data = load_data('data/data/')
    print("Training data loaded!")
    train(args, data)


if __name__ == '__main__':
    tf.app.run()
