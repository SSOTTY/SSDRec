import tensorflow.compat.v1 as tf
import numpy as np

from aggregators import *
from layers import Dense

class DGRec(object):

    def __init__(self, args, support_sizes, placeholders):
        self.support_sizes = support_sizes
        if args.aggregator_type == "mean":
            self.aggregator_cls = MeanAggregator
        elif args.aggregator_type == "seq":
            self.aggregator_cls = SeqAggregator
        elif args.aggregator_type == "maxpool":
            self.aggregator_cls = MaxPoolingAggregator
        elif args.aggregator_type == "meanpool":
            self.aggregator_cls = MeanPoolingAggregator
        elif args.aggregator_type == "gcn":
            self.aggregator_cls = GCNAggregator
        elif args.aggregator_type == "attn":
            self.aggregator_cls = AttentionAggregator
        else:
            raise Exception("Unknown aggregator: ", self.aggregator_cls)
        self.input_x = placeholders['input_x']
        self.input_y = placeholders['input_y']
        self.mask_y = placeholders['mask_y']
        self.mask = tf.cast(self.mask_y, dtype=tf.float32)
        self.dependence = placeholders['dependence']
        self.point_count = tf.reduce_sum(self.mask)
        self.support_nodes_layer1 = placeholders['support_nodes_layer1']
        self.support_nodes_layer2 = placeholders['support_nodes_layer2']
        self.support_sessions_layer1 = placeholders['support_sessions_layer1']
        self.support_sessions_layer2 = placeholders['support_sessions_layer2']
        self.support_lengths_layer1 = placeholders['support_lengths_layer1']
        self.support_lengths_layer2 = placeholders['support_lengths_layer2']

        self.training = args.training
        self.concat = args.concat
        if args.act == 'linear':
            self.act = lambda x:x
        elif args.act == 'relu':
            self.act = tf.nn.relu
        elif args.act == 'elu':
            self.act = tf.nn.elu
        else:
            raise NotImplementedError
        self.batch_size = args.batch_size
        self.hidden_size = args.hidden_size
        self.samples_1_social = args.samples_1_social
        self.samples_2_social = args.samples_2_social
        self.num_samples_social = [self.samples_1_social, self.samples_2_social]
        self.neighbors_1_dependence = args.neighbors_1_dependence
        self.neighbors_2_dependence = args.neighbors_2_dependence
        self.num_neighbors_dependence = [self.neighbors_1_dependence, self.neighbors_2_dependence]
        self.n_items = args.num_items
        self.n_users = args.num_users
        self.emb_item = args.embedding_size
        self.emb_user = args.emb_user
        self.max_length = args.max_length
        self.model_size = args.model_size
        self.dropout = args.dropout
        self.dim1 = args.dim1
        self.dim2 = args.dim2
        self.weight_decay = args.weight_decay
        self.global_only = args.global_only
        self.local_only = args.local_only
        self.without_social = args.without_social
        self.dependence_network = args.dependence_network
        self.global_weight = args.glb_weight
        self.local_weight = args.local_weight
        

        self.dims = [self.hidden_size, args.dim1, args.dim2]
        self.dense_layers = []
        self.loss = 0
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.lr = tf.maximum(1e-5, tf.train.exponential_decay(args.learning_rate,
                                                            self.global_step,
                                                            args.decay_steps,
                                                            args.decay_rate,
                                                            staircase=True))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.build()

    def global_features(self):
        self.user_embedding = tf.get_variable('user_embedding', [self.n_users, self.emb_user],\
                                        initializer=tf.glorot_uniform_initializer())
        feature_layer1 = tf.nn.embedding_lookup(self.user_embedding, self.support_nodes_layer1)
        feature_layer2 = tf.nn.embedding_lookup(self.user_embedding, self.support_nodes_layer2)
        dense_layer = Dense(self.emb_user, 
                            self.hidden_size if self.global_only else round(self.hidden_size*self.global_weight),
                            act=tf.nn.relu,
                            dropout=self.dropout if self.training else 0.)
        self.dense_layers.append(dense_layer)
        feature_layer1 = dense_layer(feature_layer1)
        feature_layer2 = dense_layer(feature_layer2)
        print(feature_layer1)
        return [feature_layer2, feature_layer1]
    
    def local_features(self):
        '''
        Use the same rnn in decode function
        '''
        initial_state_layer1 = self.lstm_cell.zero_state(self.batch_size*self.samples_1_social*self.samples_2_social, dtype=tf.float32)
        initial_state_layer2 = self.lstm_cell.zero_state(self.batch_size*self.samples_2_social, dtype=tf.float32)
        inputs_1 = tf.nn.embedding_lookup(self.embedding, self.support_sessions_layer1)
        inputs_2 = tf.nn.embedding_lookup(self.embedding, self.support_sessions_layer2)
        outputs1, states1 = tf.nn.dynamic_rnn(cell=self.lstm_cell,
                                            inputs=inputs_1, 
                                            sequence_length=self.support_lengths_layer1,
                                            initial_state=initial_state_layer1,
                                            dtype=tf.float32)
        outputs2, states2 = tf.nn.dynamic_rnn(cell=self.lstm_cell,
                                            inputs=inputs_2, 
                                            sequence_length=self.support_lengths_layer2,
                                            initial_state=initial_state_layer2,
                                            dtype=tf.float32)
        # outputs: shape[batch_size, max_time, depth]
        local_layer1 = states1.h
        local_layer2 = states2.h
        dense_layer = Dense(self.hidden_size, 
                            self.hidden_size if self.local_only else round(self.hidden_size*self.local_weight),
                            act=tf.nn.relu,
                            dropout=self.dropout if self.training else 0.)
        self.dense_layers.append(dense_layer)
        local_layer1 = dense_layer(local_layer1)
        local_layer2 = dense_layer(local_layer2)
        print(local_layer1)
        return [local_layer2, local_layer1]

    def global_and_local_features(self):
        #global features
        global_feature_layer2, global_feature_layer1 = self.global_features()
        local_feature_layer2, local_feature_layer1 = self.local_features()
        global_local_layer2 = tf.concat([global_feature_layer2, local_feature_layer2], -1)
        global_local_layer1 = tf.concat([global_feature_layer1, local_feature_layer1], -1)
        return [global_local_layer2, global_local_layer1]

    def aggregate(self, hidden, dims, num_samples, support_sizes, 
            aggregators=None, name=None, concat=False, model_size="small"):
        """ At each layer, aggregate hidden representations of neighbors to compute the hidden representations 
            at next layer.
        Args:
            samples: a list of samples of variable hops away for convolving at each layer of the
                network. Length is the number of layers + 1. Each is a vector of node indices.
            input_features: the input features for each sample of various hops away.
            dims: a list of dimensions of the hidden representations from the input layer to the
                final layer. Length is the number of layers + 1.
            num_samples: list of number of samples for each layer.
            support_sizes: the number of nodes to gather information from for each layer.
            batch_size: the number of inputs (different for batch inputs and negative samples).
        Returns:
            The hidden representation at the final layer for all nodes in batch
        """


        # length: number of layers + 1
        hidden = hidden
        new_agg = aggregators is None
        if new_agg:
            aggregators = []
        for layer in range(len(num_samples)):
            # 一层一层聚合，第一层聚合二阶和一阶邻居，第二层只聚合一阶邻居
            if new_agg:
                dim_mult = 2 if concat and (layer != 0) else 1
                # aggregator at current layer
                if layer == len(num_samples) - 1:
                    aggregator = self.aggregator_cls(dim_mult*dims[layer], dims[layer+1], act=lambda x : x,
                            dropout=self.dropout if self.training else 0., 
                            name=name, concat=concat, model_size=model_size)
                else:
                    aggregator = self.aggregator_cls(dim_mult*dims[layer], dims[layer+1], act=self.act,
                            dropout=self.dropout if self.training else 0., 
                            name=name, concat=concat, model_size=model_size)
                aggregators.append(aggregator)
            else:
                aggregator = aggregators[layer]
            # hidden representation at current layer for all support nodes that are various hops away
            next_hidden = []
            # as layer increases, the number of support nodes needed decreases
            for hop in range(len(num_samples) - layer):
                dim_mult = 2 if concat and (layer != 0) else 1
                neigh_dims = [self.batch_size * support_sizes[hop], 
                              num_samples[len(num_samples) - hop - 1], 
                              dim_mult*dims[layer]]
                h = aggregator((hidden[hop],
                                tf.reshape(hidden[hop + 1], neigh_dims)))
                next_hidden.append(h)
                # print(h)
            hidden = next_hidden
#             print('hidden[0]:')
#             print(hidden[0])
#             print('hidden[1]:')
#             print(hidden[1])
#             print('hidden[1][1]:')
#             print(hidden[1][1])
        return hidden[0], aggregators

    def decode(self):
        self.lstm_cell = lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
        initial_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
        time_major_x = tf.transpose(self.input_x)
        inputs = tf.nn.embedding_lookup(self.embedding, time_major_x)
        outputs, state = tf.nn.dynamic_rnn(cell=lstm_cell,
                                            inputs=inputs, 
                                            initial_state=initial_state,
                                            time_major=True,
                                            dtype=tf.float32,
                                            scope='decode_rnn')
        # outputs: shape[max_time, batch_size, depth]
        slices = tf.split(outputs, num_or_size_splits=self.max_length, axis=0)
        return [tf.squeeze(t,[0]) for t in slices]

    def step_by_step(self, features_0, features_1_2, dims, num_samples, support_sizes, 
            aggregators=None, name=None, concat=False, model_size="small"):
        self.aggregators = None
        outputs = []
        for feature0 in features_0:
            hidden = [feature0, features_1_2[0], features_1_2[1]]
            output1, self.aggregators = self.aggregate(hidden, dims, num_samples, support_sizes,
                                        aggregators=self.aggregators, concat=concat, model_size=self.model_size)
            outputs.append(output1)
        return tf.stack(outputs, axis=0)

    def build(self):
        if self.dependence_network:
            if self.training:
                dependence_dropout = self.dropout
            else:
                dependence_dropout = 0
            self.item_emb_matrix = tf.get_variable('item_embedding', [self.n_items, self.emb_item],\
                                        initializer=tf.glorot_uniform_initializer())
            print(self.dependence)
            self.dependence_emb = tf.nn.embedding_lookup(self.item_emb_matrix, self.dependence)
            self.one_hop_user_emb = tf.expand_dims(self.dependence_emb[:, :, 0, :], axis=2)
            self.two_hop_user_emb = self.dependence_emb
            
            score1 = tf.reduce_sum(self.one_hop_user_emb * self.two_hop_user_emb, axis=3)  # 第一层的注意力得分
            # 第一层的注意力权重
            att1 = tf.nn.softmax(score1)
            att1 = tf.expand_dims(att1, axis=3)
            self.one_hop_user_aggregated_emb = tf.reduce_sum(self.two_hop_user_emb * att1, axis=2)
            self.W1 = tf.get_variable('W1', shape=[self.emb_item, self.emb_item],initializer=tf.glorot_uniform_initializer())
            self.one_hop_user_aggregated_emb = tf.nn.dropout(tf.nn.relu(tf.matmul(self.one_hop_user_aggregated_emb, self.W1)), 1-dependence_dropout)# 第一层物品的表示
            user_emb = tf.reshape(self.one_hop_user_aggregated_emb[:, 0, :], shape=(self.n_items, self.emb_item))
            self.user_emb = tf.reshape(user_emb, shape=(self.n_items, 1, self.emb_item))
            
            score2 = tf.reduce_sum(self.user_emb * self.one_hop_user_aggregated_emb, axis=2)  # 第二层的注意力得分
            att2 = tf.reshape(tf.nn.softmax(score2), shape=(self.n_items, self.num_neighbors_dependence[0]+1, 1)) # 第二层的注意力权重
            self.embedding = tf.reduce_sum(self.one_hop_user_aggregated_emb * att2, axis=1)
            self.W2 = tf.get_variable('W2', shape=[self.emb_item, self.emb_item],initializer=tf.glorot_uniform_initializer())
            self.embedding = tf.nn.dropout(tf.nn.relu(tf.matmul(self.embedding, self.W2)), 1-dependence_dropout)  # 第二层物品的表示
#             self.one_hop_dependence_emb = tf.reduce_mean(self.dependence_emb, axis=2)
#             self.two_hop_dependence_emb = tf.reduce_mean(self.one_hop_dependence_emb, axis=1)
            fci_layer = Dense(self.emb_item*2, self.emb_item, act=lambda x:x, dropout=self.dropout if self.training else 0.)
            self.dense_layers.append(fci_layer)
            self.embedding = embedding = fci_layer(tf.concat([self.dependence_emb[:,0,0,:], self.embedding], -1))
            # self.embedding = embedding = fci_layer(tf.concat([self.dependence_emb[:,0,0,:], (1/2)*self.user_emb[:,0,:], (1/3)*self.embedding], -1))
            print('dependence')
            # self.embedding = embedding = self.item_emb_matrix
        else:
            self.embedding = embedding = tf.get_variable('item_embedding', [self.n_items, self.emb_item],\
                                        initializer=tf.glorot_uniform_initializer())  
        features_0 = self.decode() # features of zero layer nodes. 
        #outputs with shape [max_time, batch_size, dim2]
        if self.without_social:
            concat_self = features_0
            print('without_social')
        else:
            if self.global_only:
                features_1_2 = self.global_features()
                print('global_only')
            elif self.local_only:
                features_1_2 = self.local_features()
                print('local_only')
            else:
                features_1_2 = self.global_and_local_features()
            outputs = self.step_by_step(features_0, features_1_2, self.dims, self.num_samples_social, self.support_sizes,
                                    concat=self.concat)
            concat_self = tf.concat([features_0, outputs], axis=-1)
        

        # exchange first two dimensions.
        self.transposed_outputs = tf.transpose(concat_self, [1,0,2])

        self.loss = self._loss()
        self.sum_recall_10 = self._recall(10)
        self.sum_ndcg_10 = self._ndcg_k(10)
        self.sum_recall_20 = self._recall(20)
        self.sum_ndcg_20 = self._ndcg_k(20)
        self.sum_recall_50 = self._recall(50)
        self.sum_ndcg_50 = self._ndcg_k(50)
        self.sum_ndcg = self._ndcg()
        grads_and_vars = self.optimizer.compute_gradients(self.loss)
        clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var)
                        for grad, var in grads_and_vars]
        self.opt_op = self.optimizer.apply_gradients(clipped_grads_and_vars, global_step=self.global_step)
    
    def _loss(self):
        reg_loss = 0.
        xe_loss = 0.
        if self.without_social:
            fc_layer = Dense(self.hidden_size, self.emb_item, act=lambda x:x, dropout=self.dropout if self.training else 0.)
            self.dense_layers.append(fc_layer)
            self.logits = logits = tf.matmul(fc_layer(tf.reshape(self.transposed_outputs, [-1, self.hidden_size])), self.embedding, transpose_b=True)
            for dense_layer in self.dense_layers:
                for var in dense_layer.vars.values():
                    reg_loss += self.weight_decay * tf.nn.l2_loss(var)
        else:
            fc_layer = Dense(self.dim2 + self.hidden_size, self.emb_item, act=lambda x:x, dropout=self.dropout if self.training else 0.)
            self.dense_layers.append(fc_layer)
            self.logits = logits = tf.matmul(fc_layer(tf.reshape(self.transposed_outputs, [-1, self.dim2+self.hidden_size])), self.embedding, transpose_b=True)
            for dense_layer in self.dense_layers:
                for var in dense_layer.vars.values():
                    reg_loss += self.weight_decay * tf.nn.l2_loss(var)
            for aggregator in self.aggregators:
                for var in aggregator.vars.values():
                    reg_loss += self.weight_decay * tf.nn.l2_loss(var)
        if self.dependence_network:
            reg_loss += self.weight_decay * tf.nn.l2_loss(self.W1)
            reg_loss += self.weight_decay * tf.nn.l2_loss(self.W2)
        reshaped_logits = tf.reshape(logits, [self.batch_size, self.max_length, self.n_items])
        xe_loss += tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y,
                                                            logits=reshaped_logits,
                                                            name='softmax_loss')
        xe_loss *= self.mask
        return tf.reduce_sum(xe_loss) / self.point_count + reg_loss

    def _ndcg(self):
        predictions = tf.transpose(self.logits)
        targets = tf.reshape(self.input_y, [-1])
        pred_values = tf.expand_dims(tf.diag_part(tf.nn.embedding_lookup(predictions, targets)), -1)
        tile_pred_values = tf.tile(pred_values, [1, self.n_items-1])
        ranks = tf.reduce_sum(tf.cast(self.logits[:,1:] > tile_pred_values, dtype=tf.float32), -1) + 1
        ndcg = 1. / (log2(1.0 + ranks))
        mask = tf.reshape(self.mask, [-1])
        ndcg *= mask
        return tf.reduce_sum(ndcg)
    
    def _ndcg_k(self, top_k):
        predictions = tf.transpose(self.logits)
        targets = tf.reshape(self.input_y, [-1])
        pred_values = tf.expand_dims(tf.diag_part(tf.nn.embedding_lookup(predictions, targets)), -1)
        tile_pred_values = tf.tile(pred_values, [1, top_k])
        top_k_values, _ = tf.nn.top_k(self.logits, top_k)
        ranks = tf.reduce_sum(tf.cast(top_k_values > tile_pred_values, dtype=tf.float32), -1) + 1
        ndcg = 1. / (log2(1.0 + ranks))
        zero = tf.zeros(shape=(self.batch_size*self.max_length,), dtype=tf.float32)
        ndcg_k = tf.where(ndcg > 1 / log2(top_k+0.5), ndcg, zero)
        mask = tf.reshape(self.mask, [-1])
        ndcg_k *= mask
        return tf.reduce_sum(ndcg_k)

    def _recall(self, top_k):
        predictions = self.logits
        targets = tf.reshape(self.input_y, [-1])
        recall_at_k = tf.nn.in_top_k(predictions, targets, k=top_k)
        recall_at_k = tf.cast(recall_at_k, dtype=tf.float32)
        mask = tf.reshape(self.mask, [-1])
        recall_at_k *= mask
        return tf.reduce_sum(recall_at_k)

def log2(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(2, dtype=numerator.dtype))
    return numerator / denominator
