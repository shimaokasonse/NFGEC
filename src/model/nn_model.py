# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import sys
sys.path.append('../../')
sys.path.append('../')
from create_prior_knowledge import create_prior

def weight_variable(name, shape, pad=True):
    initial = np.random.uniform(-0.01, 0.01, size=shape)
    if pad == True:
        initial[0] = np.zeros(shape[1])
    initial = tf.constant_initializer(initial)
    return tf.get_variable(name=name, shape=shape, initializer=initial)

def attentive_sum(inputs,input_dim, hidden_dim):
    with tf.variable_scope("attention"):
        seq_length = len(inputs)
        W =  weight_variable('att_W', (input_dim,hidden_dim))
        U =  weight_variable('att_U', (hidden_dim,1))
        tf.get_variable_scope().reuse_variables()
        temp1 = [tf.nn.tanh(tf.matmul(inputs[i],W)) for i in range(seq_length)]
        temp2 = [tf.matmul(temp1[i],U) for i in range(seq_length)]
        pre_activations = tf.concat(1,temp2)
        attentions = tf.split(1, seq_length, tf.nn.softmax(pre_activations))
        weighted_inputs = [tf.mul(inputs[i],attentions[i]) for i in range(seq_length)]
        output = tf.add_n(weighted_inputs)
    return output, attentions


class Model:
    def __init__(self,type = "figer", encoder = "averaging", hier = False, feature = False):

        # Argument Checking
        assert(encoder in ["averaging", "lstm", "attentive"])
        assert(type in ["figer", "gillick"])
        self.type = type
        self.encoder = encoder
        self.hier = hier
        self.feature = feature

        # Hyperparameters
        self.context_length = 10
        self.emb_dim = 300
        self.target_dim = 113 if type == "figer" else 89
        self.feature_size = 600000 if type == "figer" else 100000
        self.learning_rate = 0.001
        self.lstm_dim = 100
        self.att_dim  = 100 # dim of attention module
        self.feature_dim = 50 # dim of feature representation
        self.feature_input_dim = 70
        if encoder == "averaging":
            self.rep_dim = self.emb_dim * 3
        else:
            self.rep_dim = self.lstm_dim * 2 + self.emb_dim
        if feature:
            self.rep_dim += self.feature_dim

        # Place Holders
        self.keep_prob = tf.placeholder(tf.float32)
        self.mention_representation = tf.placeholder(tf.float32,[None,self.emb_dim])
        self.context = [tf.placeholder(tf.float32, [None, self.emb_dim]) for _ in range(self.context_length*2+1)]
        self.target = tf.placeholder(tf.float32,[None,self.target_dim])

        ### dropout and splitting context into left and right
        self.mention_representation_dropout = tf.nn.dropout(self.mention_representation,self.keep_prob)
        self.left_context = self.context[:self.context_length]
        self.right_context = self.context[self.context_length+1:]



        # Averaging Encoder
        if encoder == "averaging":
            self.left_context_representation  = tf.add_n(self.left_context)
            self.right_context_representation = tf.add_n(self.right_context)
            self.context_representation       = tf.concat(1,[self.left_context_representation,self.right_context_representation])

        # LSTM Encoder
        if encoder == "lstm":
            self.left_lstm  = tf.nn.rnn_cell.LSTMCell(self.lstm_dim,state_is_tuple=True)
            self.right_lstm = tf.nn.rnn_cell.LSTMCell(self.lstm_dim,state_is_tuple=True)
            with tf.variable_scope("rnn_left") as scope:
                self.left_rnn,_  = tf.nn.rnn(self.left_lstm,self.left_context,dtype=tf.float32)
            with tf.variable_scope("rnn_right") as scope:
                self.right_rnn,_ = tf.nn.rnn(self.right_lstm,list(reversed(self.right_context)),dtype=tf.float32)
            self.context_representation = tf.concat(1,[self.left_rnn[-1],self.right_rnn[-1]])

        # Attentive Encoder
        if encoder == "attentive":
            self.left_lstm_F  = tf.nn.rnn_cell.LSTMCell(self.lstm_dim,state_is_tuple=True)
            self.right_lstm_F = tf.nn.rnn_cell.LSTMCell(self.lstm_dim,state_is_tuple=True)
            self.left_lstm_B  = tf.nn.rnn_cell.LSTMCell(self.lstm_dim,state_is_tuple=True)
            self.right_lstm_B = tf.nn.rnn_cell.LSTMCell(self.lstm_dim,state_is_tuple=True)
            with tf.variable_scope("rnn_left") as scope:
                self.left_birnn,_,_  = tf.nn.bidirectional_rnn(self.left_lstm_F,self.left_lstm_B,self.left_context,dtype=tf.float32)
            with tf.variable_scope("rnn_right") as scope:
                self.right_birnn,_,_ = tf.nn.bidirectional_rnn(self.right_lstm_F,self.right_lstm_B,list(reversed(self.right_context)),dtype=tf.float32)
            self.context_representation, self.attentions = attentive_sum(self.left_birnn + self.right_birnn, input_dim = self.lstm_dim * 2, hidden_dim = self.att_dim)


        # Logistic Regression
        if feature:
            self.features = tf.placeholder(tf.int32,[None,self.feature_input_dim])
            self.feature_embeddings = weight_variable('feat_embds', (self.feature_size, self.feature_dim), True)
            self.feature_representation = tf.nn.dropout(tf.reduce_sum(tf.nn.embedding_lookup(self.feature_embeddings,self.features),1),self.keep_prob)
            self.representation = tf.concat(1, [self.mention_representation_dropout, self.context_representation, self.feature_representation])
        else:
            self.representation = tf.concat(1, [self.mention_representation_dropout, self.context_representation])

        if hier:
            _d = "Wiki" if type == "figer" else "OntoNotes"
            S = create_prior("./resource/"+_d+"/label2id_"+type+".txt")
            assert(S.shape == (self.target_dim, self.target_dim))
            self.S = tf.constant(S,dtype=tf.float32)
            self.V = weight_variable('hier_V', (self.target_dim,self.rep_dim))
            self.W = tf.transpose(tf.matmul(self.S,self.V))
            self.logit = tf.matmul(self.representation, self.W)
        else:
            self.W = weight_variable('hier_W', (self.rep_dim,self.target_dim))
            self.logit = tf.matmul(self.representation, self.W)

        self.distribution = tf.nn.sigmoid(self.logit)

        # Loss Function
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.logit, self.target))
        # Optimizer
        self.optim = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        # Session
        self.init = tf.initialize_all_variables()
        self.session = tf.Session()
        self.session.run(self.init)

    def train(self, context_data, mention_representation_data, target_data, feature_data=None):
        feed = {self.mention_representation: mention_representation_data,
                self.target: target_data,
                self.keep_prob: [0.5]}
        if self.feature == True and feature_data is not None:
            feed[self.features] = feature_data
        for i in range(self.context_length*2+1):
            feed[self.context[i]] = context_data[:,i,:]
        self.session.run(self.optim,feed_dict=feed)

    def error(self, context_data, mention_representation_data, target_data, feature_data=None):
        feed = {self.mention_representation: mention_representation_data,
                self.target: target_data,
                self.keep_prob: [1.0]}
        if self.feature == True and feature_data is not None:
            feed[self.features] = feature_data
        for i in range(self.context_length*2+1):
            feed[self.context[i]] = context_data[:,i,:]
        return self.session.run(self.loss,feed_dict=feed)

    def predict(self, context_data, mention_representation_data, feature_data=None):
        feed = {self.mention_representation: mention_representation_data,
                self.keep_prob: [1.0]}
        if self.feature == True and feature_data is not None:
            feed[self.features] = feature_data
        for i in range(self.context_length*2+1):
            feed[self.context[i]] = context_data[:,i,:]
        return self.session.run(self.distribution,feed_dict=feed)

    def save(self):
        pass

    def load(self):
        pass

    def save_label_embeddings(self):
        pass



