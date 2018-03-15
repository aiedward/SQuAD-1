# Copyright 2018 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This file defines the top-level model"""

from __future__ import absolute_import
from __future__ import division

import time
import logging
import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import embedding_ops

from evaluate import exact_match_score, f1_score
from data_batcher import get_batch_generator
from pretty_print import print_example
from modules import RNNEncoder, SimpleSoftmaxLayer, BasicAttn, BidirecAttn, SelfAttn

logging.basicConfig(level=logging.INFO)


class QAModel(object):
    """Top-level Question Answering module"""

    def __init__(self, FLAGS, id2word, word2id, emb_matrix, mcids):
        """
        Initializes the QA model.

        Inputs:
          FLAGS: the flags passed in from main.py
          id2word: dictionary mapping word idx (int) to word (string)
          word2id: dictionary mapping word (string) to word idx (int)
          emb_matrix: numpy array shape (400002, embedding_size) containing pre-traing GloVe embeddings
          mcids_dict: dictionary mapping word2id indices of 1000 most common question words to 0-9999 embedding indices
        """
        print "Initializing the QAModel..."
        self.FLAGS = FLAGS
        self.id2word = id2word
        self.word2id = word2id
        self.mcids = mcids
        self.mcids_dict =  dict(zip(mcids,range(len(mcids))))

        # Add all parts of the graph
        with tf.variable_scope("QAModel", initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, uniform=True)):
            self.add_placeholders()
            self.get_char_features() ###
            self.add_embedding_layer(emb_matrix)
            self.add_aligned_question_embs() ###
            self.add_features() ###
            self.add_dummy_features() ###
            self.build_graph()
            self.add_loss()

        # Define trainable parameters, gradient, gradient norm, and clip by gradient norm
        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, params)
        self.gradient_norm = tf.global_norm(gradients)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, FLAGS.max_gradient_norm)
        self.param_norm = tf.global_norm(params)

        # Define optimizer and updates
        # (updates is what you need to fetch in session.run to do a gradient update)
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        opt = tf.train.AdadeltaOptimizer(learning_rate=FLAGS.learning_rate) # you can try other optimizers
        self.updates = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)

        # Define savers (for checkpointing) and summaries (for tensorboard)
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.keep)
        self.bestmodel_saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        self.summaries = tf.summary.merge_all()

    def add_placeholders(self):
        """
        Add placeholders to the graph. Placeholders are used to feed in inputs.
        """
        # Add placeholders for inputs.
        # These are all batch-first: the None corresponds to batch_size and
        # allows you to run the same model with variable batch_size
        self.context_ids = tf.placeholder(tf.int32, shape=[None, self.FLAGS.context_len])
        self.context_mask = tf.placeholder(tf.int32, shape=[None, self.FLAGS.context_len])
        self.qn_ids = tf.placeholder(tf.int32, shape=[None, self.FLAGS.question_len])
        self.qn_mask = tf.placeholder(tf.int32, shape=[None, self.FLAGS.question_len])
        self.ans_span = tf.placeholder(tf.int32, shape=[None, 2])

        # Add a placeholder to feed in the keep probability (for dropout).
        # This is necessary so that we can instruct the model to use dropout when training, but not when testing
        self.keep_prob = tf.placeholder_with_default(1.0, shape=())

        ################ ADD PLACEHOLDER FOR FEATURES & CHAR_IDS ###############
        self.feats  = tf.placeholder(tf.float32,  shape=[None, self.FLAGS.context_len, self.FLAGS.num_feats])
        
        self.char_ids  = tf.placeholder(tf.int32, shape=[None, self.FLAGS.context_len, self.FLAGS.word_len])
        self.char_mask = tf.placeholder(tf.int32, shape=[None, self.FLAGS.context_len, self.FLAGS.word_len])

        self.charQ_ids  = tf.placeholder(tf.int32, shape=[None, self.FLAGS.question_len, self.FLAGS.word_len])
        self.charQ_mask = tf.placeholder(tf.int32, shape=[None, self.FLAGS.question_len, self.FLAGS.word_len])

        self.commonQ_mask        = tf.placeholder(tf.bool,  shape=[None, self.FLAGS.question_len])
        self.commonQ_emb_indices = tf.placeholder(tf.int32, shape=[None, self.FLAGS.question_len])
        ########################################################################

    def get_char_features(self):
        with vs.variable_scope("get_char"):
            char_embedding_matrix = tf.get_variable("char_matrix", shape=(43,self.FLAGS.char_embedding_size), initializer=tf.contrib.layers.xavier_initializer()) # shape (43, char_embedding_size)
            
            char_emb = embedding_ops.embedding_lookup(char_embedding_matrix, self.char_ids) # shape (batch_size, context_len, word_len, char_embedding_size)
            char_emb = tf.reshape(char_emb, [-1, self.FLAGS.word_len, self.FLAGS.char_embedding_size]) # (batch_size * context_len, word_len, char_embedding_size)
            char_hidden = tf.layers.conv1d(char_emb,filters=self.FLAGS.num_filters,kernel_size=self.FLAGS.cnn_width, padding='same', reuse=tf.AUTO_REUSE, name="charConv", kernel_initializer=tf.contrib.layers.xavier_initializer()) # (batch_size * context_len, word_len, num_filters)
            char_hidden = tf.reshape(char_hidden, [-1, self.FLAGS.context_len, self.FLAGS.word_len, self.FLAGS.num_filters]) # (batch_size, context_len, word_len, num_filters)
            char_hidden = char_hidden + tf.cast(tf.expand_dims(self.char_mask, 3), 'float')*(-1e30) # subtract large number from padded characters
            self.char_hidden = tf.nn.dropout(tf.reduce_max(char_hidden,axis=2), self.keep_prob) # (batch_size, context_len, num_filters)
            
            charQ_emb = embedding_ops.embedding_lookup(char_embedding_matrix, self.charQ_ids) # shape (batch_size, question_len, word_len, char_embedding_size)
            charQ_emb = tf.reshape(charQ_emb, [-1, self.FLAGS.word_len, self.FLAGS.char_embedding_size]) # (batch_size * question_len, word_len, char_embedding_size)
            charQ_hidden = tf.layers.conv1d(charQ_emb,filters=self.FLAGS.num_filters,kernel_size=self.FLAGS.cnn_width, padding='same', reuse=tf.AUTO_REUSE, name="charConv") # (batch_size * question_len, word_len, num_filters)
            charQ_hidden = tf.reshape(charQ_hidden, [-1, self.FLAGS.question_len, self.FLAGS.word_len, self.FLAGS.num_filters]) # (batch_size, question_len, word_len, num_filters)
            charQ_hidden = charQ_hidden + tf.cast(tf.expand_dims(self.charQ_mask, 3), 'float')*(-1e30) # subtract large number from padded characters
            self.charQ_hidden = tf.nn.dropout(tf.reduce_max(charQ_hidden,axis=2), self.keep_prob) # (batch_size, question_len, num_filters)

            print('Calculated CNN features!')

    def add_embedding_layer(self, emb_matrix):
        """
        Adds word embedding layer to the graph.

        Inputs:
          emb_matrix: shape (400002, embedding_size).
            The GloVe vectors, plus vectors for PAD and UNK.
        """
        with vs.variable_scope("embeddings"):

            # Note: the embedding matrix is a tf.constant which means it's not a trainable parameter
            embedding_matrix = tf.constant(emb_matrix, dtype=tf.float32, name="emb_matrix") # shape (400002, embedding_size)

            # Get the word embeddings for the context and question,
            # using the placeholders self.context_ids and self.qn_ids
            self.context_embs = embedding_ops.embedding_lookup(embedding_matrix, self.context_ids) # shape (batch_size, context_len, embedding_size)
            qn_embs = embedding_ops.embedding_lookup(embedding_matrix, self.qn_ids) # shape (batch_size, question_len, embedding_size) -- NOT DECLARING AS SELF YET

            ############### use learnable embeddings for common Q words #########################
            # qn_ids: (batch_size, question_len)
            commonQ_emb_matrix = tf.get_variable("commonQ_matrix", initializer=tf.gather(embedding_matrix,self.mcids)) # shape (1000, embedding_size)
            commonQ_embs = embedding_ops.embedding_lookup(commonQ_emb_matrix, self.commonQ_emb_indices) # shape (batch_size, question_len, embedding_size)
            
            commonQ_mask = tf.reshape(self.commonQ_mask, [-1,])                     # (batch_size * question_len, )
            qn_embs      = tf.reshape(qn_embs,      [-1,self.FLAGS.embedding_size]) # (batch_size * question_len, embedding_size)
            commonQ_embs = tf.reshape(commonQ_embs, [-1,self.FLAGS.embedding_size]) # (batch_size * question_len, embedding_size)

            output = tf.where(commonQ_mask,commonQ_embs,qn_embs)                               # (batch_size * question_len, embedding_size)
            output = tf.reshape(output,[-1,self.FLAGS.question_len,self.FLAGS.embedding_size]) # (batch_size, question_len, embedding_size)
            print('Using variable embeddings for 1000 most common!')

            self.qn_embs = output
            #####################################################################################

    def add_aligned_question_embs(self):
        """
        Adds aligned question embeddings to context embeddings, and another dummy row to question embeddings. See DrQA fro details.
        """        
        with vs.variable_scope("add_alignedQ"):
            attn_layer   = BasicAttn(self.keep_prob, self.FLAGS.embedding_size, self.FLAGS.embedding_size)
            attn_dist, _ = attn_layer.build_graph(self.qn_embs, self.qn_mask, self.context_embs, self.FLAGS.hidden_size) # havent added features to *embs yet
            
            # attn_dist    : (batch_size, context_len, question_len)
            # self.qn_embs : (batch_size, context_len, embedding_size)
            a = tf.expand_dims(attn_dist,3) * tf.expand_dims(self.qn_embs,1) # (b, N, M, d) = (b,N,M,1)*(b,1,M,d)
            a = tf.reduce_sum(a, axis=2) # (b,N,d)

            # concatenate aligned question embedding to cotext embeddings
            self.context_embs = tf.concat((self.context_embs, a), axis=2) # shape (batch_size, context_len, 2*embedding_size)
            print('Added aligned question_embs!')

    def add_features(self):
        """
        Adds word features e.g. (POS,NER,ExactMatch[x3]) and character CNN output to the graph. 
        Must be called after add_embedding_layer and get_char_features
        """     
        with vs.variable_scope("features"):
            self.context_embs = tf.concat((self.context_embs, self.feats), axis=2) # shape (batch_size, context_len, (2*embedding_size)+num_feats)
            print('Added manual features!')
            self.context_embs = tf.concat((self.context_embs, self.char_hidden), axis=2) # shape (batch_size, context_len, (2*embedding_size)+num_feats+num_filters)
            print('Added CNN features (to context)!')
            self.qn_embs = tf.concat((self.qn_embs, self.charQ_hidden), axis=2) # shape (batch_size, question_len, embedding_size+num_filters)
            print('Added CNN features (to question)!')

    def add_dummy_features(self):
        """
        Adds dummy word features (all zeros, same size as features) to the graph. Must be called after add_embedding_layer
        """     
        with vs.variable_scope("dummy_features"):
            actual_batch_size = tf.shape(self.feats)[0] # may not be batch_size if at end of file, for example
            additional_features = self.FLAGS.embedding_size + self.FLAGS.num_feats
            self.qn_embs = tf.concat((self.qn_embs, tf.zeros([actual_batch_size, self.FLAGS.question_len, additional_features],tf.float32)), axis=2) # shape (batch_size, question_len, 2*embedding_size+num_filters+num_feats)
            print('Added dummy features (You''re using a shared encoder, right?)')

    def build_graph(self):
        """Builds the main part of the graph for the model, starting from the input embeddings to the final distributions for the answer span.

        Defines:
          self.logits_start, self.logits_end: Both tensors shape (batch_size, context_len).
            These are the logits (i.e. values that are fed into the softmax function) for the start and end distribution.
            Important: these are -large in the pad locations. Necessary for when we feed into the cross entropy function.
          self.probdist_start, self.probdist_end: Both shape (batch_size, context_len). Each row sums to 1.
            These are the result of taking (masked) softmax of logits_start and logits_end.
        """

        # Use a RNN to get hidden states for the context and the question
        # Note: here the RNNEncoder is shared (i.e. the weights are the same)
        # between the context and the question.
        # encoderC = RNNEncoder(self.FLAGS.hidden_size, self.keep_prob, self.FLAGS.num_rnn_layers_embed, scope="RNNEncoderC")
        # encoderQ = RNNEncoder(self.FLAGS.hidden_size, self.keep_prob, self.FLAGS.num_rnn_layers_embed, scope="RNNEncoderQ")
        encoder = RNNEncoder(self.FLAGS.hidden_size, self.keep_prob, self.FLAGS.num_rnn_layers_embed, scope="RNNEncoder")

        context_hiddens  = encoder.build_graph(self.context_embs, self.context_mask) # (batch_size, context_len, hidden_size*2)
        question_hiddens = encoder.build_graph(self.qn_embs, self.qn_mask) # (batch_size, question_len, hidden_size*2)

        # ################### BASIC ATTENTION ###################
        # # Use context hidden states to attend to question hidden states
        # attn_layer = BasicAttn(self.keep_prob, self.FLAGS.hidden_size*2, self.FLAGS.hidden_size*2)
        # _, attn_output = attn_layer.build_graph(question_hiddens, self.qn_mask, context_hiddens) # attn_output is shape (batch_size, context_len, hidden_size*2)

        # # Concat attn_output to context_hiddens to get blended_reps
        # blended_reps = tf.concat([context_hiddens, attn_output], axis=2) # (batch_size, context_len, hidden_size*4)

        # # Apply fully connected layer to each blended representation
        # # Note, blended_reps_final corresponds to b' in the handout
        # # Note, tf.contrib.layers.fully_connected applies a ReLU non-linarity here by default
        # blended_reps_final = tf.contrib.layers.fully_connected(blended_reps, num_outputs=self.FLAGS.hidden_size) # blended_reps_final is shape (batch_size, context_len, hidden_size)
        # #######################################################

        #######################################################
        # Use context AND question hidden states to calculate bidirectional attention
        bidaf_layer  = BidirecAttn(self.keep_prob, self.FLAGS.hidden_size)
        bidaf_output = bidaf_layer.build_graph(c=context_hiddens, c_mask=self.context_mask, q=question_hiddens, q_mask=self.qn_mask) # attn_output is shape (batch_size, context_len, hidden_size*8)
        blended_reps = bidaf_output

        # ## self-attention
        # selfattn_layer  = SelfAttn(self.keep_prob, self.FLAGS.hidden_size)
        # selfattn_output = selfattn_layer.build_graph(c=context_hiddens, c_mask=self.context_mask) # shape (batch_size, context_len, hidden_size*2)
        # blended_reps = tf.concat([bidaf_output,selfattn_output], axis=2)

        # Add modeling layer after BiDAF
        encoderMod   = RNNEncoder(self.FLAGS.hidden_size, self.keep_prob, self.FLAGS.num_rnn_layers_model, scope="RNNEncoderMod")
        blended_reps = encoderMod.build_graph(blended_reps, self.context_mask) # (batch_size, context_len, hidden_size*2)

        # get final representation for START
        blended_reps_start = tf.contrib.layers.fully_connected(blended_reps, num_outputs=self.FLAGS.hidden_size) # blended_reps_start is shape (batch_size, context_len, hidden_size)

        # get final representation for END        
        encoderEnd   = RNNEncoder(self.FLAGS.hidden_size, self.keep_prob, num_rnn_layers=1, scope="RNNEncoderEnd")
        blended_reps_end = encoderEnd.build_graph(blended_reps, self.context_mask)

        #######################################################

        # Use softmax layer to compute probability distribution for start location
        # Note this produces self.logits_start and self.probdist_start, both of which have shape (batch_size, context_len)
        with vs.variable_scope("StartDist"):
            softmax_layer_start = SimpleSoftmaxLayer()
            self.logits_start, self.probdist_start = softmax_layer_start.build_graph(blended_reps_start, self.context_mask)

        # Use softmax layer to compute probability distribution for end location
        # Note this produces self.logits_end and self.probdist_end, both of which have shape (batch_size, context_len)
        with vs.variable_scope("EndDist"):
            softmax_layer_end = SimpleSoftmaxLayer()
            self.logits_end, self.probdist_end = softmax_layer_end.build_graph(blended_reps_end, self.context_mask)


    def add_loss(self):
        """
        Add loss computation to the graph.

        Uses:
          self.logits_start: shape (batch_size, context_len)
            IMPORTANT: Assumes that self.logits_start is masked (i.e. has -large in masked locations).
            That's because the tf.nn.sparse_softmax_cross_entropy_with_logits
            function applies softmax and then computes cross-entropy loss.
            So you need to apply masking to the logits (by subtracting large
            number in the padding location) BEFORE you pass to the
            sparse_softmax_cross_entropy_with_logits function.

          self.ans_span: shape (batch_size, 2)
            Contains the gold start and end locations

        Defines:
          self.loss_start, self.loss_end, self.loss: all scalar tensors
        """
        with vs.variable_scope("loss"):

            # Calculate loss for prediction of start position
            loss_start = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_start, labels=self.ans_span[:, 0]) # loss_start has shape (batch_size)
            self.loss_start = tf.reduce_mean(loss_start) # scalar. avg across batch
            tf.summary.scalar('loss_start', self.loss_start) # log to tensorboard

            # Calculate loss for prediction of end position
            loss_end = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_end, labels=self.ans_span[:, 1])
            self.loss_end = tf.reduce_mean(loss_end)
            tf.summary.scalar('loss_end', self.loss_end)

            # calculate L2 loss
            # reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            # loss_reg = tf.contrib.layers.apply_regularization(regularizer, reg_variables)

            # Add the two losses
            self.loss = self.loss_start + self.loss_end #+ loss_reg
            tf.summary.scalar('loss', self.loss)


    def run_train_iter(self, session, batch, summary_writer):
        """
        This performs a single training iteration (forward pass, loss computation, backprop, parameter update)

        Inputs:
          session: TensorFlow session
          batch: a Batch object
          summary_writer: for Tensorboard

        Returns:
          loss: The loss (averaged across the batch) for this batch.
          global_step: The current number of training iterations we've done
          param_norm: Global norm of the parameters
          gradient_norm: Global norm of the gradients
        """
        # Match up our input data with the placeholders
        input_feed = {}
        input_feed[self.context_ids] = batch.context_ids
        input_feed[self.context_mask] = batch.context_mask
        input_feed[self.qn_ids] = batch.qn_ids
        input_feed[self.qn_mask] = batch.qn_mask
        input_feed[self.ans_span] = batch.ans_span
        input_feed[self.keep_prob] = 1.0 - self.FLAGS.dropout # apply dropout
        input_feed[self.feats] = batch.feats
        input_feed[self.char_ids] = batch.char_ids
        input_feed[self.char_mask] = batch.char_mask
        input_feed[self.commonQ_mask] = batch.commonQ_mask
        input_feed[self.commonQ_emb_indices] = batch.commonQ_emb_indices
        input_feed[self.charQ_ids] = batch.charQ_ids
        input_feed[self.charQ_mask] = batch.charQ_mask

        # output_feed contains the things we want to fetch.
        output_feed = [self.updates, self.summaries, self.loss, self.global_step, self.param_norm, self.gradient_norm]

        # Run the model
        [_, summaries, loss, global_step, param_norm, gradient_norm] = session.run(output_feed, input_feed)

        # All summaries in the graph are added to Tensorboard
        summary_writer.add_summary(summaries, global_step)

        return loss, global_step, param_norm, gradient_norm


    def get_loss(self, session, batch):
        """
        Run forward-pass only; get loss.

        Inputs:
          session: TensorFlow session
          batch: a Batch object

        Returns:
          loss: The loss (averaged across the batch) for this batch
        """

        input_feed = {}
        input_feed[self.context_ids] = batch.context_ids
        input_feed[self.context_mask] = batch.context_mask
        input_feed[self.qn_ids] = batch.qn_ids
        input_feed[self.qn_mask] = batch.qn_mask
        input_feed[self.ans_span] = batch.ans_span
        input_feed[self.feats] = batch.feats
        input_feed[self.char_ids] = batch.char_ids
        input_feed[self.char_mask] = batch.char_mask
        input_feed[self.commonQ_mask] = batch.commonQ_mask
        input_feed[self.commonQ_emb_indices] = batch.commonQ_emb_indices
        input_feed[self.charQ_ids] = batch.charQ_ids
        input_feed[self.charQ_mask] = batch.charQ_mask
        # note you don't supply keep_prob here, so it will default to 1 i.e. no dropout

        output_feed = [self.loss]

        [loss] = session.run(output_feed, input_feed)

        return loss


    def get_prob_dists(self, session, batch):
        """
        Run forward-pass only; get probability distributions for start and end positions.

        Inputs:
          session: TensorFlow session
          batch: Batch object

        Returns:
          probdist_start and probdist_end: both shape (batch_size, context_len)
        """
        input_feed = {}
        input_feed[self.context_ids] = batch.context_ids
        input_feed[self.context_mask] = batch.context_mask
        input_feed[self.qn_ids] = batch.qn_ids
        input_feed[self.qn_mask] = batch.qn_mask
        input_feed[self.feats] = batch.feats
        input_feed[self.char_ids] = batch.char_ids
        input_feed[self.char_mask] = batch.char_mask
        input_feed[self.commonQ_mask] = batch.commonQ_mask
        input_feed[self.commonQ_emb_indices] = batch.commonQ_emb_indices
        input_feed[self.charQ_ids] = batch.charQ_ids
        input_feed[self.charQ_mask] = batch.charQ_mask
        # note you don't supply keep_prob here, so it will default to 1 i.e. no dropout

        output_feed = [self.probdist_start, self.probdist_end]
        [probdist_start, probdist_end] = session.run(output_feed, input_feed)
        return probdist_start, probdist_end


    def get_start_end_pos(self, session, batch, K):
        """
        Run forward-pass only; get the most likely answer span.

        Inputs:
          session: TensorFlow session
          batch: Batch object

        Returns:
          start_pos, end_pos: both numpy arrays shape (batch_size).
            The most likely start and end positions for each example in the batch.
        """

        # ################# ORIGINAL (NAIVE) FORMULATION #################
        # # Get start_dist and end_dist, both shape (batch_size, context_len)
        # start_dist, end_dist = self.get_prob_dists(session, batch)
        # # Take argmax to get start_pos and end_post, both shape (batch_size)
        # start_pos = np.argmax(start_dist, axis=1)
        # end_pos = np.argmax(end_dist, axis=1)

        ################# SMART FORMULATION #################
        # From DrQA, choose the start and end location pair (i, j) with i <= j <= i + K 
        # that maximizes p_start(i)*p_end(j), K=15 by default

        # Get start_dist and end_dist, both shape (batch_size, context_len)
        start_dist, end_dist = self.get_prob_dists(session, batch)

        start_dist = np.expand_dims(start_dist,2) # (b,N,1)
        end_dist   = np.expand_dims(end_dist,1)   # (b,1,N)
        probs = np.matmul(start_dist, end_dist)   # (b,N,N)
        probs = np.triu(probs)      # mask out bottom diagonal to enforce i<=j
        probs = np.tril(probs, K)   # mask out upper  diagonal (above K) to enforce j<=(i+K)

        # get argmax in row/column (i,j) format for each batch
        indices = np.transpose(np.asarray([np.unravel_index(np.argmax(x, axis=None), x.shape) for x in probs]))
        start_pos = indices[0]
        end_pos   = indices[1]

        return start_pos, end_pos

    def get_dev_loss(self, session, dev_context_path, dev_qn_path, dev_ans_path):
        """
        Get loss for entire dev set.

        Inputs:
          session: TensorFlow session
          dev_qn_path, dev_context_path, dev_ans_path: paths to the dev.{context/question/answer} data files

        Outputs:
          dev_loss: float. Average loss across the dev set.
        """
        logging.info("Calculating dev loss...")
        tic = time.time()
        loss_per_batch, batch_lengths = [], []

        # Iterate over dev set batches
        # Note: here we set discard_long=True, meaning we discard any examples
        # which are longer than our context_len or question_len.
        # We need to do this because if, for example, the true answer is cut
        # off the context, then the loss function is undefined.
        for batch in get_batch_generator(self.word2id, dev_context_path, dev_qn_path, dev_ans_path, self.FLAGS.batch_size, context_len=self.FLAGS.context_len, question_len=self.FLAGS.question_len, discard_long=True, num_feats=self.FLAGS.num_feats, word_len=self.FLAGS.word_len, mcids_dict=self.mcids_dict):

            # Get loss for this batch
            loss = self.get_loss(session, batch)
            curr_batch_size = batch.batch_size
            loss_per_batch.append(loss * curr_batch_size)
            batch_lengths.append(curr_batch_size)

        # Calculate average loss
        total_num_examples = sum(batch_lengths)
        toc = time.time()
        print "Computed dev loss over %i examples in %.2f seconds" % (total_num_examples, toc-tic)

        # Overall loss is total loss divided by total number of examples
        dev_loss = sum(loss_per_batch) / float(total_num_examples)

        return dev_loss


    def check_f1_em(self, session, context_path, qn_path, ans_path, dataset, num_samples=100, print_to_screen=False):
        """
        Sample from the provided (train/dev) set.
        For each sample, calculate F1 and EM score.
        Return average F1 and EM score for all samples.
        Optionally pretty-print examples.

        Note: This function is not quite the same as the F1/EM numbers you get from "official_eval" mode.
        This function uses the pre-processed version of the e.g. dev set for speed,
        whereas "official_eval" mode uses the original JSON. Therefore:
          1. official_eval takes your max F1/EM score w.r.t. the three reference answers,
            whereas this function compares to just the first answer (which is what's saved in the preprocessed data)
          2. Our preprocessed version of the dev set is missing some examples
            due to tokenization issues (see squad_preprocess.py).
            "official_eval" includes all examples.

        Inputs:
          session: TensorFlow session
          qn_path, context_path, ans_path: paths to {dev/train}.{question/context/answer} data files.
          dataset: string. Either "train" or "dev". Just for logging purposes.
          num_samples: int. How many samples to use. If num_samples=0 then do whole dataset.
          print_to_screen: if True, pretty-prints each example to screen

        Returns:
          F1 and EM: Scalars. The average across the sampled examples.
        """
        logging.info("Calculating F1/EM for %s examples in %s set..." % (str(num_samples) if num_samples != 0 else "all", dataset))

        f1_total = 0.
        em_total = 0.
        example_num = 0

        tic = time.time()

        # Note here we select discard_long=False because we want to sample from the entire dataset
        # That means we're truncating, rather than discarding, examples with too-long context or questions
        for batch in get_batch_generator(self.word2id, context_path, qn_path, ans_path, self.FLAGS.batch_size, context_len=self.FLAGS.context_len, question_len=self.FLAGS.question_len, discard_long=False, num_feats=self.FLAGS.num_feats, word_len=self.FLAGS.word_len, mcids_dict=self.mcids_dict):

            pred_start_pos, pred_end_pos = self.get_start_end_pos(session, batch, self.FLAGS.max_span)

            # Convert the start and end positions to lists length batch_size
            pred_start_pos = pred_start_pos.tolist() # list length batch_size
            pred_end_pos = pred_end_pos.tolist() # list length batch_size

            for ex_idx, (pred_ans_start, pred_ans_end, true_ans_tokens) in enumerate(zip(pred_start_pos, pred_end_pos, batch.ans_tokens)):
                example_num += 1

                # Get the predicted answer
                # Important: batch.context_tokens contains the original words (no UNKs)
                # You need to use the original no-UNK version when measuring F1/EM
                pred_ans_tokens = batch.context_tokens[ex_idx][pred_ans_start : pred_ans_end + 1]
                pred_answer = " ".join(pred_ans_tokens)

                # Get true answer (no UNKs)
                true_answer = " ".join(true_ans_tokens)

                # Calc F1/EM
                f1 = f1_score(pred_answer, true_answer)
                em = exact_match_score(pred_answer, true_answer)
                f1_total += f1
                em_total += em

                # Optionally pretty-print
                if print_to_screen:
                    print_example(self.word2id, batch.context_tokens[ex_idx], batch.qn_tokens[ex_idx], batch.ans_span[ex_idx, 0], batch.ans_span[ex_idx, 1], pred_ans_start, pred_ans_end, true_answer, pred_answer, f1, em)

                if num_samples != 0 and example_num >= num_samples:
                    break

            if num_samples != 0 and example_num >= num_samples:
                break

        f1_total /= example_num
        em_total /= example_num

        toc = time.time()
        logging.info("Calculating F1/EM for %i examples in %s set took %.2f seconds" % (example_num, dataset, toc-tic))

        return f1_total, em_total


    def train(self, session, train_context_path, train_qn_path, train_ans_path, dev_qn_path, dev_context_path, dev_ans_path):
        """
        Main training loop.

        Inputs:
          session: TensorFlow session
          {train/dev}_{qn/context/ans}_path: paths to {train/dev}.{context/question/answer} data files
        """

        # Print number of model parameters
        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retrieval took %f secs)" % (num_params, toc - tic))

        # We will keep track of exponentially-smoothed loss
        exp_loss = None

        # Checkpoint management.
        # We keep one latest checkpoint, and one best checkpoint (early stopping)
        checkpoint_path = os.path.join(self.FLAGS.train_dir, "qa.ckpt")
        bestmodel_dir = os.path.join(self.FLAGS.train_dir, "best_checkpoint")
        bestmodel_ckpt_path = os.path.join(bestmodel_dir, "qa_best.ckpt")
        best_dev_f1 = None
        best_dev_em = None
        best_dev_av = None

        # for TensorBoard
        summary_writer = tf.summary.FileWriter(self.FLAGS.train_dir, session.graph)

        epoch = 0

        logging.info("Beginning training loop...")
        while self.FLAGS.num_epochs == 0 or epoch < self.FLAGS.num_epochs:
            epoch += 1
            epoch_tic = time.time()

            # Loop over batches
            for batch in get_batch_generator(self.word2id, train_context_path, train_qn_path, train_ans_path, self.FLAGS.batch_size, context_len=self.FLAGS.context_len, question_len=self.FLAGS.question_len, discard_long=True, num_feats=self.FLAGS.num_feats, word_len=self.FLAGS.word_len, mcids_dict=self.mcids_dict):

                # Run training iteration
                iter_tic = time.time()
                loss, global_step, param_norm, grad_norm = self.run_train_iter(session, batch, summary_writer)
                iter_toc = time.time()
                iter_time = iter_toc - iter_tic

                # Update exponentially-smoothed loss
                if not exp_loss: # first iter
                    exp_loss = loss
                else:
                    exp_loss = 0.99 * exp_loss + 0.01 * loss

                # Sometimes print info to screen
                if global_step % self.FLAGS.print_every == 0:
                    logging.info(
                        'epoch %d, iter %d, loss %.5f, smoothed loss %.5f, grad norm %.5f, param norm %.5f, batch time %.3f' %
                        (epoch, global_step, loss, exp_loss, grad_norm, param_norm, iter_time))

                # Sometimes save model
                if global_step % self.FLAGS.save_every == 0:
                    logging.info("Saving to %s..." % checkpoint_path)
                    self.saver.save(session, checkpoint_path, global_step=global_step)

                # Sometimes evaluate model on dev loss, train F1/EM and dev F1/EM
                if global_step % self.FLAGS.eval_every == 0:

                    # Get loss for entire dev set and log to tensorboard
                    dev_loss = self.get_dev_loss(session, dev_context_path, dev_qn_path, dev_ans_path)
                    logging.info("Epoch %d, Iter %d, dev loss: %f" % (epoch, global_step, dev_loss))
                    write_summary(dev_loss, "dev/loss", summary_writer, global_step)


                    # Get F1/EM on train set and log to tensorboard
                    train_f1, train_em = self.check_f1_em(session, train_context_path, train_qn_path, train_ans_path, "train", num_samples=1000)
                    logging.info("Epoch %d, Iter %d, Train F1 score: %f, Train EM score: %f" % (epoch, global_step, train_f1, train_em))
                    write_summary(train_f1, "train/F1", summary_writer, global_step)
                    write_summary(train_em, "train/EM", summary_writer, global_step)


                    # Get F1/EM on dev set and log to tensorboard
                    dev_f1, dev_em = self.check_f1_em(session, dev_context_path, dev_qn_path, dev_ans_path, "dev", num_samples=0)
                    logging.info("Epoch %d, Iter %d, Dev F1 score: %f, Dev EM score: %f" % (epoch, global_step, dev_f1, dev_em))
                    write_summary(dev_f1, "dev/F1", summary_writer, global_step)
                    write_summary(dev_em, "dev/EM", summary_writer, global_step)
                    dev_av = (dev_f1+dev_em)/2

                    # Early stopping based on dev EM. You could switch this to use F1 instead.
                    if best_dev_av is None or dev_av > best_dev_av:
                        best_dev_av = dev_av
                        logging.info("Saving to %s..." % bestmodel_ckpt_path)
                        self.bestmodel_saver.save(session, bestmodel_ckpt_path, global_step=global_step)


            epoch_toc = time.time()
            logging.info("End of epoch %i. Time for epoch: %f" % (epoch, epoch_toc-epoch_tic))

        sys.stdout.flush()



def write_summary(value, tag, summary_writer, global_step):
    """Write a single summary value to tensorboard"""
    summary = tf.Summary()
    summary.value.add(tag=tag, simple_value=value)
    summary_writer.add_summary(summary, global_step)
