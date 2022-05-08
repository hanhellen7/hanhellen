# -*- coding: utf-8 -*-
import os
import time
import sys
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from data import pad_sequences, batch_yield
from utils import get_logger, conlleval


class BilstmCrf(object):
    """Build bilstm model."""

    def __init__(self, args, tag2label, vocab, paths, config):
        """Init paramsI."""
        self.batch_size = args.batch_size
        self.epoch_num = args.epoch
        self.hidden_dim = args.hidden_dim
        self.update_embedding = args.update_embedding
        self.dropout_keep_prob = args.dropout
        self.optimizer = args.optimizer
        self.lr = args.lr
        self.clip_grad = args.clip
        self.CRF = args.CRF
        self.shuffle = args.shuffle
        self.tag2label = tag2label
        self.num_tags = len(tag2label)
        self.vocab = vocab
        self.model_path = paths['model_path']
        self.summary_path = paths['summary_path']
        self.logger = get_logger(paths['log_path'])
        self.result_path = paths['result_path']
        self.config = config

    def build_graph(self):
        """Build graph."""
        self.add_placeholders()
        self.lookup_layer_op()
        self.biLSTM_layer_op()
        self.softmax_pred_op()
        self.loss_op()
        self.trainstep_op()
        self.init_op()

    def add_placeholders(self):
        """Add placeholders."""
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name="word_ids")
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")
        self.embeddings = tf.placeholder(tf.float32, shape=[None, None], name="embeddings")

        self.dropout_pl = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        self.lr_pl = tf.placeholder(dtype=tf.float32, shape=[], name="lr")

    def lookup_layer_op(self):
        """Embedding lookup."""
        with tf.variable_scope("words"):
            _word_embeddings = tf.Variable(tf.constant(0.01, shape=[3905, 300]), dtype=tf.float32, trainable=self.update_embedding, name="_word_embeddings")
            _word_embeddings.assign(self.embeddings)
            word_embeddings = tf.nn.embedding_lookup(params=_word_embeddings, ids=self.word_ids, name="word_embeddings")
        self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout_pl)

    def biLSTM_layer_op(self):
        """Build biLSTM graph."""
        with tf.variable_scope("bi-lstm"):
            cell_fw = LSTMCell(self.hidden_dim)
            cell_bw = LSTMCell(self.hidden_dim)
            (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw, cell_bw=cell_bw, inputs=self.word_embeddings, sequence_length=self.sequence_lengths, dtype=tf.float32)
            output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
            output = tf.nn.dropout(output, self.dropout_pl)

        with tf.variable_scope("proj"):
            w = tf.get_variable(name="W", shape=[2 * self.hidden_dim, self.num_tags], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)

            b = tf.get_variable(name="b",
                                shape=[self.num_tags],
                                initializer=tf.zeros_initializer(),
                                dtype=tf.float32)

            s = tf.shape(output)
            output = tf.reshape(output, [-1, 2 * self.hidden_dim])
            pred = tf.matmul(output, w) + b
            self.logits = tf.reshape(pred, shape=[-1, s[1], self.num_tags])

    def loss_op(self):
        """Choice and caculate loss."""
        if self.CRF:
            log_likelihood, self.transition_params = crf_log_likelihood(inputs=self.logits, tag_indices=self.labels, sequence_lengths=self.sequence_lengths)
            self.loss = -tf.reduce_mean(log_likelihood)
        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels)
            mask = tf.sequence_mask(self.sequence_lengths)
            losses = tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(losses)
        tf.summary.scalar("loss", self.loss)

    def softmax_pred_op(self):
        """Softmax operation."""
        if not self.CRF:
            self.labels_softmax_ = tf.argmax(self.logits, axis=-1)
            self.labels_softmax_ = tf.cast(self.labels_softmax_, tf.int32)

    def trainstep_op(self):
        """Train step."""
        with tf.variable_scope("train_step"):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            if self.optimizer == 'Adam':
                optim = tf.train.AdamOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Adadelta':
                optim = tf.train.AdadeltaOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Adagrad':
                optim = tf.train.AdagradOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'RMSProp':
                optim = tf.train.RMSPropOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Momentum':
                optim = tf.train.MomentumOptimizer(learning_rate=self.lr_pl, momentum=0.9)
            elif self.optimizer == 'SGD':
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)
            else:
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)

            grads_and_vars = optim.compute_gradients(self.loss)
            grads_and_vars_clip = [[tf.clip_by_value(g, -self.clip_grad, self.clip_grad), v] for g, v in grads_and_vars]
            self.train_op = optim.apply_gradients(grads_and_vars_clip, global_step=self.global_step)

    def init_op(self):
        """Init global variable operation."""
        self.init_op = tf.global_variables_initializer()

    def add_summary(self, sess):
        """
        Write summary to file.

        :param sess: tf Session to summary
        """
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.summary_path, sess.graph)

    def train(self, train, dev, embeddings):
        """
        Train model.

        :param train: train data
        :param dev: evalution data
        :param embedding: embedding
        """
        saver = tf.train.Saver(tf.global_variables())
        self.embedding_data = embeddings

        with tf.Session(config=self.config) as sess:
            sess.run(self.init_op)
            self.add_summary(sess)

            for epoch in range(self.epoch_num):
                self.run_one_epoch(sess, train, dev, self.tag2label, epoch, saver)

    def test(self, test, test_out):
        """
        Test model.

        :param test: test data
        :param test_out: test output
        """
        saver = tf.train.Saver()
        with tf.Session(config=self.config) as sess:
            self.logger.info('=========== testing ===========')
            saver.restore(sess, self.model_path)
            label_list, seq_len_list = self.dev_one_epoch(sess, test)
            self.evaluate(label_list, test, test_out=test_out)

    def predict_one(self, sess, sent):
        """
        Predict sentence.

        :param sess:tf Session
        :param sent: a list contain a lot of data to predict
        :return: tag list
        """
        label_list = []
        for seqs, labels in batch_yield(sent, self.batch_size, self.vocab, self.tag2label, shuffle=False):
            label_list_, _ = self.predict_one_batch(sess, seqs)
            label_list.extend(label_list_)
        label2tag = {label:tag for tag,label in self.tag2label.items()}
        tag = [label2tag[label] for label in label_list[0]]
        return tag

    def run_one_epoch(self, sess, train, dev, tag2label, epoch, saver):
        """
        Run one epoch during train。

        :param sess: tf Session
        :param train: train dataset
        :param dev: develop dataset
        :param tag2label: tag to label dict
        :param epoch: epoch number
        :param saver: tf saver
        """
        num_batches = (len(train) + self.batch_size - 1) // self.batch_size
        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        batches = batch_yield(train, self.batch_size, self.vocab, self.tag2label, shuffle=self.shuffle)
        for step, (seqs, labels) in enumerate(batches):
            sys.stdout.write(' processing: {} batch / {} batches.'.format(step + 1, num_batches) + '\r')
            step_num = epoch * num_batches + step + 1
            feed_dict, _ = self.get_feed_dict(seqs, self.embedding_data, labels, self.lr, self.dropout_keep_prob)
            _, loss_train, summary, step_num_ = sess.run([self.train_op, self.loss, self.merged, self.global_step], feed_dict=feed_dict)
            if step + 1 == 1 or (step + 1) % 300 == 0 or step + 1 == num_batches:
                self.logger.info('{} epoch {}, step {}, loss: {:.4}, global_step: {}'.format(start_time, epoch + 1, step + 1, loss_train, step_num))

            self.file_writer.add_summary(summary, step_num)

            if step + 1 == num_batches:
                saver.save(sess, self.model_path, global_step=step_num)

        self.logger.info('===========validation / test===========')
        label_list_dev, seq_len_list_dev = self.dev_one_epoch(sess, dev)
        self.evaluate(label_list_dev, dev, epoch)

    def get_feed_dict(self, seqs, embeddings=None, labels=None, lr=None, dropout=1.0):
        """
        Get data feed to model.

        :param seqs: input list
        :param labels: input label
        :param lr: learn rate
        :param dropout: droup out ratio
        :return: feed_dict
        """
        word_ids, seq_len_list = pad_sequences(seqs, pad_mark=0)

        feed_dict = {self.word_ids: word_ids,
                     self.sequence_lengths: seq_len_list}
        if labels is not None:
            labels_, _ = pad_sequences(labels, pad_mark=0)
            feed_dict[self.labels] = labels_
        if lr is not None:
            feed_dict[self.lr_pl] = lr
        if dropout is not None:
            feed_dict[self.dropout_pl] = dropout
        if embeddings is not None:
            feed_dict[self.embeddings] = embeddings

        return feed_dict, seq_len_list

    def dev_one_epoch(self, sess, dev):
        """
        Develop one epoch.

        :param sess: tf Session
        :param dev: develop dataset
        :return: label_list, seq_len_list
        """
        label_list, seq_len_list = [], []
        for seqs, labels in batch_yield(dev, self.batch_size, self.vocab, self.tag2label, shuffle=False):
            label_list_, seq_len_list_ = self.predict_one_batch(sess, seqs)
            label_list.extend(label_list_)
            seq_len_list.extend(seq_len_list_)
        return label_list, seq_len_list

    def predict_one_batch(self, sess, seqs):
        """
        Predict one batch data.

        :param sess: tf Session
        :param seqs: sequence to predict
        :return: label_list, seq_len_list
        """
        feed_dict, seq_len_list = self.get_feed_dict(seqs)
        if self.CRF:
            logits, transition_params = sess.run([self.logits, self.transition_params],
                                                 feed_dict=feed_dict)
            label_list = []
            for logit, seq_len in zip(logits, seq_len_list):
                viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_params)
                label_list.append(viterbi_seq)
            return label_list, seq_len_list
        else:
            label_list = sess.run(self.labels_softmax_, feed_dict=feed_dict)
            return label_list, seq_len_list

    def evaluate(self, label_list, data, epoch=None, test_out = None):
        """
        Evaluate model and ouput result such as recall, percision, accuary.

        :param label_list: predict label
        :param data: sentence and true label
        :param epoch: epoch number
        """
        label2tag = {}
        for tag, label in self.tag2label.items():
            label2tag[label] = tag
        model_predict = []
        for label_, (sent, tag) in zip(label_list, data):
            tag_ = [label2tag[label__] for label__ in label_]
            sent_res = []
            if  len(label_) != len(sent):
                print(sent)
                print(len(label_))
                print(tag)
            for i in range(len(sent)):
                sent_res.append([sent[i], tag[i], tag_[i]])
            model_predict.append(sent_res)
        if epoch != None:
            print("not test")
            epoch_num = "str"
            label_path = os.path.join(self.result_path, 'label_' + epoch_num)
            metric_path = os.path.join(self.result_path, 'result_metric_' + epoch_num)
        else:
            print("this is test")
            label_path = test_out
            metric_path = test_out+"metric"
        for _ in conlleval(model_predict, label_path, metric_path):
            self.logger.info(_)
