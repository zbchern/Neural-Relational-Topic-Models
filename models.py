import numpy as np
import tensorflow as tf
import logging
import utils
import os
from sklearn.metrics import roc_auc_score

np.random.seed(0)
tf.set_random_seed(0)


class NRTM(object):
    def __init__(self, config, doc_contents, train_pairs, train_labels, test_links, test_pairs_hit, vocab, links):
        self.batch_size = config.batch_size
        self.init_lr = config.init_lr_train
        self.lr_decay = config.lr_decay
        self.hidden_dim = config.hidden_dim
        self.layers_list = eval(config.pretrain_layers_list)
        self.cf_layers_list = eval(config.cf_layers_list)
        self.activations = eval(config.activations)
        self.loss = config.loss

        self.trained_print_step = config.trained_print_step
        self.test_step = config.test_step
        self.print_words_step = config.print_words_step
        self.top_k = config.top_k
        self.max_epoch = config.train_max_epoch

        # data set
        self.dataset = config.dataset
        self.doc_contents = doc_contents
        self.train_pairs = train_pairs
        self.train_labels = train_labels
        self.test_links = test_links
        self.test_pairs_hit = test_pairs_hit
        self.vocab = vocab
        self.links = links

        self.lambda_w = config.lambda_w

        self.input_num = doc_contents.shape[0]
        self.input_dim = doc_contents.shape[1]
        self.weights = []
        self.cf_weights = []

        self.batch_data = tf.placeholder(name='batch_data', shape=[None, self.input_dim], dtype=tf.float32)
        self.batch_labels = tf.placeholder(name='batch_labels', shape=[None], dtype=tf.float32)
        self.links_batch = tf.placeholder(name='links_batch', shape=[None], dtype=tf.float32)
        self.get_emb = tf.placeholder(name='get_emb', shape=[2, None], dtype=tf.int32)
        self.keep_prob = tf.placeholder(name='keep_prob', dtype=tf.float32)
        self.z_test = tf.placeholder(name='topic_embedding', shape=[self.input_num, self.hidden_dim], dtype=tf.float32)
        self.learning_rate = tf.placeholder(name='learning_rate', dtype=tf.float32)
        self._build()

        # Launch the session
        conf = tf.ConfigProto()
        conf.gpu_options.per_process_gpu_memory_fraction = 0.5
        conf.gpu_options.allow_growth = True
        self.sess = tf.Session(config=conf)
        self.sess.run(tf.global_variables_initializer())

    def train(self):
        logging.info('Training...')

        best_hr = [0.]
        best_ndcg = [0.]

        # file to store the generated topic words
        if os.path.exists('res/topics_' + self.dataset + '.txt'):
            os.remove('res/topics_' + self.dataset + '.txt')
            logging.info('Successfully remove existing topic file!')

        batch_num = int(len(self.train_pairs) / self.batch_size) + 1
        for epoch in range(self.max_epoch):

            if (epoch + 1) % 20 == 0:
                self.init_lr = self.init_lr * self.lr_decay

            logging.info('Training at epoch ' + str(epoch + 1) + ' ...')

            loss_total, gen_loss_total, latent_loss_total, reg_loss_total, cf_loss_total = 0., 0., 0., 0., 0.
            for batch in range(batch_num):
                batch_data, batch_labels = utils.load_batch(self.train_pairs, self.train_labels, batch, self.batch_size)
                batch_data = np.transpose(batch_data)
                docu1 = batch_data[0]
                docu2 = batch_data[1]

                docus = np.concatenate((docu1, docu2), axis=0)
                get_emb = [list(range(len(docu1))), list(range(len(docu1), len(docu1) + len(docu2)))]

                feed_dict = {self.batch_data: self.doc_contents[docus],
                             self.batch_labels: np.array(batch_labels),
                             self.learning_rate: self.init_lr,
                             self.keep_prob: 1.,
                             self.get_emb: get_emb}

                _, loss_tmp, gen_loss_tmp, latent_loss_tmp, reg_loss_tmp, cf_loss_tmp = self.sess.run(
                    (self.train_op, self.loss, self.gen_loss, self.latent_loss, self.reg_loss, self.cf_loss), feed_dict=feed_dict)

                loss_total += loss_tmp
                gen_loss_total += gen_loss_tmp
                latent_loss_total += latent_loss_tmp
                reg_loss_total += reg_loss_tmp
                cf_loss_total += cf_loss_tmp

            if (epoch + 1) % self.trained_print_step == 0:
                logging.info(
                    'Epoch {0}: avg batch loss = {1}, gen loss = {2}, latent loss = {3}, reg loss = {4}, cf loss = {5}\n'.format(
                        epoch + 1, loss_total / batch_num, gen_loss_total / batch_num, latent_loss_total / batch_num,
                        reg_loss_total / batch_num, 1000. * cf_loss_total / batch_num))

            if (epoch + 1) % self.test_step == 0:
                logging.info('Testing at epoch ' + str(epoch + 1) + ' ...')

                z_test = self.sess.run(self.z, feed_dict={self.batch_data: self.doc_contents, self.keep_prob: 1.0})
                feed_dict = {self.z_test: z_test,
                             self.keep_prob: 1.0}

                # ave_rank, ave_auc = self._auc_test(feed_dict)
                # logging.info('ave rank = ' + str(ave_rank) + ', ave auc = ' + str(ave_auc) + '\n')

                hits, ndcgs = self._hit_test(feed_dict)
                logging.info('HR = ' + str(hits))
                logging.info('NDCGS = ' + str(ndcgs) + '\n')
                if best_hr[-1] < hits[-1]:
                    best_hr = hits
                if best_ndcg[-1] < ndcgs[-1]:
                    best_ndcg = ndcgs

            if (epoch + 1) % self.print_words_step == 0:
                utils.print_top_words(self.sess.run(self.weights_words), self.vocab, self.dataset)

        logging.info('BEST HR = ' + str(best_hr))
        logging.info('BEST NDCGS = ' + str(best_ndcg) + '\n\n\n')

    def _build(self):
        self.x_recon, reg_loss1 = self._inference_generation()

        # reconstruction loss
        if self.loss == 'rmse':
            self.gen_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.batch_data - self.x_recon), axis=1))
        else:
            self.x_recon = tf.squeeze(self.x_recon)
            self.gen_loss = tf.reduce_mean(
                tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.batch_data, logits=self.x_recon),
                              axis=1))

        self.latent_loss = 0.5 * tf.reduce_mean(
            tf.reduce_sum(tf.square(self.z_mean) + tf.exp(self.z_log_sigma_sq) - self.z_log_sigma_sq - 1, axis=1))

        self.cf_loss, reg_loss2, self.logits = self._collaborative_filtering(self.z, False)
        self.reg_loss = reg_loss1 + reg_loss2
        self.loss = self.gen_loss + self.latent_loss + 1000. * self.cf_loss + self.lambda_w * self.reg_loss
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        self.cf_loss_test, _, self.logits_test = self._collaborative_filtering(self.z_test, True)

    def _hit_test(self, feed_dict):

        hits = [[] for _ in range(self.top_k)]
        ndcgs = [[] for _ in range(self.top_k)]

        for test_link in self.test_pairs_hit:
            input_links = []
            input_labels = [0] * 100
            input_labels[0] = 1
            for i in range(1, len(test_link)):
                input_links.append([test_link[0], test_link[i]])

            input_links = np.array(input_links).transpose()
            input_labels = np.array(input_labels)

            feed_dict[self.batch_labels] = input_labels
            feed_dict[self.get_emb] = input_links
            probs = self.sess.run(self.logits_test, feed_dict=feed_dict)

            probs_arg = probs.argsort()

            for k in range(1, self.top_k + 1):
                top_k = probs_arg[-k:][::-1]
                hits_tmp = utils.getHits(top_k, 0)
                ndcgs_tmp = utils.getNDCG(top_k, 0)

                hits[k - 1].append(hits_tmp)
                ndcgs[k - 1].append(ndcgs_tmp)

        return np.mean(hits, axis=1), np.mean(ndcgs, axis=1)

    def _auc_test(self, feed_dict):
        sum_rank = 0
        num_links = 0
        auc = []

        for (id, p) in enumerate(self.test_links):
            if len(p) == 0:
                continue
            get_emb, batch_labels = utils.load_batch_test(id, p, self.input_num)
            get_emb = np.asarray(get_emb).transpose()
            feed_dict[self.batch_labels] = np.array(batch_labels)
            feed_dict[self.get_emb] = get_emb
            logits = self.sess.run(self.logits_test, feed_dict=feed_dict)

            ordered = logits.argsort()[::-1].tolist()  # descending order
            ranks = [ordered.index(x) for x in p]
            sum_rank += sum(ranks)
            num_links += len(ranks)
            # compute auc
            y_score = np.delete(logits, id)
            y_true = np.delete(batch_labels, id)
            auc.append(roc_auc_score(y_true, y_score))
        ave_rank = 1.0 * sum_rank / num_links
        ave_auc = np.mean(np.array(auc))
        return ave_rank, ave_auc

    def _collaborative_filtering(self, z, reuse):
        """Forward passing to get cf logits"""
        net = tf.concat((tf.gather(z, self.get_emb[0]), tf.gather(z, self.get_emb[1])), axis=1)
        with tf.variable_scope("cf"):
            for i in range(len(self.cf_layers_list)-1):
                net = tf.layers.dense(net, self.cf_layers_list[i], activation=tf.nn.sigmoid, name='fc_' + str(i),
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(), reuse=reuse)
            net = tf.layers.dense(net, self.cf_layers_list[-1], name='fc_last',
                                  kernel_initializer=tf.contrib.layers.xavier_initializer(), reuse=reuse)

            net = tf.squeeze(net, axis=1)
            cf_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.batch_labels, logits=net))

            reg_loss = 0.
            self.cf_weights = [v for v in tf.trainable_variables() if 'cf' in v.name]
            for weight in self.cf_weights:
                reg_loss + tf.nn.l2_loss(weight)

        return cf_loss, reg_loss, tf.nn.sigmoid(net)

    def _inference_generation(self):

        with tf.variable_scope('inference'):

            rec = {'W1': tf.get_variable(name='W1', shape=[self.input_dim, self.layers_list[0]],
                                         initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32),
                   'b1': tf.get_variable(name='b1', shape=[self.layers_list[0]],
                                         initializer=tf.constant_initializer(0.), dtype=tf.float32)}
            for layer_id in range(1, len(self.layers_list)):
                key_w = 'W' + str(layer_id + 1)
                key_b = 'b' + str(layer_id + 1)
                rec[key_w] = tf.get_variable(name=key_w,
                                             shape=[self.layers_list[layer_id - 1], self.layers_list[layer_id]],
                                             initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
                rec[key_b] = tf.get_variable(name=key_b, shape=[self.layers_list[layer_id]],
                                             initializer=tf.constant_initializer(0.), dtype=tf.float32)

            rec['W_z_mean'] = tf.get_variable(name='W_z_mean', shape=[self.layers_list[-1], self.hidden_dim],
                                              initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            rec['b_z_mean'] = tf.get_variable(name='b_z_mean', shape=[self.hidden_dim],
                                              initializer=tf.constant_initializer(0.), dtype=tf.float32)
            rec['W_z_log_sigma'] = tf.get_variable(name='W_z_log_sigma', shape=[self.layers_list[-1], self.hidden_dim],
                                                   initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            rec['b_z_log_sigma'] = tf.get_variable(name='b_z_log_sigma', shape=[self.hidden_dim],
                                                   initializer=tf.constant_initializer(0.), dtype=tf.float32)

        for key in rec:
            self.weights.append(rec[key])

        net = utils.activate(tf.matmul(self.batch_data, rec['W1']) + rec['b1'], activator=self.activations[0])

        for layer_id in range(1, len(self.layers_list)):
            key_w = 'W' + str(layer_id + 1)
            key_b = 'b' + str(layer_id + 1)
            net = utils.activate(tf.matmul(net, rec[key_w]) + rec[key_b], activator=self.activations[layer_id])

        net = tf.nn.dropout(net, self.keep_prob)

        self.z_mean = tf.matmul(net, rec['W_z_mean']) + rec['b_z_mean']
        self.z_log_sigma_sq = tf.matmul(net, rec['W_z_log_sigma']) + rec['b_z_log_sigma']

        eps = tf.random_normal((tf.shape(self.batch_data)[0], self.hidden_dim), 0, 1, seed=0, dtype=tf.float32)
        self.z = self.z_mean + tf.sqrt(tf.maximum(tf.exp(self.z_log_sigma_sq), 1e-10)) * eps

        with tf.variable_scope('generation'):
            gen = {}
            key_w = 'Wz'
            key_b = 'bz'
            gen[key_w] = tf.get_variable(name=key_w, shape=[self.hidden_dim, self.layers_list[-1]],
                                         initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            gen[key_b] = tf.get_variable(name=key_b, shape=[self.layers_list[-1]],
                                         initializer=tf.constant_initializer(0.), dtype=tf.float32)
            for layer_id in reversed(range(1, len(self.layers_list))):
                key_w = 'W' + str(layer_id + 1)
                key_b = 'b' + str(layer_id + 1)
                gen[key_w] = tf.transpose(rec[key_w])
                gen[key_b] = rec['b' + str(layer_id)]

            gen['W1'] = tf.transpose(rec['W1'])
            gen['b1'] = tf.get_variable('b1', shape=[self.input_dim], initializer=tf.constant_initializer(0.),
                                        dtype=tf.float32)

        self.weights += [gen['Wz'], gen['bz'], gen['b1']]
        self.saver = tf.train.Saver(self.weights)

        reg_loss = 0.
        for weight in self.weights:
            reg_loss += tf.nn.l2_loss(weight)

        net = utils.activate(tf.matmul(self.z, gen['Wz']) + gen['bz'], activator=self.activations[-1])

        self.weights_words = gen['Wz']
        for layer_id in reversed(range(1, len(self.layers_list))):
            key_w = 'W' + str(layer_id + 1)
            key_b = 'b' + str(layer_id + 1)
            net = utils.activate(tf.matmul(net, gen[key_w]) + gen[key_b], activator=self.activations[layer_id])
            self.weights_words = tf.matmul(self.weights_words, gen[key_w])

        x_recon = tf.matmul(net, gen['W1']) + gen['b1']

        # weights used to generate topic words
        self.weights_words = tf.matmul(self.weights_words, gen['W1'])

        return x_recon, reg_loss

    def save_model(self, weight_path):
        self.saver.save(self.sess, weight_path)
        logging.info("Weights saved at " + weight_path)

    def load_model(self, weight_path):
        self.saver.restore(self.sess, weight_path)
        logging.info("Weights restored from " + weight_path)
