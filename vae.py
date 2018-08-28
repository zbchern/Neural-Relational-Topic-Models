import numpy as np
import tensorflow as tf
import logging
import utils


def xavier_init(fan_in, fan_out, dtype=tf.float32, constant=1):
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=dtype)


class VariationalAutoEncoder(object):
    def __init__(self, config, doc_contents, vocab):
        self.batch_size = config.batch_size
        self.init_lr = config.init_lr_pretrain
        self.lr_decay = config.lr_decay
        self.max_epoch = config.pretrain_max_epoch
        self.print_step = config.pretrain_print_step
        self.noise = config.noise
        self.layers_list = eval(config.pretrain_layers_list)
        self.activations = eval(config.activations)
        self.hidden_dim = config.hidden_dim
        self.loss = config.loss
        self.pretrain_dir = config.pretrain_dir

        self.doc_contents = doc_contents
        self.vocab = vocab
        self.dataset = config.dataset

        self.print_words_step = config.print_words_step

        self.encode_weights = []
        self.encode_biases = []
        self.decode_weights = []
        self.decode_biases = []
        self.weights = []

        np.random.seed(0)
        tf.set_random_seed(0)

    def pretrain(self):
        logging.info('Pretraining...')
        idx = np.random.rand(self.doc_contents.shape[0]) < 0.8
        train_doc = self.doc_contents[idx]
        train_doc_tmp = train_doc
        test_doc = self.doc_contents[~idx]

        for layer_id in range(len(self.layers_list)):
            train_doc_tmp = self._train_vallina(layer_id, train_doc_tmp)

        self._train_latent(train_doc_tmp)

        self._train_all(train_doc, x_valid=True)

    def _train_all(self, train_doc, x_valid):

        logging.info('Combined pre-training...')

        tf.reset_default_graph()
        input_dim = train_doc.shape[1]  # the same as self.layers_list[-1]

        with tf.variable_scope('inference'):
            rec = {'W1': tf.get_variable(name='W1', initializer=tf.constant(self.encode_weights[0]), dtype=tf.float32),
                   'b1': tf.get_variable(name='b1', initializer=tf.constant(self.encode_biases[0]), dtype=tf.float32)}
            for layer_id in range(1, len(self.layers_list)):
                key_w = 'W' + str(layer_id + 1)
                key_b = 'b' + str(layer_id + 1)
                rec[key_w] = tf.get_variable(name=key_w, initializer=tf.constant(self.encode_weights[layer_id]),
                                             dtype=tf.float32)
                rec[key_b] = tf.get_variable(name=key_b, initializer=tf.constant(self.encode_biases[layer_id]),
                                             dtype=tf.float32)

            rec['W_z_mean'] = tf.get_variable(name='W_z_mean', initializer=tf.constant(self.encode_weights[-2]),
                                              dtype=tf.float32)
            rec['b_z_mean'] = tf.get_variable(name='b_z_mean', initializer=tf.constant(self.encode_biases[-2]),
                                              dtype=tf.float32)
            rec['W_z_log_sigma'] = tf.get_variable(name='W_z_log_sigma',
                                                   initializer=tf.constant(self.encode_weights[-1]), dtype=tf.float32)
            rec['b_z_log_sigma'] = tf.get_variable(name='b_z_log_sigma',
                                                   initializer=tf.constant(self.encode_biases[-1]), dtype=tf.float32)

        with tf.variable_scope('generation'):
            gen = {}
            key_w = 'Wz'
            key_b = 'bz'
            gen[key_w] = tf.get_variable(name=key_w, initializer=tf.constant(self.decode_weights[-1]), dtype=tf.float32)
            gen[key_b] = tf.get_variable(name=key_b, initializer=tf.constant(self.decode_biases[-1]), dtype=tf.float32)
            for layer_id in reversed(range(1, len(self.layers_list))):
                key_w = 'W' + str(layer_id + 1)
                key_b = 'b' + str(layer_id + 1)
                gen[key_w] = tf.transpose(rec[key_w])
                gen[key_b] = rec['b' + str(layer_id)]

            gen['W1'] = tf.transpose(rec['W1'])
            gen['b1'] = tf.get_variable('b1', shape=[input_dim], initializer=tf.constant_initializer(0.), dtype=tf.float32)

        for key in rec:
            self.weights.append(rec[key])

        self.weights += [gen['Wz'], gen['bz'], gen['b1']]
        self.saver = tf.train.Saver(self.weights)

        doc_x_ = tf.placeholder(name='doc_x_', shape=[None, input_dim], dtype=tf.float32)
        net = utils.activate(tf.matmul(doc_x_, rec['W1']) + rec['b1'], activator=self.activations[0])
        learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')

        for layer_id in range(1, len(self.layers_list)):
            key_w = 'W' + str(layer_id + 1)
            key_b = 'b' + str(layer_id + 1)
            net = utils.activate(tf.matmul(net, rec[key_w]) + rec[key_b], activator=self.activations[layer_id])

        z_mean = tf.matmul(net, rec['W_z_mean']) + rec['b_z_mean']
        z_log_sigma_sq = tf.matmul(net, rec['W_z_log_sigma']) + rec['b_z_log_sigma']

        eps = tf.random_normal((self.batch_size, self.hidden_dim), 0, 1, seed=0, dtype=tf.float32)
        z = z_mean + tf.sqrt(tf.maximum(tf.exp(z_log_sigma_sq), 1e-10)) * eps

        net = utils.activate(tf.matmul(z, gen['Wz']) + gen['bz'], activator=self.activations[-1])

        self.weights_words = gen['Wz']
        for layer_id in reversed(range(1, len(self.layers_list))):
            key_w = 'W' + str(layer_id + 1)
            key_b = 'b' + str(layer_id + 1)
            net = utils.activate(tf.matmul(net, gen[key_w]) + gen[key_b], activator=self.activations[layer_id])
            self.weights_words = tf.matmul(self.weights_words, gen[key_w])

        x_recon = tf.squeeze(tf.matmul(net, gen['W1']) + gen['b1'])
        self.weights_words = tf.matmul(self.weights_words, gen['W1'])

        gen_loss = tf.reduce_mean(
            tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=doc_x_, logits=x_recon), axis=1))

        latent_loss = 0.5 * tf.reduce_mean(
            tf.reduce_sum(tf.square(z_mean) + tf.exp(z_log_sigma_sq) - z_log_sigma_sq - 1, axis=1))

        loss = gen_loss + latent_loss

        train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        batch_num = int(len(train_doc) / self.batch_size) + 1
        for epoch in range(self.max_epoch * 2):
            loss_total, gen_loss_total, latent_loss_total = 0., 0., 0.
            for batch in range(batch_num):
                batch_x_, idx = utils.get_batch(train_doc, self.batch_size)
                feed_dict = {doc_x_: batch_x_,
                             learning_rate: self.init_lr}
                _, loss_tmp, gen_loss_tmp, latent_loss_tmp = sess.run((train_op, loss, gen_loss, latent_loss),
                                                                      feed_dict=feed_dict)
                loss_total += loss_tmp
                gen_loss_total += gen_loss_tmp
                latent_loss_total += latent_loss_tmp

            if (epoch + 1) % self.print_step == 0:
                if x_valid:
                    valid_loss = self._validate_test(train_doc, sess, gen_loss, doc_x_, learning_rate)
                    logging.info(
                        'Epoch {0}: avg batch loss = {1}, gen loss = {2}, latent loss = {3}, valid loss = {4}'.format(
                            epoch + 1, loss_total / batch_num, gen_loss_total / batch_num,
                            latent_loss_total / batch_num, valid_loss))
                else:
                    logging.info('Epoch {0}: avg batch loss = {1}, gen loss = {2}, latent loss = {3}'.format(epoch + 1,
                            loss_total / batch_num, gen_loss_total / batch_num, latent_loss_total / batch_num))

            # print out the topic words generated in stacked variational auto-encoder
            if (epoch + 1) % self.print_words_step == 0:
                utils.print_top_words(sess.run(self.weights_words), self.vocab, self.dataset)

        self.saver.save(sess, self.pretrain_dir)
        logging.info('Weights saved at ' + self.pretrain_dir)

    def _validate_test(self, train_doc, sess, gen_loss, doc_x_, learning_rate):
        batch_num = int((train_doc.shape[0] + 0.) / self.batch_size)
        n_samples = batch_num * self.batch_size

        valid_loss = 0.
        for batch in range(batch_num):
            ids = range(batch * self.batch_size, (batch + 1) * self.batch_size)
            batch_data = train_doc[ids]
            feed_dict = {doc_x_: batch_data,
                         learning_rate: self.init_lr}
            gen_loss_tmp = sess.run(gen_loss, feed_dict=feed_dict)
            valid_loss += gen_loss_tmp / n_samples * self.batch_size
        return valid_loss

    def _train_latent(self, train_doc):

        logging.info('Latent pre-training...')
        tf.reset_default_graph()
        batch_num = int(len(train_doc) / self.batch_size) + 1
        input_dim = train_doc.shape[1]  # the same as self.layers_list[-1]

        doc_x_ = tf.placeholder(dtype=tf.float32, shape=[None, input_dim], name='doc_x')
        learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')

        with tf.variable_scope('latent'):
            z_mean = tf.layers.dense(inputs=doc_x_, units=self.hidden_dim, name='z_mean')

            z_log_sigma_sq = tf.layers.dense(inputs=doc_x_, units=self.hidden_dim, name='z_log_sigma_sq')

            eps = tf.random_normal((self.batch_size, self.hidden_dim), 0, 1, dtype=tf.float32)
            z = z_mean + tf.sqrt(tf.maximum(tf.exp(z_log_sigma_sq), 1e-10)) * eps

            x_recon = tf.layers.dense(inputs=z, units=self.layers_list[-1], name='x_recon')

            gen_loss = tf.reduce_mean(
                tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=doc_x_, logits=x_recon), axis=1))

            latent_loss = 0.5 * tf.reduce_mean(
                tf.reduce_sum(tf.square(z_mean) + tf.exp(z_log_sigma_sq) - z_log_sigma_sq - 1, axis=1))

            loss = gen_loss + latent_loss

            train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        for epoch in range(self.max_epoch):

            # if (epoch + 1) == int(self.max_epoch * 0.75):
            #     self.init_lr = self.init_lr * self.lr_decay

            loss_total, gen_loss_total, latent_loss_total = 0., 0., 0.
            for batch in range(batch_num):
                batch_x_, idx = utils.get_batch(train_doc, self.batch_size)
                feed_dict = {doc_x_: batch_x_,
                             learning_rate: self.init_lr}
                _, loss_tmp, gen_loss_tmp, latent_loss_tmp = sess.run((train_op, loss, gen_loss, latent_loss),
                                                                      feed_dict=feed_dict)
                loss_total += loss_tmp
                gen_loss_total += gen_loss_tmp
                latent_loss_total += latent_loss_tmp

            if (epoch + 1) % self.print_step == 0:
                logging.info('Epoch {0}: avg batch loss = {1}, gen loss = {2}, latent loss = {3}'.format(epoch + 1,
                                                                                                         loss_total / batch_num,
                                                                                                         gen_loss_total / batch_num,
                                                                                                         latent_loss_total / batch_num))
        weights = tf.trainable_variables(scope='latent')
        self.encode_weights.append(sess.run(weights[0]))
        self.encode_biases.append(sess.run(weights[1]))
        self.encode_weights.append(sess.run(weights[2]))
        self.encode_biases.append(sess.run(weights[3]))
        self.decode_weights.append(sess.run(weights[4]))
        self.decode_biases.append(sess.run(weights[5]))

    def _train_vallina(self, layer_id, train_doc):
        tf.reset_default_graph()
        batch_num = int(len(train_doc) / self.batch_size) + 1
        learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')

        logging.info('Vallina pre-training...')
        logging.info('Training layer {0}'.format(layer_id + 1))

        input_dim = train_doc.shape[1]

        doc_x = tf.placeholder(dtype=tf.float32, shape=[None, input_dim], name='doc_x')
        doc_x_ = tf.placeholder(dtype=tf.float32, shape=[None, input_dim], name='doc_x_')

        encoder = {
            'weights': tf.Variable(initial_value=xavier_init(input_dim, self.layers_list[layer_id], dtype=tf.float32)),
            'biases': tf.Variable(tf.zeros(shape=[self.layers_list[layer_id]], dtype=tf.float32))}
        decoder = {'weights': tf.transpose(encoder['weights']),
                   'biases': tf.Variable(tf.zeros(shape=[input_dim]), dtype=tf.float32)}

        encoded = utils.activate(tf.matmul(doc_x, encoder['weights']) + encoder['biases'], self.activations[layer_id])
        decoded = tf.matmul(encoded, decoder['weights']) + decoder['biases']

        # reconstruction loss
        if self.loss == 'rmse':
            rec_loss = tf.reduce_mean(tf.reduce_sum(tf.square(doc_x_ - doc_x), axis=1))
        else:
            rec_loss = tf.reduce_mean(
                tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=doc_x_, logits=decoded), axis=1))

        train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(rec_loss)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        for epoch in range(self.max_epoch):

            # if (epoch + 1) == int(self.max_epoch * 0.75):
            #     self.init_lr = self.init_lr * self.lr_decay

            rec_loss_total = 0.
            for batch in range(batch_num):
                batch_x_, idx = utils.get_batch(train_doc, self.batch_size)
                batch_x = utils.add_noise(batch_x_, self.noise)
                feed_dict = {doc_x: batch_x,
                             doc_x_: batch_x_,
                             learning_rate: self.init_lr}
                _, rec_loss_tmp = sess.run((train_op, rec_loss), feed_dict=feed_dict)
                rec_loss_total += rec_loss_tmp

            # if (epoch + 1) % self.print_step == 0:
            #     logging.info('Epoch {0}: batch loss = {1}'.format(epoch + 1, rec_loss_tmp))

            if (epoch + 1) % self.print_step == 0:
                logging.info('Epoch {0}: avg batch loss = {1}'.format(epoch + 1, (rec_loss_total / batch_num)))

        self.encode_weights.append(sess.run(encoder['weights']))
        self.encode_biases.append(sess.run(encoder['biases']))
        self.decode_weights.append(sess.run(decoder['weights']))
        self.decode_biases.append(sess.run(decoder['biases']))

        return sess.run(encoded, feed_dict={doc_x: train_doc})
