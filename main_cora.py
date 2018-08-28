import tensorflow as tf
import utils
import os
import logging
import models
import vae

utils.set_best_gpu()

flags = tf.app.flags

# model training settings
flags.DEFINE_integer('batch_size', 128, 'training batch size')
flags.DEFINE_float('init_lr_pretrain', 0.01, 'initial learning rate for pre-train')
flags.DEFINE_float('init_lr_train', 0.001, 'initial learning rate for training')
flags.DEFINE_float('lr_decay', 0.1, 'learning rate decay rate')
flags.DEFINE_float('lambda_w', 1e-4, 'lambda for the regularizer of W')
flags.DEFINE_string('noise', 'None', "[None, 'gaussian', 'mask']")
flags.DEFINE_integer('hidden_dim', 50, 'dimension of embedding layer')
flags.DEFINE_integer('top_k', 10, 'top k words')
flags.DEFINE_string('pretrain_layers_list', '[200, 100]', 'pretrain layers of model')  # will also be used in training
flags.DEFINE_string('cf_layers_list', '[50, 25, 8, 1]', 'layers of model')
flags.DEFINE_string('activations', "['sigmoid', 'sigmoid']", 'activations for different layers')
flags.DEFINE_string('loss', 'cross-entropy', "cross-entropy, rmse")

flags.DEFINE_integer('train_max_epoch', 50, 'max epoch for training')
flags.DEFINE_integer('pretrain_max_epoch', 50, 'max epoch for pretrain')
flags.DEFINE_integer('pretrain_print_step', 1, 'print step for pretrain')
flags.DEFINE_integer('trained_print_step', 1, 'print step for training')
flags.DEFINE_integer('test_step', 2, 'step for testing')
flags.DEFINE_integer('print_words_step', 2, 'step for printing words')

flags.DEFINE_integer('negative_num', 5, 'negative samples for each positive citation')

# system settings
flags.DEFINE_string('data_dir', 'data/cora/', 'directory of data')
flags.DEFINE_string('dataset', 'cora', 'dataset')
flags.DEFINE_string('pretrain_dir', 'saver/pretrain_cora/', 'directory for storing pre-training files')
flags.DEFINE_string('trained_dir', 'saver/trained_cora/', 'directory for storing training files')
flags.DEFINE_integer('mode', 2, '1 for pretrain, 2 for training')

FLAGS = flags.FLAGS

if not os.path.exists(FLAGS.pretrain_dir):
    os.makedirs(FLAGS.pretrain_dir)
if not os.path.exists(FLAGS.trained_dir):
    os.makedirs(FLAGS.trained_dir)

if FLAGS.mode == 1:
    utils.init_logging(FLAGS.pretrain_dir + 'pretrain.log')
else:
    utils.init_logging(FLAGS.trained_dir + 'trained.log')

logging.info('Loading data...')
data = utils.load_data(FLAGS.data_dir, FLAGS.negative_num)
logging.info('Finish loading data')


def main(_):

    utils.print_settings(FLAGS)

    logging.info('#' * 60)
    logging.info('Current mode is {0}'.format(FLAGS.mode))
    logging.info('#' * 60 + '\n')

    if FLAGS.mode == 1:
        vae_net = vae.VariationalAutoEncoder(FLAGS, data['doc_contents'], data['vocab'])
        vae_net.pretrain()
    elif FLAGS.mode == 2:
        nrtm = models.NRTM(FLAGS, data['doc_contents'], data['train_links_neg'], data['train_labels_neg'],
                           data['test_links'], data['test_links_hit'], data['vocab'], data['links'])
        nrtm.load_model(FLAGS.pretrain_dir)
        nrtm.train()


if __name__ == '__main__':
    tf.app.run()
