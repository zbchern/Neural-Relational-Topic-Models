import os
import scipy.io
import numpy as np
import random
import math
import tensorflow as tf
import logging

np.random.seed(0)
tf.set_random_seed(0)


def set_best_gpu(top_k=1):
    best_gpu = _scan(top_k)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, best_gpu))
    return best_gpu


def _scan(top_k):
    CMD1 = 'nvidia-smi| grep MiB | grep -v Default | cut -c 4-8'
    CMD2 = 'nvidia-smi -L | wc -l'
    CMD3 = 'nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits'

    total_gpu = int(os.popen(CMD2).read())

    assert top_k <= total_gpu, 'top_k > total_gpu !'

    # first choose the free gpus
    gpu_usage = set(map(lambda x: int(x), os.popen(CMD1).read().split()))
    free_gpus = set(range(total_gpu)) - gpu_usage

    # then choose the most memory free gpus
    gpu_free_mem = list(map(lambda x: int(x), os.popen(CMD3).read().split()))
    gpu_sorted = list(sorted(range(total_gpu), key=lambda x: gpu_free_mem[x], reverse=True))[len(free_gpus):]

    res = list(free_gpus) + list(gpu_sorted)
    return res[:top_k]


def init_logging(log_path):
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    logFormatter = logging.Formatter('%(asctime)s [%(levelname)s]: %(message)s')

    fileHandler = logging.FileHandler(log_path)
    fileHandler.setFormatter(logFormatter)
    log.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    log.addHandler(consoleHandler)


def print_settings(FLAGS):
    logging.info("################ Hyper Settings ################")
    for keys, values in zip(FLAGS.__flags.keys(), FLAGS.__flags.values()):
        values = str(values)
        logging.info(keys + ": " + values)
    logging.info("################ Hyper Settings ################")


def load_data(data_dir, negative_num):
    data_dict = {}
    doc_contents = scipy.io.loadmat(data_dir + 'mult_nor.mat')['X']
    links = load_links(data_dir + 'citations.dat')
    # train_links, test_links = split_links(links)
    train_links, test_links = data_split(links, 0.8)

    train_links_neg, train_labels_neg = add_negatives(links, train_links, negative_num)
    test_links_hit = prepare_hit_data(links, test_links, 99)

    vocab = load_vocab(data_dir + "vocabulary.dat")

    data_dict['links'] = links
    data_dict['doc_contents'] = doc_contents
    data_dict['train_links'] = train_links
    data_dict['test_links'] = test_links
    data_dict['train_links_neg'] = train_links_neg
    data_dict['train_labels_neg'] = train_labels_neg
    data_dict['test_links_hit'] = test_links_hit
    data_dict['vocab'] = vocab

    return data_dict


def prepare_hit_data(links, data, negative_num):
    total_links = []

    link_num = len(data)

    for i in range(link_num):

        if len(data[i]) == 0:
            continue

        for j in data[i]:
            link_tmp = [i, j]
            for k in range(negative_num):
                r = np.random.randint(link_num)
                while r in links[i]:
                    r = np.random.randint(link_num)
                link_tmp.append(r)
            total_links.append(link_tmp)

    return total_links


def add_negatives(links, data, negative_num):

    total_links = []
    total_labels = []

    link_num = len(data)

    for i in range(link_num):
        for j in data[i]:
            # actual link
            total_links.append([i, j])
            total_labels.append(1)
            # negative links
            for k in range(negative_num):
                r = np.random.randint(link_num)
                while r in links[i]:
                    r = np.random.randint(link_num)
                total_links.append([i, r])
                total_labels.append(0)

    total = list(zip(total_links, total_labels))
    random.shuffle(total)
    total_links[:], total_labels[:] = zip(*total)

    return total_links, total_labels


def print_top_words(weights, vocab, dataset, n_top_words=10):

    logging.info('---------------Printing the Topics------------------')
    topics = []
    if not os.path.exists('topics/'):
        os.makedirs('topics/')
    
    with open('topics/topics_' + dataset + '.txt', 'a') as writer:
        for i in range(len(weights)):
            topic = (" ".join([vocab[j] for j in weights[i].argsort()[:-n_top_words - 1:-1]]))
            writer.write(topic)
            writer.write('\n')
            logging.info(topic)
            topics.append(topic)
        writer.write('******\n')
    logging.info('---------------End of Topics------------------\n')
    return topics


def get_top_words(weights, vocab, n_top_words=10):

    topics = []
    for i in range(len(weights)):
        topic = (" ".join([vocab[j] for j in weights[i].argsort()[:-n_top_words - 1:-1]]))
        topics.append(topic)
    return topics


def load_links(path):
    links = []
    ind = 0
    for line in open(path):
        arr = line.strip().split()
        arr = [int(x) for x in arr]
        this_num_links = arr[0]
        if this_num_links == 0:
            links.append([])
            ind += 1
            continue
        links.append(arr[1:])
        ind += 1
    return links


def data_split(links, ratio_train=0.8):
    # filtered = [i for i in range(len(links)) if len(links[i])>0]
    filtered = range(len(links))
    num_total = len(filtered)
    num_train = int(num_total * ratio_train)
    num_test = num_total - num_train
    perm_idx = np.random.permutation(filtered)
    train_idx = perm_idx[:num_train]
    test_idx = perm_idx[num_train:]

    train_links = [None] * num_total
    for i in range(num_total):
        train_links[i] = []
    num_train_links = 0
    for i in train_idx:
        train_links[i] = links[i]
        num_train_links += len(train_links[i])
    num_train_links /= 2

    test_links = [None] * num_total
    for i in range(num_total):
        test_links[i] = []
    num_test_links = 0
    for i in test_idx:
        this_link = links[i]
        if len(this_link) and this_link[-1] == i:
            this_link = this_link[:-1]
        test_links[i] = this_link
        num_test_links += len(test_links[i])

    return train_links, test_links


def add_noise(x, noise_type):
    if noise_type == 'gaussian':
        noise = np.random.normal(0, 0.1, (len(x), len(x[0])))
        return x + noise
    elif noise_type.__contains__('mask'):
        frac = float(noise_type.split('-')[1])
        x_tmp = np.copy(x)
        for i in x_tmp:
            noise = np.random.choice(len(i), int(round(frac * len(i))), replace=False)
            i[noise] = 0
        return x_tmp
    else:
        return x


def getNDCG(ranklist, gtItem):
    for i in xrange(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2.) / math.log(i + 2.)
    return 0.


def getHits(ranklist, gtItem):
    if gtItem in ranklist:
        return 1.
    else:
        return 0.


def activate(net, activator):
    if activator == 'sigmoid':
        return tf.nn.sigmoid(net, name='encoded')
    elif activator == 'softmax':
        return tf.nn.softmax(net, name='encoded')
    elif activator == 'linear':
        return net
    elif activator == 'tanh':
        return tf.nn.tanh(net, name='encoded')
    elif activator == 'relu':
        return tf.nn.relu(net, name='encoded')


def load_vocab(path):
    vocab = []
    for line in open(path):
        arr = line.strip().split()
        vocab.append(arr[0])
    return vocab


def get_batch(X, batch_size):
    ids = np.random.choice(len(X), batch_size, replace=False)
    return X[ids], ids


def load_batch(data, label, ind, batch_size):
    if ind + batch_size < len(data):
        batch_data = data[ind * batch_size:(ind + 1) * batch_size]
        batch_label = label[ind * batch_size:(ind + 1) * batch_size]
    else:
        batch_data = data[ind * batch_size:]
        batch_label = label[ind * batch_size:]
    return batch_data, batch_label


def load_batch_test(id, link_ids, num_doc):
    # return doc wise testing data.
    batch_label = np.zeros(num_doc)
    batch_label[link_ids] = 1
    batch_data = [np.asarray([id, trg]) for trg in range(num_doc)]
    return batch_data, batch_label
