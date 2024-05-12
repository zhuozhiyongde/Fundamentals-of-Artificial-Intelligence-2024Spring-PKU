import gzip
import tarfile
import torch
import six.moves.cPickle as pickle
import numpy as np
from urllib.request import urlretrieve
import os
import math
import progressbar
import logging


def exists_or_mkdir(path, verbose=True):
    """Check a folder by given name, if not exist, create the folder and return False,
    if directory exists, return True.

    Parameters
    ----------
    path : str
        A folder path.
    verbose : boolean
        If True (default), prints results.

    Returns
    --------
    boolean
        True if folder already exist, otherwise, returns False and create the folder.

    Examples
    --------
    >>> tlx.files.exists_or_mkdir("checkpoints/train")

    """
    if not os.path.exists(path):
        if verbose:
            logging.info("[*] creates %s ..." % path)
        os.makedirs(path)
        return False
    else:
        if verbose:
            logging.info("[!] %s exists ..." % path)
        return True


def maybe_download_and_extract(filename, working_directory, url_source, extract=False, expected_bytes=None):
    """Checks if file exists in working_directory otherwise tries to dowload the file,
    and optionally also tries to extract the file if format is ".zip" or ".tar"

    Parameters
    -----------
    filename : str
        The name of the (to be) dowloaded file.
    working_directory : str
        A folder path to search for the file in and dowload the file to
    url : str
        The URL to download the file from
    extract : boolean
        If True, tries to uncompress the dowloaded file is ".tar.gz/.tar.bz2" or ".zip" file, default is False.
    expected_bytes : int or None
        If set tries to verify that the downloaded file is of the specified size, otherwise raises an Exception, defaults is None which corresponds to no check being performed.

    Returns
    ----------
    str
        File path of the dowloaded (uncompressed) file.

    Examples
    --------
    >>> down_file = tlx.files.maybe_download_and_extract(filename='train-images-idx3-ubyte.gz',
    ...                                            working_directory='data/',
    ...                                            url_source='http://yann.lecun.com/exdb/mnist/')
    >>> tlx.files.maybe_download_and_extract(filename='ADEChallengeData2016.zip',
    ...                                             working_directory='data/',
    ...                                             url_source='http://sceneparsing.csail.mit.edu/data/',
    ...                                             extract=True)

    """

    # We first define a download function, supporting both Python 2 and 3.
    def _download(filename, working_directory, url_source):

        progress_bar = progressbar.ProgressBar()

        def _dlProgress(count, blockSize, totalSize, pbar=progress_bar):
            if (totalSize != 0):

                if not pbar.max_value:
                    totalBlocks = math.ceil(float(totalSize) / float(blockSize))
                    pbar.max_value = int(totalBlocks)

                pbar.update(count, force=True)

        filepath = os.path.join(working_directory, filename)

        logging.info('Downloading %s...\n' % filename)

        urlretrieve(url_source + filename, filepath, reporthook=_dlProgress)

    exists_or_mkdir(working_directory, verbose=False)
    filepath = os.path.join(working_directory, filename)

    if not os.path.exists(filepath):

        _download(filename, working_directory, url_source)
        statinfo = os.stat(filepath)
        logging.info('Succesfully downloaded %s %s bytes.' % (filename, statinfo.st_size))  # , 'bytes.')
        if (not (expected_bytes is None) and (expected_bytes != statinfo.st_size)):
            raise Exception('Failed to verify ' + filename + '. Can you get to it with a browser?')
        if (extract):
            if tarfile.is_tarfile(filepath):
                logging.info('Trying to extract tar file')
                tarfile.open(filepath, 'r').extractall(working_directory)
                logging.info('... Success!')
            elif zipfile.is_zipfile(filepath):
                logging.info('Trying to extract zip file')
                with zipfile.ZipFile(filepath) as zf:
                    zf.extractall(working_directory)
                logging.info('... Success!')
            else:
                logging.info("Unknown compression_format only .tar.gz/.tar.bz2/.tar and .zip supported")
    return filepath


def load_imdb_dataset(path='data', nb_words=None, skip_top=0, maxlen=None, test_split=0.2, seed=113, start_char=1, oov_char=2, index_from=3):
    """Load IMDB dataset.

    Parameters
    ----------
    path : str
        The path that the data is downloaded to, defaults is ``data/imdb/``.
    nb_words : int
        Number of words to get.
    skip_top : int
        Top most frequent words to ignore (they will appear as oov_char value in the sequence data).
    maxlen : int
        Maximum sequence length. Any longer sequence will be truncated.
    seed : int
        Seed for reproducible data shuffling.
    start_char : int
        The start of a sequence will be marked with this character. Set to 1 because 0 is usually the padding character.
    oov_char : int
        Words that were cut out because of the num_words or skip_top limit will be replaced with this character.
    index_from : int
        Index actual words with this index and higher.

    Examples
    --------
    >>> X_train, y_train, X_test, y_test = tlx.files.load_imdb_dataset(
    ...                                 nb_words=20000, test_split=0.2)
    >>> print('X_train.shape', X_train.shape)
    (20000,)  [[1, 62, 74, ... 1033, 507, 27],[1, 60, 33, ... 13, 1053, 7]..]
    >>> print('y_train.shape', y_train.shape)
    (20000,)  [1 0 0 ..., 1 0 1]

    References
    -----------
    - `Modified from keras. <https://github.com/fchollet/keras/blob/master/keras/datasets/imdb.py>`__

    """
    path = os.path.join(path, 'imdb')

    filename = "imdb.pkl"
    url = 'https://s3.amazonaws.com/text-datasets/'
    maybe_download_and_extract(filename, path, url)

    if filename.endswith(".gz"):
        f = gzip.open(os.path.join(path, filename), 'rb')
    else:
        f = open(os.path.join(path, filename), 'rb')

    X, labels = pickle.load(f)
    f.close()

    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(labels)

    if start_char is not None:
        X = [[start_char] + [w + index_from for w in x] for x in X]
    elif index_from:
        X = [[w + index_from for w in x] for x in X]

    if maxlen:
        new_X = []
        new_labels = []
        for x, y in zip(X, labels):
            if len(x) < maxlen:
                new_X.append(x)
                new_labels.append(y)
        X = new_X
        labels = new_labels
    if not X:
        raise Exception('After filtering for sequences shorter than maxlen=' + str(maxlen) + ', no sequence was kept. '
                        'Increase maxlen.')
    if not nb_words:
        nb_words = max([max(x) for x in X])

    # by convention, use 2 as OOV word
    # reserve 'index_from' (=3 by default) characters: 0 (padding), 1 (start), 2 (OOV)
    if oov_char is not None:
        X = [[oov_char if (w >= nb_words or w < skip_top) else w for w in x] for x in X]
    else:
        nX = []
        for x in X:
            nx = []
            for w in x:
                if (w >= nb_words or w < skip_top):
                    nx.append(w)
            nX.append(nx)
        X = nX

    X_train = X[:int(len(X) * (1 - test_split))]
    y_train = labels[:int(len(X) * (1 - test_split))]

    X_test = X[int(len(X) * (1 - test_split)):]
    y_test = labels[int(len(X) * (1 - test_split)):]

    return X_train, y_train, X_test, y_test


class Accuracy:

    def __init__(self, topk=1):
        super(Accuracy, self).__init__()
        self.topk = topk
        self.reset()

    def update(self, y_pred, y_true):

        y_pred = torch.argsort(y_pred, dim=-1, descending=True)
        y_pred = y_pred[:, :self.topk]
        if (len(y_true.shape) == 1) or (len(y_true.shape) == 2 and y_true.shape[-1] == 1):
            y_true = torch.reshape(y_true, (-1, 1))
        elif y_true.shape[-1] != 1:
            y_true = torch.argmax(y_true, dim=-1, keepdim=True)
        correct = y_pred == y_true
        correct = correct.to(torch.float32)
        correct = correct.cpu().numpy()
        num_samples = np.prod(np.array(correct.shape[:-1]))
        num_corrects = correct[..., :self.topk].sum()
        self.total += num_corrects
        self.count += num_samples

    def result(self):
        return float(self.total) / self.count if self.count > 0 else 0.

    def reset(self):
        self.total = 0.0
        self.count = 0.0
