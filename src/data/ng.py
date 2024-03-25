import os
import sys
import numpy as np
import torch
import torch.nn as nn

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


def ng_loader(root: str, train=True):
    """Loads the 20 newsgroup dataset.

    Adapted from https://github.com/torrvision/focal_calibration/blob/nlp/NewsGroup/ng_loader.py

    Args:
        train (bool): If True, load the training set, otherwise load the test set.
    """
    BASE_DIR = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "NewsGroup")
    GLOVE_DIR = os.path.join(BASE_DIR, "glove.6B")
    TEXT_DATA_DIR = os.path.join(BASE_DIR, "20_newsgroup")
    MAX_SEQUENCE_LENGTH = 100
    MAX_NUM_WORDS = 20000
    EMBEDDING_DIM = 100
    print("Indexing word vectors.")

    embeddings_index = {}

    f = open(os.path.join(GLOVE_DIR, "glove.6B.100d.txt"), encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        embeddings_index[word] = coefs
    f.close()

    print("Found %s word vectors." % len(embeddings_index))

    # second, prepare text samples and their labels
    print("Processing text dataset")

    texts = []  # list of text samples
    labels_index = {}  # dictionary mapping label name to numeric id
    labels = []  # list of label ids
    for name in sorted(os.listdir(TEXT_DATA_DIR)):
        path = os.path.join(TEXT_DATA_DIR, name)
        if os.path.isdir(path):
            label_id = len(labels_index)
            labels_index[name] = label_id
            for fname in sorted(os.listdir(path)):
                if fname.isdigit():
                    fpath = os.path.join(path, fname)
                    if sys.version_info < (3,):
                        f = open(fpath)
                    else:
                        f = open(fpath, encoding="latin-1")
                    t = f.read()
                    i = t.find("\n\n")  # skip header
                    if 0 < i:
                        t = t[i:]
                    texts.append(t)
                    f.close()
                    labels.append(label_id)

    print("Found %s texts." % len(texts))

    # finally, vectorize the text samples into a 2D integer tensor
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index
    print("Found %s unique tokens." % len(word_index))

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    labels = to_categorical(np.asarray(labels))

    print("Shape of data tensor:", data.shape)
    print("Shape of label tensor:", labels.shape)

    # split the data into a training set and a validation set
    indices = np.arange(data.shape[0])
    np.random.seed(1)
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]

    if train:
        x_data = data[:-900]  # Train set
        y_data = labels[:-900]
    else:
        x_data = data[-900:]  # Test set
        y_data = labels[-900:]

    print(data.shape[0])

    # prepare embedding matrix
    num_words = min(MAX_NUM_WORDS, len(word_index))
    embedding_matrix = torch.zeros(num_words, EMBEDDING_DIM)
    for word, i in word_index.items():
        if i >= MAX_NUM_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = torch.from_numpy(embedding_vector)

    x_data = torch.from_numpy(x_data).type(torch.LongTensor)
    y_data = torch.from_numpy(np.argmax(y_data, 1))

    return embedding_matrix, x_data, y_data, num_words, EMBEDDING_DIM


class NewsGroupDataset(torch.utils.data.Dataset):
    """20 newsgroup dataset.

    Args:
        root (str): Root directory of dataset where ``NewsGroup`` folder exists.
        train (bool): If True, load the training set, otherwise load the test set.
    """

    def __init__(self, root: str, train=True):
        self.train = train
        (
            self.embedding_matrix,
            self.x_data,
            self.y_data,
            self.num_words,
            self.EMBEDDING_DIM,
        ) = ng_loader(root=root, train=self.train)

        self.embedding_model = nn.Embedding(self.num_words, self.EMBEDDING_DIM)
        self.embedding_model.state_dict()["weight"].copy_(
            self.embedding_matrix)

    def __getitem__(self, index):
        x = self.x_data[index]
        y = self.y_data[index]

        with torch.no_grad():
            x = self.embedding_model(x)

        return x, y

    def __len__(self):
        return len(self.x_data)
