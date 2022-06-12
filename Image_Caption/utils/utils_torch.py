import itertools
import os

import numpy as np
from PIL import Image


def preprocess_input(x):
    x -= 0.5
    x *= 2.
    return x


# returns (3, h, w)
def preprocess(image_path, trans):
    img = Image.open(image_path).convert('RGB')
    x = trans(img)
    x = x.unsqueeze(0)
    # x = preprocess_input(x)
    return x


def split_data(l, img):
    temp = []
    for i in img:
        if os.path.basename(i) in l:
            temp.append(i)
    return temp


def padding_tensor(sequences, maxlen):
    """
    :param sequences: list of tensors
    :param maxlen: fixed length of output tensors
    :return:
    """
    num = len(sequences)
    # max_len = max([s.size(0) for s in sequences])
    out_dims = (num, maxlen)
    out_tensor = sequences[0].data.new(*out_dims).fill_(0)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        out_tensor[i, :length] = tensor
    return out_tensor


def words_from_tensors_fn(idx2word, max_len=40, startseq='<start>', endseq='<end>'):
    def words_from_tensors(captions: np.array) -> list:
        """
        :param captions: [b, max_len]
        :return:
        """
        captoks = []
        for capidx in captions:
            # capidx = [1, max_len]
            captoks.append(list(itertools.takewhile(lambda word: word != endseq,
                                                    map(lambda idx: idx2word[idx], iter(capidx))))[1:])
        return captoks

    return words_from_tensors


