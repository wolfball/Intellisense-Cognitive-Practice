#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from typing import Dict
from loguru import logger
from pprint import pformat
import torch
import numpy as np
import pandas as pd
import scipy
import sklearn.preprocessing as pre
import yaml
import augmentation as aug

# Some defaults for non-specified arguments in yaml
DEFAULT_ARGS = {
    'outputpath': 'experiments',
    'loss': 'BceLoss',
    'batch_size': 64,
    'num_workers': 2,
    'epochs': 100,
    'scheduler_args': {
        'patience': 3,
        'factor': 0.1,
    },
    'early_stop': 7,  # After how many itercv/epochs to stop if no improvement in loss
    'optimizer': {
        'type': 'Adam',
        'args': {
            'lr': 0.001,
        }
    },
}

# add data augmentation
def parse_transforms(transform_list):
    """parse_transforms
    parses the config files transformation strings to coresponding methods
    :param transform_list: String list
    """
    transforms = []

    for trans in transform_list:
        if trans == 'timeshift':
            transforms.append(aug.TimeShift(0, 50))
        elif trans == 'freqshift':
            transforms.append(aug.FreqShift(0, 4))
        elif trans == 'freqmask':
            transforms.append(aug.FreqMask(2, 8))
        elif trans == 'timemask':
            transforms.append(aug.TimeMask(2, 60))
        elif trans == 'noise':
            transforms.append(aug.AddNoise(0, 1e-5))
    return torch.nn.Sequential(*transforms)


def encode_label(label, label_to_idx):
    target = np.zeros(len(label_to_idx))
    if isinstance(label, str):
        label = label.split(",")
    for lb in label:
        target[label_to_idx[lb]] = 1
    return target


def parse_config_or_kwargs(config_file, **kwargs):
    """parse_config_or_kwargs

    :param config_file: Config file that has parameters, yaml format
    :param **kwargs: Other alternative parameters or overwrites for config
    """
    with open(config_file) as reader:
        yaml_config = yaml.load(reader, Loader=yaml.FullLoader)
    # values from config file are all possible params
    arguments = dict(yaml_config, **kwargs)
    # In case some arguments were not passed, replace with default ones
    for key, value in DEFAULT_ARGS.items():
        arguments.setdefault(key, value)
    return arguments


def dump_config(config_file, config):
    with open(config_file, "w") as writer:
        yaml.dump(config, writer, default_flow_style=False)


def find_contiguous_regions(activity_array):
    """Find contiguous regions from bool valued numpy.array.
    Copy of https://dcase-repo.github.io/dcase_util/_modules/dcase_util/data/decisions.html#DecisionEncoder

    Reason is:
    1. This does not belong to a class necessarily
    2. Import DecisionEncoder requires sndfile over some other imports..which causes some problems on clusters

    """

    # Find the changes in the activity_array
    change_indices = np.logical_xor(activity_array[1:],
                                    activity_array[:-1]).nonzero()[0]

    # Shift change_index with one, focus on frame after the change.
    change_indices += 1

    if activity_array[0]:
        # If the first element of activity_array is True add 0 at the beginning
        change_indices = np.r_[0, change_indices]

    if activity_array[-1]:
        # If the last element of activity_array is True, add the length of the array
        change_indices = np.r_[change_indices, activity_array.size]

    # Reshape the result into two columns
    return change_indices.reshape((-1, 2))


def split_train_cv(
        data_frame: pd.DataFrame,
        frac: float = 0.9,
        y=None,  # Only for stratified, computes necessary split
        stratified: bool = True):
    """split_train_cv

    :param data_frame:
    :type data_frame: pd.DataFrame
    :param frac:
    :type frac: float
    """
    if stratified:
        # Use statified sampling
        from skmultilearn.model_selection import iterative_train_test_split
        index_train, _, index_cv, _ = iterative_train_test_split(
            data_frame.index.values.reshape(-1, 1), y, test_size=1. - frac)
        train_data = data_frame[data_frame.index.isin(index_train.squeeze())]
        cv_data = data_frame[data_frame.index.isin(index_cv.squeeze())]
    else:
        # Simply split train_test
        train_data = data_frame.sample(frac=frac, random_state=10)
        cv_data = data_frame[~data_frame.index.isin(train_data.index)]
    return train_data, cv_data


def pprint_dict(in_dict, outputfun=sys.stdout.write, formatter='yaml'):
    """pprint_dict

    :param outputfun: function to use, defaults to sys.stdout
    :param in_dict: dict to print
    """
    if formatter == 'yaml':
        format_fun = yaml.dump
    elif formatter == 'pretty':
        format_fun = pformat
    for line in format_fun(in_dict).split('\n'):
        outputfun(line)


def getfile_outlogger(outputfile):
    log_format = "[<green>{time:YYYY-MM-DD HH:mm:ss}</green>] {message}"
    logger.configure(handlers=[{"sink": sys.stderr, "format": log_format}])
    if outputfile:
        logger.add(outputfile, enqueue=True, format=log_format)
    return logger


def decode_with_timestamps(idx_to_label: Dict, labels: np.array):
    """decode_with_timestamps
    Decodes the predicted label array (2d) into a list of
    [(Labelname, onset, offset), ...]

    :param encoder: Encoder during training
    :type encoder: pre.MultiLabelBinarizer
    :param labels: n-dim array
    :type labels: np.array
    """
    if labels.ndim == 3:
        return [_decode_with_timestamps(idx_to_label, lab) for lab in labels]
    else:
        return _decode_with_timestamps(idx_to_label, labels)


def median_filter(x, window_size, threshold=0.5):
    """median_filter

    :param x: input prediction array of shape (B, T, C) or (B, T).
        Input is a sequence of probabilities 0 <= x <= 1
    :param window_size: An integer to use 
    :param threshold: Binary thresholding threshold
    """
    x = binarize(x, threshold=threshold)
    if x.ndim == 3:
        size = (1, window_size, 1)
    elif x.ndim == 2 and x.shape[0] == 1:
        # Assume input is class-specific median filtering
        # E.g, Batch x Time  [1, 501]
        size = (1, window_size)
    elif x.ndim == 2 and x.shape[0] > 1:
        # Assume input is standard median pooling, class-independent
        # E.g., Time x Class [501, 10]
        size = (window_size, 1)
    return scipy.ndimage.median_filter(x, size=size)


def _decode_with_timestamps(idx_to_label, labels):
    result_labels = []
    for i, label_column in enumerate(labels.T):
        change_indices = find_contiguous_regions(label_column)
        # append [onset, offset] in the result list
        for row in change_indices:
            result_labels.append((idx_to_label[i], row[0], row[1]))
    return result_labels


def binarize(pred, threshold=0.5):
    # Batch_wise
    if pred.ndim == 3:
        return np.array(
            [pre.binarize(sub, threshold=threshold) for sub in pred])
    else:
        return pre.binarize(pred, threshold=threshold)


def predictions_to_time(df, ratio):
    df.onset = df.onset * ratio
    df.offset = df.offset * ratio
    return df

def double_threshold(x, high_thres, low_thres, n_connect=1):
    """double_threshold
    Helper function to calculate double threshold for n-dim arrays
    :param x: input array
    :param high_thres: high threshold value
    :param low_thres: Low threshold value
    :param n_connect: Distance of <= n clusters will be merged
    """
    assert x.ndim <= 3, "Whoops something went wrong with the input ({}), check if its <= 3 dims".format(
        x.shape)
    if x.ndim == 3:
        apply_dim = 1
    elif x.ndim < 3:
        apply_dim = 0
    # x is assumed to be 3d: (batch, time, dim)
    # Assumed to be 2d : (time, dim)
    # Assumed to be 1d : (time)
    # time axis is therefore at 1 for 3d and 0 for 2d (
    return np.apply_along_axis(lambda x: _double_threshold(
        x, high_thres, low_thres, n_connect=n_connect),
                               axis=apply_dim,
                               arr=x)


def _double_threshold(x, high_thres, low_thres, n_connect=1, return_arr=True):
    """_double_threshold
    Computes a double threshold over the input array
    :param x: input array, needs to be 1d
    :param high_thres: High threshold over the array
    :param low_thres: Low threshold over the array
    :param n_connect: Postprocessing, maximal distance between clusters to connect
    :param return_arr: By default this function returns the filtered indiced, but if return_arr = True it returns an array of tsame size as x filled with ones and zeros.
    """
    assert x.ndim == 1, "Input needs to be 1d"
    high_locations = np.where(x > high_thres)[0]
    locations = x > low_thres
    encoded_pairs = find_contiguous_regions(locations)

    filtered_list = list(
        filter(
            lambda pair:
            ((pair[0] <= high_locations) & (high_locations <= pair[1])).any(),
            encoded_pairs))

    filtered_list = connect_(filtered_list, n_connect)
    if return_arr:
        zero_one_arr = np.zeros_like(x, dtype=int)
        for sl in filtered_list:
            zero_one_arr[sl[0]:sl[1]] = 1
        return zero_one_arr
    return filtered_list

def connect_clusters(x, n=1):
    if x.ndim == 1:
        return connect_clusters_(x, n)
    if x.ndim >= 2:
        return np.apply_along_axis(lambda a: connect_clusters_(a, n=n), -2, x)


def connect_clusters_(x, n=1):
    """connect_clusters_
    Connects clustered predictions (0,1) in x with range n
    :param x: Input array. zero-one format
    :param n: Number of frames to skip until connection can be made
    """
    assert x.ndim == 1, "input needs to be 1d"
    reg = find_contiguous_regions(x)
    start_end = connect_(reg, n=n)
    zero_one_arr = np.zeros_like(x, dtype=int)
    for sl in start_end:
        zero_one_arr[sl[0]:sl[1]] = 1
    return zero_one_arr


def connect_(pairs, n=1):
    """connect_
    Connects two adjacent clusters if their distance is <= n
    :param pairs: Clusters of iterateables e.g., [(1,5),(7,10)]
    :param n: distance between two clusters
    """
    if len(pairs) == 0:
        return []
    start_, end_ = pairs[0]
    new_pairs = []
    for i, (next_item, cur_item) in enumerate(zip(pairs[1:], pairs[0:])):
        end_ = next_item[1]
        if next_item[0] - cur_item[1] <= n:
            pass
        else:
            new_pairs.append((start_, cur_item[1]))
            start_ = next_item[0]
    new_pairs.append((start_, end_))
    return new_pairs


def predictions_to_time(df, ratio):
    df.onset = df.onset * ratio
    df.offset = df.offset * ratio
    return df