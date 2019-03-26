# coding: utf-8
"""
Data module
"""
import sys
import os
import os.path
from typing import Optional
import random
import torch

from torchtext.datasets import TranslationDataset
from torchtext import data
from torchtext.data import Dataset, Field, Batch

from joeynmt.constants import UNK_TOKEN, EOS_TOKEN, BOS_TOKEN, PAD_TOKEN
from joeynmt.vocabulary import build_vocab, Vocabulary


def load_data(data_cfg: dict) -> (Dataset, Dataset, Optional[Dataset],
                                  Vocabulary, Vocabulary):
    """
    Load train, dev and optionally test data as specified in configuration.
    Vocabularies are created from the training set with a limit of `voc_limit`
    tokens and a minimum token frequency of `voc_min_freq`
    (specified in the configuration dictionary).

    The training data is filtered to include sentences up to `max_sent_length`
    on source and target side.

    :param data_cfg: configuration dictionary for data
        ("data" part of configuation file)
    :return:
        - train_data: training dataset
        - dev_data: development dataset
        - test_data: testdata set if given, otherwise None
        - src_vocab: source vocabulary extracted from training data
        - trg_vocab: target vocabulary extracted from training data
    """
    # load data from files
    src_lang = data_cfg["src"]
    trg_lang = data_cfg["trg"]
    train_path = data_cfg["train"]
    dev_path = data_cfg["dev"]
    test_path = data_cfg.get("test", None)
    level = data_cfg["level"]
    lowercase = data_cfg["lowercase"]
    max_sent_length = data_cfg["max_sent_length"]

    tok_fun = lambda s: list(s) if level == "char" else s.split()

    src_field = data.Field(init_token=None, eos_token=EOS_TOKEN,
                           pad_token=PAD_TOKEN, tokenize=tok_fun,
                           batch_first=True, lower=lowercase,
                           unk_token=UNK_TOKEN,
                           include_lengths=True)

    trg_field = data.Field(init_token=BOS_TOKEN, eos_token=EOS_TOKEN,
                           pad_token=PAD_TOKEN, tokenize=tok_fun,
                           unk_token=UNK_TOKEN,
                           batch_first=True, lower=lowercase,
                           include_lengths=True)

    train_data = TranslationDataset(path=train_path,
                                    exts=("." + src_lang, "." + trg_lang),
                                    fields=(src_field, trg_field),
                                    filter_pred=
                                    lambda x: len(vars(x)['src'])
                                    <= max_sent_length
                                    and len(vars(x)['trg'])
                                    <= max_sent_length)

    src_max_size = data_cfg.get("src_voc_limit", sys.maxsize)
    src_min_freq = data_cfg.get("src_voc_min_freq", 1)
    trg_max_size = data_cfg.get("trg_voc_limit", sys.maxsize)
    trg_min_freq = data_cfg.get("trg_voc_min_freq", 1)

    src_vocab_file = data_cfg.get("src_vocab", None)
    trg_vocab_file = data_cfg.get("trg_vocab", None)

    src_vocab = build_vocab(field="src", min_freq=src_min_freq,
                            max_size=src_max_size,
                            dataset=train_data, vocab_file=src_vocab_file)
    trg_vocab = build_vocab(field="trg", min_freq=trg_min_freq,
                            max_size=trg_max_size,
                            dataset=train_data, vocab_file=trg_vocab_file)
    dev_data = TranslationDataset(path=dev_path,
                                  exts=("." + src_lang, "." + trg_lang),
                                  fields=(src_field, trg_field))
    test_data = None
    if test_path is not None:
        # check if target exists
        if os.path.isfile(test_path + "." + trg_lang):
            test_data = TranslationDataset(
                path=test_path, exts=("." + src_lang, "." + trg_lang),
                fields=(src_field, trg_field))
        else:
            # no target is given -> create dataset from src only
            test_data = MonoDataset(path=test_path, ext="." + src_lang,
                                    field=src_field)
    src_field.vocab = src_vocab
    trg_field.vocab = trg_vocab
    return train_data, dev_data, test_data, src_vocab, trg_vocab


class DataIterator:

    def __init__(self, dataset, batch_size, shuffle, sort_key, train,
                 bucket_size=100,
                 device=None):
        """
        Iterator that generates batches of examples from the dataset.

        During training, the data is split into buckets, which are sorted
        by length, before they are split into batches.
        This reduces the necessary padding to the longest element in the batch.

        TODOs
            - flags?
            - check whether one-time generation is faster: make all the buckets
            - skip smaller batches during training?
            - add custom batch
            - CUDA OOM?

        :param dataset: dataset to build an iterator for
        :param batch_size: size of mini-batches
        :param bucket_size: size of buckets for efficient composition of
            training batches
        :param shuffle: shuffle the order of instances.
            Is set to False automatically when train=False.
        :param sort_key: key to sort instances by. Required when train=True.
        :param train: whether this is training or not.
            Influences sorting and shuffling.
        :param device: device to create batches on. -1 for CPU and None for the
            currently active GPU device.
        """

        if train:
            self.shuffle = shuffle
            self.sort_key = sort_key
        else:
            self.shuffle = False
            self.sort_key = None

        self.dataset = dataset
        self.batches = []
        self.batch_size = batch_size
        self.train = train
        self.bucket_size = bucket_size
        # TODO
        self.device = device
        if not torch.cuda.is_available() and self.device is None:
            self.device = -1

    def _init_epoch(self):
        """
        Called at beginning of each epoch

        :return:
        """
        # create new index
        self.data_indices = [i for i in range(len(self.dataset))]
        if self.shuffle:
            random.shuffle(self.data_indices)
        self._make_batches()

    def _make_batches(self):
        """
        Make one bucket full of batches

        """
        # list of indices to split into batches next
        bucket = self._fill_bucket()
        # split bucket into batches
        self.batches.extend(self._batches_from_bucket(bucket))

    def __next__(self):
        b = self._get_batch()
        if b is None:
            raise StopIteration
        else:  # TODO put custom batch here
            return Batch(b, self.dataset, self.device, self.train)

    def __iter__(self):
        self._init_epoch()
        return self

    def _get_batch(self):
        """
        Get the next batch

        :return:
        """
        if self.data_indices and not self.batches:  # need to make more batches
            self._make_batches()
        if self.batches:  # there are prepared batches left
            return self.batches.pop(0)
        else:  # end of epoch
            return

    def _batches_from_bucket(self, bucket):
        """
        Split bucket into batches.

        :param bucket: list of indices over dataset
        :return: list of list of indices over dataset, each list is contains
            max `self.batch_size` indices
        """
        if self.sort_key is not None and self.train:  # sort elements in bucket
            # elements within batches will be sorted likewise
            bucket = sorted(bucket, reverse=True,
                            key=lambda b: self.sort_key(self.dataset[b]))
        batches = []
        while bucket:  # split into batches
            batch = []
            # TODO decide whether to discard too small batches
            for i in range(min(self.batch_size, len(bucket))):
                batch_id = bucket.pop(0)
                batch.append(self.dataset[batch_id])
            batches.append(batch)
        if self.shuffle:  # shuffle batch order if required
            random.shuffle(batches)
        return batches

    def _fill_bucket(self):
        """
        Prepare new batches with bucketing.

        :param bucket_size:
        :return:
        """
        # first collect a new bucket of examples
        i = 0
        bucket = []
        before_bucket = len(self.data_indices)
        while i < self.bucket_size*self.batch_size and self.data_indices:
            idx = self.data_indices.pop(0)  # from 0 for validation order
            bucket.append(idx)
            i += 1
        assert len(self.data_indices)+len(bucket) == before_bucket
        return bucket


class MonoDataset(Dataset):
    """Defines a dataset for machine translation without targets."""

    @staticmethod
    def sort_key(ex):
        return len(ex.src)

    def __init__(self, path: str, ext: str, field: Field, **kwargs) -> None:
        """
        Create a monolingual dataset (=only sources) given path and field.

        :param path: Prefix of path to the data file
        :param ext: Containing the extension to path for this language.
        :param field: Containing the fields that will be used for data.
        :param kwargs: Passed to the constructor of data.Dataset.
        """

        fields = [('src', field)]

        src_path = os.path.expanduser(path + ext)

        examples = []
        with open(src_path) as src_file:
            for src_line in src_file:
                src_line = src_line.strip()
                if src_line != '':
                    examples.append(data.Example.fromlist(
                        [src_line], fields))

        super(MonoDataset, self).__init__(examples, fields, **kwargs)
