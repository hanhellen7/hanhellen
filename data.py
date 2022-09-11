# -*- coding: utf-8 -*-
"""data helper function."""
import pickle
import os
import random
import numpy as np
from config import tag2label
from tqdm import tqdm
from glob import glob


def read_corpus(corpus_path):
    """
    Read corpus and return samples list.

    :param corpus_path: corpus path
    :return: data contain char list and tag list
    """
    data = []
    with open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()
    sent_, tag_ = [], []
    for i, line in enumerate(tqdm(lines)):
        if line != '\n':
            [char, label] = line.replace('\n', '').strip().split('\t')
            sent_.append(char)
            tag_.append(label)
        else:
            data.append((sent_, tag_))
            sent_, tag_ = [], []

        if i == len(lines) - 1:
            data.append((sent_, tag_))
    return data


def vocab_build(vocab_path, corpus_path, min_count):
    """
    Build vocabulary to index dictionary.

    :param vocab_path: vocab path
    :param corpus_path: corpus path
    :param min_count: minimum word frequence in word to id
    :return: dictionary transform vocabulary to index
    """
    data = read_corpus(corpus_path)
    word2id = {}
    for sent_, tag_ in data:
        for word in sent_:
            if word.isdigit():
                word = '<NUM>'
            elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
                word = '<ENG>'
            if word not in word2id:
                word2id[word] = [len(word2id) + 1, 1]
            else:
                word2id[word][1] += 1
    low_freq_words = []
    for word, [word_id, word_freq] in word2id.items():
        if word_freq < min_count and word != '<NUM>' and word != '<ENG>':
            low_freq_words.append(word)
    for word in low_freq_words:
        del word2id[word]

    new_id = 1
    for word in word2id.keys():
        word2id[word] = new_id
        new_id += 1
    word2id['<UNK>'] = new_id
    word2id['<PAD>'] = 0

    print(len(word2id))
    with open(vocab_path, 'wb') as fw:
        pickle.dump(word2id, fw)
    return word2id


def sentence2id(sent, word2id):
    """
    Transform sentence to index list.

    :param sent: single text line
    :param word2id: dict
    :return: sentence index list
    """
    sentence_id = []
    for word in sent:
        if word.isdigit():
            word = '<NUM>'
        elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
            word = '<ENG>'
        if word not in word2id:
            word = '<UNK>'
        sentence_id.append(word2id[word])
    return sentence_id


def read_dictionary(vocab_path):
    """
    Load dictionary.

    :param vocab_path:vocab path
    :return:dict
    """
    vocab_path = os.path.join(vocab_path)
    with open(vocab_path, 'rb') as fr:
        word2id = pickle.load(fr)
    print('vocab_size:', len(word2id))
    return word2id


def random_embedding(vocab, embedding_dim):
    """
    Create random embedding.

    :param vocab: word list
    :param embedding_dim: embedding dim
    :return: random word embedding list
    """
    embedding_mat = np.random.uniform(-0.25, 0.25, (len(vocab), embedding_dim))
    embedding_mat = np.float32(embedding_mat)
    return embedding_mat


def pad_sequences(sequences, pad_mark='O'):
    """
    Change seq's length.

    :param sequences: sentence index list
    :param pad_mark: token pad sequence
    :return: seq list and seq len list
    """
    max_len = max(map(lambda x: len(x), sequences))
    seq_list, seq_len_list = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
        seq_list.append(seq_)
        seq_len_list.append(min(len(seq), max_len))
    return seq_list, seq_len_list


def batch_yield(data, batch_size, vocab, tag2label, shuffle=False):
    """
    Yield batch to feed.

    :param data: input data
    :param batch_size:same as model's batch size
    :param vocab: word to id dict
    :param tag2label: tag to label dict
    :param shuffle: shuffle input data
    :return:char list and lable list
    """
    if shuffle:
        random.shuffle(data)

    seqs, labels = [], []
    for (sent_, tag_) in data:
        sent_ = sentence2id(sent_, vocab)
        label_ = [tag2label[tag] for tag in tag_]

        if len(seqs) == batch_size:
            yield seqs, labels
            seqs, labels = [], []

        seqs.append(sent_)
        labels.append(label_)

    if len(seqs) != 0:
        yield seqs, labels
