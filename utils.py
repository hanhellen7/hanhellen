# -*- coding: utf-8 -*-
import logging
from config import tag2label


def get_entity(tag_seq, char_seq):
    """
    Extract word and label form seq.

    :param tag_seq: tag list
    :param char_seq: char list
    :return: entity_dict
    """
    entity_dict = {}
    for tag_index, tag in yeild_entity(tag_seq, char_seq):
        entity_dict.setdefault(tag, []).append("".join(char_seq[tag_index[0]:tag_index[1]+1]))
    return entity_dict


def yeild_entity(tag_seq, char_seq):
    """
    Get word's index and tag.

    :param tag_seq: tag list
    :param char_seq: char list
    :return: tag_index, tag
    """
    no_head_tag_seq = [t.split("-")[-1] for t in tag_seq]
    tag_index = [0, 0]
    for i, tag in enumerate(no_head_tag_seq):
        if i == 0:
            pre_tag = tag
        else:
            if pre_tag == tag:
                tag_index[1] += 1
            else:
                yield tag_index, pre_tag
                tag_index = [i, i]
                pre_tag = tag
    yield tag_index, tag


def get_logger(filename):
    """Init logger."""
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    handler = logging.FileHandler(filename,mode='a')
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)
    return logger


def conlleval(label_predict, label_path, metric_path):
    """
    Write metrics to file.

    :param label_predict:
    :param label_path:
    :param metric_path:
    :return:
    """
    with open(metric_path, "w", encoding="utf-8") as fw:
        metrics = metric_calculate(label_predict, label_path)
        fw.writelines(metrics)

    return metrics


def metric_calculate(label_predict, label_path):
    """
    Format metric matrix.

    :param label_predict: result list, contain char,true tag,predict tag
    :return:summary of percision,recall,f1
    """
    metrics = []
    keys = list(tag2label.keys())
    fact = {key: 0 for key in keys}
    predict = {key: 0 for key in keys}
    vaild = {key: 0 for key in keys}
    with open(label_path, "w", encoding="utf-8") as fw:
        line = []
        for sent_result in label_predict:
            for char, tag, tag_ in sent_result:
                for key in keys:
                    if key == tag:
                        fact[key] += 1
                    if key == tag_:
                        predict[key] += 1
                    if key == tag_ and tag_ == tag:
                            vaild[key] += 1
                line.append("{} {} {}\n".format(char, tag, tag_))
            line.append("\n")
        fw.writelines(line)

    labels = [label.split("-")[1] for label in keys if "-" in label]
    r, p, f1 = caculate_param(fact, predict, vaild, labels)
    metrics.append("{}: precision:  {:.2%}; recall:  {:.2%}; F1:  {}\n".format("all",p,r,f1))

    for label in set(labels):
        r, p, f1 = caculate_param(fact, predict, vaild, [label])
        metrics.append("{}: precision:  {:.2%}; recall:  {:.2%}; F1:  {}\n".format(label, p, r, f1))
    return metrics


def caculate_param(fact, predict, vaild, labels):
    """
    Caculate metric matrix.

    :param fact: true label num
    :param predict: model predict label num
    :param vaild: correct label num
    :param labels: label list
    :return: recall, percision, F1 score
    """
    fact_num = 0
    predict_num = 0
    vaild_num = 0
    for label in labels:
        if 'O' != label:
            fact_num += fact["B-%s" % label] + fact["I-%s" % label]
            predict_num += predict["B-%s" % label] + predict["I-%s" % label]
            vaild_num += vaild["B-%s" % label] + vaild["I-%s" % label]
        else:
            fact_num += fact[label]
            predict_num += predict[label]
            vaild_num += vaild[label]

    if fact_num == 0:
        recall = 0
    else:
        recall = float(vaild_num / fact_num)
    if predict_num == 0:
        percision = 0
    else:
        percision = float(vaild_num / predict_num)
    if percision == 0 and recall == 0:
        f1 = 0
    else:
        f1 = 2 * percision * recall / (percision + recall)
    return recall, percision, f1
