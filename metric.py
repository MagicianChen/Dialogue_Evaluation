from nltk.translate.bleu_score import corpus_bleu
from nltk.collocations import BigramCollocationFinder
from nltk.probability import FreqDist
import argparse
import codecs
import numpy as np
import math
import os, re
import ipdb
import fasttext


def word2vec(word_list,w2v):
    vectors = []
    for word in word_list:
        vectors.append(w2v[word])
    return np.stack(vectors)


def cal_Distinct(corpus):
    bigram_finder = BigramCollocationFinder.from_words(corpus)
    dist_2 = len(bigram_finder.ngram_fd) / bigram_finder.N

    dist = FreqDist(corpus)
    dist_1 = len(dist) / len(corpus)

    return dist_1, dist_2


def cal_vector_extrema(context1, context2,w2v):
    vec1 = word2vec(context1,w2v)
    vec2 = word2vec(context2,w2v)
    vec1 = np.max(vec1, axis=0)
    vec2 = np.max(vec2, axis=0)
    assert len(vec1) == len(vec2)
    zero_list = np.zeros(len(vec1))
    if vec1.all() == zero_list.all() or vec2.all() == zero_list.all():
        if vec_x.all() == vec_y.all():
            return float(1)
        else:
            return float(0)
    res = np.array([[vec1[i] * vec2[i], vec1[i] * vec1[i], vec2[i] * vec2[i]] for i in range(len(vec1))])
    ext = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))
    return ext


def cal_embedding_average(context1, context2,w2v):
    vec1 = word2vec(context1, w2v)
    vec2 = word2vec(context2, w2v)
    
    vec1 = np.array([0 for _ in range(len(vec1[0]))])
    for x in vec1:
        x = np.array(x)
        vec1 = np.add(x, vec1)
    vec1 = vec1 / math.sqrt(sum(np.square(vec1)))
    
    vec2 = np.array([0 for _ in range(len(vec2[0]))])
    for y in vec2:
        y = np.array(y)
        vec2 = np.add(y, vec2)
    vec2 = vec2 / math.sqrt(sum(np.square(vec2)))
    
    assert len(vec1) == len(vec2)
    zero_list = np.array([0 for _ in range(len(vec1))])
    if vec.all() == zero_list.all() or vec2.all() == zero_list.all():
        if vec_x.all() == vec_y.all():
            return float(1)
        else:
            return float(0)
    res = np.array([[vec1[i] * vec2[i], vec1[i] * vec1[i], vec2[i] * vec2[i]] for i in range(len(vec1))])
    avg = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))
    return avg


def cal_greedy_matching(context1, context2,w2v):
    vec1 = word2vec(context1, w2v)
    vec2 = word2vec(context2, w2v)
    
    cosine = []
    sum_1 = 0
    for x in vec1:
        for y in vec2:
            assert len(x) == len(y), "len(x_v) != len(y_v)"
            zero_list = np.zeros(len(x))

            if x.all() == zero_list.all() or y.all() == zero_list.all():
                if x.all() == y.all():
                    cos = float(1)
                else:
                    cos = float(0)
            else:
                res = np.array([[x[i] * y[i], x[i] * x[i], y[i] * y[i]] for i in range(len(x))])
                cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))

            cosine.append(cos)
        if cosine:
            sum_1 += max(cosine)
            cosine = []

    sum_1 = sum_1 / len(vec1)

    cosine = []
    sum_2 = 0
    for y in vec2:

        for x in vec1:
            assert len(x) == len(y)
            zero_list = np.zeros(len(y))

            if x.all() == zero_list.all() or y.all() == zero_list.all():
                if (x == y).all():
                    cos = float(1)
                else:
                    cos = float(0)
            else:
                res = np.array([[x[i] * y[i], x[i] * x[i], y[i] * y[i]] for i in range(len(x))])
                cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))

            cosine.append(cos)

        if cosine:
            sum_2 += max(cosine)
            cosine = []

    sum_2 = sum_2 / len(vec2)
    gre = (sum_1 + sum_2) / 2
    return gre


def cal_coherence(resp_words, source_words, w2v, stopwords):
    resp_words = [i for i in resp_words if i not in stopwords]
    source_words = [i for i in source_words if i not in stopwords]
    avg_source = cal_embedding_average(source_words, resp_words, w2v)

    return avg_source



if __name__ == "__main__":
    path = '' # file_path
    with open(path,'r',encoding='utf-8') as f:
        source, ref, tgt = [], [], []
        for idx, line in enumerate(f.readlines()):
            ref.append(line.split('\t')[2].strip().split())
            tgt.append(line.split('\t')[1].split())
            source.append(line.split('\t')[0].split())
    # ----------------------------------#
    # load your stopword file here for coherence calculation

    # ---------------------------------#

    # load the word embedding model
    w2v = fasttext.load_model('')  # change the word embedding model

    candidates, references, prediction = [], []
    greedy, extra, avg, coherence = [], [], [], []


    for line1, line2 in zip(tgt, ref):
        # calculate embedding-based score for each sample
        greedy.append(cal_greedy_matching(line2, line1, w2v))
        extra.append(cal_vector_extrema(line2,line1, w2v))
        avg.append(cal_embedding_average(line2,line1,w2v))
        coherence.append(cal_coherence(line1, line3, w2v, zh_stop_words)) # change stopword here
        candidates.extend(line1)
        if len(line1) > 3 and len(line2) > 3:
            prediction.append(line1)
            references.append([line2])

    # Distinct-1, Distinct-2
    distinct_1, distinct_2 = cal_Distinct(candidates)
    # BLEU
    Bleu_score = corpus_bleu(references, prediction)

    output_path = ''  # output file
    with open(output_path,'w',encoding='utf-8') as f:
        # need to average the embedding metrics
        f.write(
           'Bleu: ' + str(Bleu_score) + '\tdis-1/2: ' + str(distinct_1) + '/' + str(distinct_2) + '\tgreedy: ' + str(
               np.mean(np.array(greedy))) + '\textra: ' + str(np.mean(np.array(extra)))+'\tavg: '+ str(np.mean(np.array(avg)))
           + '\tcoh: ' + str(np.mean(np.array(coherence))))

