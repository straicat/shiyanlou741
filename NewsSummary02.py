# -*- coding: utf-8 -*-

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import math
from itertools import product, count
from string import punctuation
from heapq import nlargest

stopwords = set(stopwords.words('english') + list(punctuation))


def calc_similarity(sen1, sen2):
    cnt = 0
    for word in sen1:
        if word in sen2:
            cnt += 1
    return cnt / (math.log(len(sen1)) + math.log(len(sen2)))


def create_graph(word_sent):
    num = len(word_sent)
    board = [[0.0 for _ in range(num)] for _ in range(num)]
    for i, j in product(range(num), repeat=2):
        if i != j:
            board[i][j] = calc_similarity(word_sent[i], word_sent[j])
    return board


def different(scores, old_scores):
    for i in range(len(scores)):
        if math.fabs(scores[i] - old_scores[i]) >= 0.0001:
            return True
    return False


def calc_score(weight_graph, scores, i):
    length = len(weight_graph)
    d = 0.85
    added_score = 0.0
    for j in range(length):
        frac = 0.0
        deno = 0.0
        frac = weight_graph[j][i] * scores[j]
        for k in range(length):
            deno += weight_graph[j][k]
        added_score += frac / deno
    return (1-d) + d * added_score


def  weighted_pagerank(weight_graph):
    scores = [0.5 for _ in range(len(weight_graph))]
    old_scores = [0.0 for _ in range(len(weight_graph))]
    while different(scores, old_scores):
        for i in range(len(weight_graph)):
            old_scores[i] = scores[i]
        for i in range(len(weight_graph)):
            scores[i] = calc_score(weight_graph, scores, i)
    return scores


def summarize(text, n):
    sents = sent_tokenize(text)
    word_sent = [word_tokenize(s.lower()) for s in sents]
    for i in range(len(word_sent)):
        for word in word_sent[i]:
            if word in stopwords:
                word_sent[i].remove(word)
    simi_graph = create_graph(word_sent)
    scores = weighted_pagerank(simi_graph)
    sent_selected = nlargest(n, zip(scores, count()))
    sent_idx = []
    for i in range(n):
        sent_idx.append(sent_selected[i][1])
    return [sents[i] for i in sent_idx]


if __name__ == '__main__':
    with open('news.txt') as f:
        text = f.read().replace('\n', '')
    print(summarize(text, 2))
    