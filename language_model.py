#!/bin/env python

import argparse
from itertools import product
import math
import nltk
from pathlib import Path
import pandas as pd
import scipy.stats as stats

from utils import *


def load_data(data_dir):

    train_path = data_dir.joinpath('train.txt').absolute().as_posix()
    valid_path  = data_dir.joinpath('valid.txt').absolute().as_posix()
    test_path  = data_dir.joinpath('test.txt').absolute().as_posix()

    with open(train_path, 'r', encoding='utf-8') as f:
        train = [l.strip() for l in f.readlines()]
    with open(valid_path, 'r', encoding='utf-8') as f:
        valid = [l.strip() for l in f.readlines()]
    with open(test_path, 'r', encoding='utf-8') as f:
        test = [l.strip() for l in f.readlines()]
    return train, valid, test


class LanguageModel(object):

    def __init__(self, train_data, valid_data, n, method):
        self.n = n
        self.method = method
        self.train_tokens = preprocess(train_data, n)
        self.valid_tokens = preprocess(valid_data, n)
        self.vocab = nltk.FreqDist(self.train_tokens)
        
        # combine training set and validation set for laplace smoothing model
        self.lap_train_tokens = preprocess(train_data+valid_data, n)
        self.lap_vocab = nltk.FreqDist(self.lap_train_tokens)
        
        self.sep_case = tuple([EOS]*(self.n-1) + [BOS])
        
        if method:
            self.model  = self._create_model()

    def _create_model(self):
        ''' Choose the smoothing model '''
        if self.method == 'l':
            return self.lap_smooth()
        elif self.method == 'h':
            return self.heldout_smooth()
        elif self.method == 'c':
            return self.crossval_smooth()
        else:
            # no smoothing
            n_grams = nltk.ngrams(self.train_tokens, self.n)
            n_vocab = nltk.FreqDist(n_grams)
            N = sum(self.vocab.values())            
            model = { n_gram: count/N for n_gram, count in n_vocab.items() }

            return model
        
    def _reverse_mapping(self, orginal_dict):
        ''' Reverse the key, value mapping of a given dictionary '''
        reversed_dict = {}
        for key, value in orginal_dict.items():
            try:
                reversed_dict[value].append(key)
            except KeyError:
                reversed_dict[value] = [key]
        return reversed_dict
        
    def lap_smooth(self, laplace=1):
        ''' Laplace's law smoothing model, adding-one smoothing by default '''
        n_grams = list(nltk.ngrams(self.lap_train_tokens, self.n))
        n_vocab = nltk.FreqDist(n_grams)
        
        # manually handle the sentence interval
        del n_vocab[self.sep_case]
        
        # handle the known cases
        smoother = len(n_grams) + laplace * (len(self.lap_vocab)**self.n)
        smoothed_model = { n_gram: (count + laplace) / smoother for n_gram, count in n_vocab.items() }
        
        # handle the unknown cases
        smoothed_model["unknown"] = laplace / smoother
        
        # manually set the probability of sentence interval to 1
        smoothed_model[self.sep_case] = 1.
        
        return smoothed_model
    
    def heldout_smooth(self):
        ''' Held-out smoothing model '''
        # preprare training set
        n_grams = list(nltk.ngrams(self.train_tokens, self.n))
        n_vocab = nltk.FreqDist(n_grams)
        
        # prepare held-out set
        valid_n_grams = list(nltk.ngrams(self.valid_tokens, self.n))
        valid_n_vocab = nltk.FreqDist(valid_n_grams)
        
        # manually handle the sentence interval
        del n_vocab[self.sep_case]
        
        # count n-grams that appeared r times
        freq_vocab = self._reverse_mapping(n_vocab)
        
        # count T_r, the total number of times that all n-grams that appeared r times in a held-out dataset
        ho_prob = {}
        for key, value in freq_vocab.items():
            Tr = sum([valid_n_vocab[ngram] for ngram in value])
            # T_r / N_r
            ho_prob[key] = Tr/len(value)
        
        # handle the known cases
        smoother = len(n_grams)
        smoothed_model = { n_gram: ho_prob[count] / smoother for n_gram, count in n_vocab.items() }     
        
        # handle the unknown cases
        known = n_vocab.keys() 
        T0 = sum([value if key not in known else 0 for key, value in valid_n_vocab.items()])
        smoothed_model["unknown"] = T0 / ((len(self.vocab)**self.n - len(n_vocab)) * smoother)
        
        # manually set the probability of sentence interval to 1
        smoothed_model[self.sep_case] = 1.
        
        return smoothed_model
        
    def crossval_smooth(self):
        ''' Cross-validation smoothing model '''
        # preprare training set (set 0)
        train_n_grams = list(nltk.ngrams(self.train_tokens, self.n))
        train_n_vocab = nltk.FreqDist(train_n_grams) 
        
        # preprare validation set (set 1)
        valid_n_grams = list(nltk.ngrams(self.valid_tokens, self.n))
        valid_n_vocab = nltk.FreqDist(valid_n_grams)
        
        # manually handle the sentence interval
        del train_n_vocab[self.sep_case]
        del valid_n_vocab[self.sep_case]
        
        # count n-grams that appeared r times
        train_freq_vocab = self._reverse_mapping(train_n_vocab)
        valid_freq_vocab = self._reverse_mapping(valid_n_vocab)
        
        # count T_r, the total number of times that all n-grams that appeared r times in a held-out dataset
        Tr_0 = {key: sum([valid_n_vocab[ngram] for ngram in value]) for key, value in train_freq_vocab.items()}
        Tr_1 = {key: sum([train_n_vocab[ngram] for ngram in value]) for key, value in valid_freq_vocab.items()}
         
        known_0 = train_n_vocab.keys()
        T0_0 = sum([value if key not in known_0 else 0 for key, value in valid_n_vocab.items()])
        known_1 = valid_n_vocab.keys()
        T0_1 = sum([value if key not in known_1 else 0 for key, value in train_n_vocab.items()])
        
        N0_0 = len(self.vocab)**self.n - len(train_n_grams)
        N0_1 = len(self.vocab)**self.n - len(valid_n_grams)
        
        # handle the known cases
        union = train_n_vocab.keys() | valid_n_vocab.keys()
        smoother = (len(train_n_grams) + len(valid_n_grams)) / 2
        smoothed_model = {}
        for ngram in union:
            if ngram in train_n_vocab.keys():
                r_0 = train_n_vocab[ngram]
                Tr_01 = Tr_0[r_0]
            else:
                # ngram not appeared in set 0
                r_0 = 0
                Tr_01 = T0_0           
            if ngram in valid_n_vocab.keys():
                r_1 = valid_n_vocab[ngram]
                Tr_10 = Tr_1[r_1]
            else:
                # ngram not appeared in set 1
                r_1 = 0
                Tr_10 = T0_1

            Nr_0 = len(train_freq_vocab[r_0]) if r_0 != 0 else N0_0
            Nr_1 = len(valid_freq_vocab[r_1]) if r_1 != 0 else N0_1
            smoothed_model[ngram] = (Tr_01 + Tr_10) / ((Nr_0 + Nr_1) * smoother)
        
        # handle the unknown cases
        smoothed_model["unknown"] = (T0_0 + T0_1) / ((N0_0 + N0_1) * smoother)
        
        # manually set the probability of sentence interval to 1
        smoothed_model[self.sep_case] = 1.

        return smoothed_model
        
        
    def perplexity(self, test_data, model=None):
        ''' Compute model's perplexity over a given test corpus '''
        if model is not None:
            self.model = model
        test_tokens = preprocess(test_data, self.n)
        test_ngrams = nltk.ngrams(test_tokens, self.n)
        N = len(test_tokens)

        probabilities = [self.model["unknown"] if ngram not in self.model.keys() else self.model[ngram] for ngram in test_ngrams]

        return math.exp((-1/N) * sum(map(math.log, probabilities)))

    def _best_candidate(self, prev, i, without=[]):
        ''' Return the candidate word with highest ngram probability '''
        blacklist  = ["<UNK>"] + without
        candidates = ((ngram[-1],prob) for ngram,prob in self.model.items() if ngram[:-1]==prev)
        candidates = filter(lambda candidate: candidate[0] not in blacklist, candidates)
        candidates = sorted(candidates, key=lambda candidate: candidate[1], reverse=True)
        if len(candidates) == 0:
            return ("<EOS>", 1)
        else:
            return candidates[0 if prev != () and prev[-1] != "<BOS>" else i]
     
    def generate_sentences(self, num, min_len=12, max_len=24):
        ''' Generate sentences '''
        for i in range(num):
            sent, prob = ["<BOS>"] * max(1, self.n-1), 1
            while sent[-1] != "<EOS>":
                prev = () if self.n == 1 else tuple(sent[-(self.n-1):])
                blacklist = sent + (["<EOS>"] if len(sent) < min_len else [])
                print(prev, i, blacklist)
                next_token, next_prob = self._best_candidate(prev, i, without=blacklist)
                sent.append(next_token)
                prob *= next_prob
                
                if len(sent) >= max_len:
                    sent.append("<EOS>")

            yield ' '.join(sent), -1/math.log(prob)

def spearman_corr(model_l, model_h, model_c):
    ''' Compute the Spearman's rank correlation coefficient among three smoothing models '''
    intersect = (model_l.keys() & model_h.keys()) & model_c.keys()
 
    dist_l = {}
    dist_h = {}
    dist_c = {}
    
    for ngram in intersect:
        dist_l[ngram] = model_l[ngram]
        dist_h[ngram] = model_h[ngram]
        dist_c[ngram] = model_c[ngram]
        
    sorted_l = {k: v for k, v in sorted(dist_l.items(), key=lambda item: item[1], reverse=True)}
    sorted_h = {k: v for k, v in sorted(dist_h.items(), key=lambda item: item[1], reverse=True)}
    sorted_c = {k: v for k, v in sorted(dist_c.items(), key=lambda item: item[1], reverse=True)}     
    
    cor_lh = stats.spearmanr(list(sorted_l.values()), list(sorted_h.values()))
    cor_lc = stats.spearmanr(list(sorted_l.values()), list(sorted_c.values()))
    cor_hc = stats.spearmanr(list(sorted_h.values()), list(sorted_c.values()))
    
    print(cor_lh, cor_lc, cor_hc)
    
    rank_l = {key: rank for rank, key in enumerate(sorted(set(dist_l.values()), reverse=True), 1)}
    rank_l = {k: rank_l[v] for k,v in dist_l.items()}
    rank_h = {key: rank for rank, key in enumerate(sorted(set(dist_h.values()), reverse=True), 1)}
    rank_h = {k: rank_h[v] for k,v in dist_h.items()}
    rank_c = {key: rank for rank, key in enumerate(sorted(set(dist_c.values()), reverse=True), 1)}
    rank_c = {k: rank_c[v] for k,v in dist_c.items()}
    
    sep_ngram = {}
    for ngram in intersect:
        var = stats.variation([rank_l[ngram], rank_h[ngram], rank_c[ngram]])
        if var > 0.8:
            sep_ngram[ngram] = {'rank_l': rank_l[ngram], 'rank_h': rank_h[ngram], 'rank_c': rank_c[ngram] }
    return sep_ngram, cor_lh, cor_lc, cor_hc

def candidate_dist(sent, n, model):
    ''' Return the probability distribution of next word candidate'''
    sent = sent.split()
    gold = sent[-1]
    sent = sent[:-1]
    
    prev = () if n == 1 else tuple(sent[-(n-1):])
    candidates = ((ngram[-1],prob) for ngram,prob in model.items() if ngram[:-1]==prev)
    candidates = sorted(candidates, key=lambda candidate: candidate[1], reverse=True)
    if len(candidates) == 0:
        return ("<EOS>", 1)
    else:
        print(candidates[:10])
        for rank, cand in enumerate(candidates,1):
            if list(cand)[0] == gold:
                print("预测概率：", cand, "排序：", rank, "\n")
                return candidates
        print("Not Found")
        return candidates
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.data = "./corpus/"
    args.n = 3
    args.method = None
    args.num = 5

    '''load and prepare train/test data'''
    data_path = Path(args.data)
    train, valid, test = load_data(data_path)
    
    '''prepare the language model'''
    print("Loading {}-gram model...".format(args.n))
    lm = LanguageModel(train, valid, args.n, method=args.method)
    print("Vocabulary size: {}".format(len(lm.vocab)))

    '''compute different smoothing models'''
    l_model = lm.lap_smooth()
    h_model = lm.heldout_smooth()
    c_model = lm.crossval_smooth()
    
    '''compute Spearman's rank correlation coefficient'''
    sep_ngram, _, _, _ = spearman_corr(l_model, h_model, c_model)
    
    '''case study'''
    cases_pred = ["芜湖 的 风景 给 他 留下 了 深刻 的 印象", "扶贫 开发 工作 取得 很 大 反响", "我们 将 和 台湾 同胞 携手 合作 ， 共同 谱写 两岸 关系 的 新篇章 "]
    cases_sent = ["芜湖 的 风景 给 他 留下 了 深刻 的 印象 。", "扶贫 开发 工作 取得 很 大 反响 。", "我们 将 和 台湾 同胞 携手 合作 ， 共同 谱写 两岸 关系 的 新篇章 。"]
    for case_pred, case_sent in zip(cases_pred, cases_sent):
        for model in [l_model, h_model, c_model]:
            print("--- Case Analysis ---")
            
            candidate_dist(case_pred, args.n, model)
            print("--- Case Perplexity ---")
            print("Model perplexity: {:.3f}\n".format(lm.perplexity([case_sent], model)))
    
    '''compute model perplexity'''
    for model in [l_model, h_model, c_model]:      
        # compute the language model perplexity over the test corpus
        perplexity = lm.perplexity(test, model)
        print("--- Overall Perplexity ---")
        print("Model perplexity: {:.3f}".format(perplexity))


#    print("Generating sentences...")
#    for sentence, prob in lm.generate_sentences(args.num):
#        print("{} ({:.5f})".format(sentence, prob))