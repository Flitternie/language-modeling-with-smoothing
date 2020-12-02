#!/bin/env python

import nltk

BOS = "<BOS>"
SOS = BOS + " "
EOS = "<EOS>"
UNK = "<UNK>"

def add_sentence_tokens(sentences, n):
    sos = SOS * (n-1) if n > 1 else SOS
    return ['{}{} {}'.format(sos, s, EOS) for s in sentences]

def replace_singletons(tokens):
    vocab = nltk.FreqDist(tokens)
    return [token if vocab[token] > 1 else UNK for token in tokens]

def add_unk(vocab, sentences):
    added_sentences = []
    for s in sentences:
        added_sentences.append([token if token in vocab.keys() else UNK for token in s])
    return added_sentences    

def preprocess(sentences, n):
    sentences = add_sentence_tokens(sentences, n)
    tokens = ' '.join(sentences).split(' ')
    return tokens