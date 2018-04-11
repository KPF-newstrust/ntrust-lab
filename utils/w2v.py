import os, sys
import numpy as np
import pickle
import gensim

import tensorflow as tf


class Word2Vec:
    def __init__(self, 
        model_file=None, 
        skipgram=0,
        unit=100,
        window=5,
        max_vocab=500000,
        h_softmax=0,
        batch=1000,
        epoch=1): 
        """
        Args: 
            see: https://radimrehurek.com/gensim/models/word2vec.html
        """

        self.model_file = model_file

        if self.model_file and os.path.exists(self.model_file):
            self.model = gensim.models.Word2Vec.load(self.model_file)
            tf.logging.info('%d voca is loaded' % len(self.model.wv.vocab))
        else:
            self.model = gensim.models.Word2Vec(
                None, 
                sg=skipgram,
                size=unit,
                window=window,
                min_count=1, # 빈도 1개이상만 처리
                max_vocab_size=max_vocab, 
                hs=h_softmax,
                batch_words=batch,
                iter=epoch)

        self.wv = self.model.wv

    def fit(self, sentences):
        assert self.model, 'Model has beedn frozen'

        self.model.build_vocab(sentences,
                update=(len(self.model.wv.vocab) > 0))
        self.model.train(sentences, 
                total_examples=len(sentences), epochs=1)

    def words(self):
        return self.wv.index2word

    def freeze(self):
        del self.model

    def save(self):
        if not (self.model and self.model_file): return
        self.model.save(self.model_file)

    def accuracy(self, 
            positive=[], 
            negative=[]):

        return self.wv.most_similar(
                positive=positive, 
                negative=negative)

    def transform(self, docs):
        return [self._transform_sentences(sts) for sts in docs]

    def _transform_sentences(self, sentences):
        vecs = []
        for st in sentences:
            for w in st:
                if w not in self.wv: continue
                vecs.append(self.wv[w])
        return np.mean(vecs, axis=0)


