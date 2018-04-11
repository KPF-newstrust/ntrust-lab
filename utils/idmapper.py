import os, sys
import numpy as np
import pickle

import tensorflow as tf


class IdMapper:
    def __init__(self, model_file, vocab_size=10000): 
        self.model_file = model_file
        self.dic = self._load()
        self.vocab_size=vocab_size

    def _load(self):
        if not os.path.exists(self.model_file):
            return {}
        with open(self.model_file, 'rb') as f:
            return pickle.load(f)

    def _save(self):
        with open(self.model_file, 'wb') as f:
            pickle.dump(self.dic, f, 
                    protocol=pickle.HIGHEST_PROTOCOL)

    def _transform(self, item):
        if item not in self.dic:
            max_id = len(self.dic) if len(self.dic) < self.vocab_size else 0
            self.dic[item] = max_id
        return self.dic[item]

    def fit_transform(self, sentences, maxlen=100):
        y = []
        for sentence in sentences:
            words = []
            for word in sentence[:maxlen]:
                words.append(self._transform(word))
            words += [0] * (maxlen - len(words))
            y.append(words)

        self._save()
        return y

