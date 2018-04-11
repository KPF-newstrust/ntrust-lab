import sys, os
import re, csv

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np
import pickle
from sklearn.cluster import DBSCAN
from sklearn import metrics
from konlpy.tag import Mecab


def visualize(sess, varname, X, meta_file):
    model_path = os.path.dirname(meta_file)

    tf.logging.info('visualize count {}'.format(X.shape))
    Ws = tf.Variable(X, trainable=True, name=varname)
    sess.run(Ws.initializer)

    # save weights
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(model_path, '%s.ckpt' % varname))

    # associate meta & embeddings
    conf = projector.ProjectorConfig()
    embedding = conf.embeddings.add()
    embedding.tensor_name = varname
    embedding.metadata_path = meta_file

    writer = tf.summary.FileWriter(model_path)
    projector.visualize_embeddings(writer, conf)

    tf.logging.info('Run `tensorboard --logdir={}` to run visualize result on tensorboard'.format(model_path))


class DBScan:
    def __init__(self, 
        model_path=None, 
        unit=100,
        eps=0.5): 

        if model_path:
            self.model_file = os.path.join(model_path, 'dbscan.model')
            self.meta_file = os.path.join(model_path, 'dbscan.meta')
            self.X_file = os.path.join(model_path, 'dbscan_x.npy')

        if self.model_file and os.path.exists(self.model_file):
            try:
                with open(self.model_file, 'rb') as f:
                    self.model = pickle.load(f)
            except:
                tf.logging.info('fail to load dbscan: {}'.format(sys.exc_info()))
            tf.logging.info('dbscan loaded')
        else:
            self.model = DBSCAN(
                eps=eps,           # neighborhood로 인정되는 최대 거리
                min_samples=2,     # core point size
                metric='euclidean',
                n_jobs=-1)

        self.X = np.zeros([0, unit], dtype=np.float32)
        try:
            if self.X_file and os.path.exists(self.X_file):
                self.X = np.load(self.X_file, mmap_mode='r+')
        except:
            tf.logging.info('fail to load X from file {} {}'.format(self.X_file, sys.exc_info()))
            

    def save(self, tags=[]):
        if self.model_file:
            try:
                with open(self.model_file, 'wb') as f:
                    pickle.dump(
                        self.model, f, protocol=pickle.HIGHEST_PROTOCOL)
            except:
                tf.logging.info(
                    'fail to save dbscan: {}'.format(sys.exc_info()))

        if self.meta_file and tags:
            with open(self.meta_file, 'ab') as f:
                for tag, label in zip(tags, self.model.labels_):
                    f.write(('[%03x] %s' % (label, tag)).encode('utf-8') + b'\n')

        if self.X_file:
            np.save(self.X_file, self.X) 

    def fit(self, X):
        self.X = np.concatenate((self.X, np.array(X)), axis=0)
        # return [nsamples]
        return self.model.fit_predict(X)

    def labels(self):
        # return: 각 sample(doc)의 label
        return self.model.labels_

    def n_clusters(self):
        L = self.labels()
        return len(set(L)) - (1 if -1 in L else 0)

    def core_samples(self):
        return self.model.core_sample_indices_

    def core_tags(self, tags):
        labels = self.model.labels_
        cores = self.model.core_sample_indices_

        clu_tags = []
        for _ in range(self.n_clusters()):
            clu_tags.append([])

        for i in cores:
            clu = labels[i]
            if clu < 0: continue

            tag = tags[i] if len(tags) > i else ''
            clu_tags[clu].append(tag)
        return clu_tags

    def eval(labels, cls_labels, X):
        nclusters = len(set(labels)) - (1 if -1 in labels else 0)
        return dict(
            n_clusters=nclusters,
            homogeneity="%0.3f" % metrics.homogeneity_score(cls_labels, labels),
            completeness="%0.3f" % metrics.completeness_score(cls_labels, labels),
            v_measure="%0.3f" % metrics.v_measure_score(cls_labels, labels),
            adjusted_rand_index="%0.3f" % metrics.adjusted_rand_score(cls_labels, labels),
            adjusted_mutual_info="%0.3f" % metrics.adjusted_mutual_info_score(cls_labels, labels),
            silhouette="%0.3f" % metrics.silhouette_score(X, labels))


SEP = re.compile(r"[\.!\?]\s+|$", re.M)

def to_sentences(doc):
    sts = []
    start_sentence=0
    for sep in SEP.finditer(doc):
        st= doc[start_sentence:sep.start(0)]
        start_sentence = sep.end(0)
        if len(st) < 10: continue
        sts.append(st)
    return sts


def clear_str(str):
    str = re.sub(r'\[[^\]]+\]', '', str)
    str = re.sub(r'\([^\)]+\)', '', str)
    str = re.sub(r'\<[^\>]+\>', '', str)
    str = re.sub(r'[^\.\?!\uAC00-\uD7AF]', ' ', str)
    str = re.sub(r'\s{2,}', ' ', str)
    return str


def parse(sentence, mecab, allowed_morps=None):
    """문장을 형태소로 변환"""

    return [pos[0]for pos in mecab.pos(sentence) \
            if not allowed_morps or pos[1] in allowed_morps]


