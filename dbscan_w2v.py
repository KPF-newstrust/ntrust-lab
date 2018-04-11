import sys, os, logging
import re, csv

import tensorflow as tf
from konlpy.tag import Mecab

from utils.w2v import Word2Vec
from utils.clustering import DBScan, to_sentences, clear_str, parse, visualize


flags = tf.app.flags
flags.DEFINE_string('data', None, 'Path to training csv')
flags.DEFINE_string('model_path', None, 'Path to save model files')
flags.DEFINE_string('w2v_path', None, 'Word2Vec Path to load')
flags.DEFINE_boolean('skipgram', False, 'word2vec params')
flags.DEFINE_integer('unit', 100, 'word2vec params')
flags.DEFINE_integer('window', 5, 'word2vec params')
flags.DEFINE_integer('max_vocab', 500000, 'word2vec params')
flags.DEFINE_boolean('h_softmax', False, 'word2vec params')
flags.DEFINE_integer('batch', 1000, 'word2vec params')
flags.DEFINE_float('eps', 2.0, 'dbscan params')
FLAGS = flags.FLAGS

FLAGS.skipgram = int(FLAGS.skipgram)
FLAGS.h_softmax = int(FLAGS.h_softmax)


def extract_docs(csv_path):
    mecab = Mecab()

    docs, labels = [], []
    with open(csv_path, encoding='utf8') as f:
        reader = csv.reader(
            f, delimiter='|', 
            escapechar=':', 
            quoting=csv.QUOTE_NONE, 
            skipinitialspace=True)

        for row in reader:
            sts = []
            doc = clear_str(row[1])
            for st in to_sentences(doc):
                morps = parse(st, mecab)
                if not morps: continue
                sts.append(morps)

            if not sts: continue  # ignore invalid news
            docs.append(sts)
            labels.append(row[0])
    return docs, labels


def summarize_docs(docs):
    tags = []
    for doc in docs:
        ss = []
        for st in doc:
            ss += st
        tags.append(' '.join(ss)[:50])
    return tags


def main(_):
    if not FLAGS.data or not os.path.exists(FLAGS.data):
        tf.logging.fatal('Flag --data must be set.')
        sys.exit(1)

    if not FLAGS.model_path:
        tf.logging.fatal('Flag --model_path must be set.')
        sys.exit(1)

    if not os.path.exists(FLAGS.model_path):
        os.makedirs(FLAGS.model_path)

    if not FLAGS.w2v_path or not os.path.exists(FLAGS.w2v_path):
        tf.logging.fatal('Flag --w2v_path must be set.')
        sys.exit(1)

    tf.logging.set_verbosity(tf.logging.INFO)

    tf.logging.info('parsing train csv...')
    docs, cls_labels = extract_docs(FLAGS.data)

    w2v_file = os.path.join(FLAGS.w2v_path, 'w2v.pkl')
    w2v = Word2Vec(
        model_file=w2v_file,
        skipgram=FLAGS.skipgram,
        unit=FLAGS.unit,
        window=FLAGS.window,
        max_vocab=FLAGS.max_vocab,
        h_softmax=FLAGS.h_softmax,
        batch=FLAGS.batch)
    w2v.freeze()
    X = w2v.transform(docs)

    tf.logging.info('training...')
    dbscan = DBScan(
        model_path=FLAGS.model_path, 
        unit=FLAGS.unit,
        eps=FLAGS.eps)
    dbscan.fit(X)

    clu_labels = dbscan.labels()
    ev = DBScan.eval(clu_labels, cls_labels, X)
    tf.logging.info('eval {}'.format(ev) )

    tags = summarize_docs(docs)
    dbscan.save(tags=tags)

    with tf.Session() as sess:
        visualize(
            sess, 
            varname='w2v_dbscan', 
            X=dbscan.X, 
            meta_file=dbscan.meta_file)

    
if __name__ == '__main__':
    tf.app.run()
