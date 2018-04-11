import sys, os
import re, csv

import tensorflow as tf
from konlpy.tag import Mecab

from utils.w2v import Word2Vec
from utils.clustering import to_sentences, clear_str, parse

flags = tf.app.flags
flags.DEFINE_string('data', None, 'Path to training csv')
flags.DEFINE_string('model_path', None, 'Path to save model files')
flags.DEFINE_boolean('skipgram', False, 'word2vec params')
flags.DEFINE_integer('unit', 100, 'word2vec params')
flags.DEFINE_integer('window', 5, 'word2vec params')
flags.DEFINE_integer('max_vocab', 500000, 'word2vec params')
flags.DEFINE_boolean('h_softmax', False, 'word2vec params')
flags.DEFINE_integer('batch', 1000, 'word2vec params')
FLAGS = flags.FLAGS

FLAGS.skipgram = int(FLAGS.skipgram)
FLAGS.h_softmax = int(FLAGS.h_softmax)


def extract_docs(csv_path):
    mecab = Mecab()

    sts, labels, tags = [], [], []
    with open(csv_path, encoding='utf8') as f:
        reader = csv.reader(f, 
                delimiter='|', 
                escapechar=':', 
                quoting=csv.QUOTE_NONE, 
                skipinitialspace=True)

        for row in reader:
            doc = clear_str(row[1])
            for st in to_sentences(doc):
                morps = parse(st, mecab)
                if not morps: continue

                sts.append(morps)
                labels.append(row[0])
                tags.append(row[1][:50])
    return sts, labels, tags


def main(_):
    if not FLAGS.data or not os.path.exists(FLAGS.data):
        tf.logging.fatal('Flag --data must be set.')
        sys.exit(1)

    if not FLAGS.model_path:
        tf.logging.fatal('Flag --model_path must be set.')
        sys.exit(1)

    if not os.path.exists(FLAGS.model_path):
        os.makedirs(FLAGS.model_path)

    tf.logging.set_verbosity(tf.logging.INFO)

    w2v_file = os.path.join(FLAGS.model_path, 'w2v.pkl')
    w2v = Word2Vec(
        model_file=w2v_file,
        skipgram=FLAGS.skipgram,
        unit=FLAGS.unit,
        window=FLAGS.window,
        max_vocab=FLAGS.max_vocab,
        h_softmax=FLAGS.h_softmax,
        batch=FLAGS.batch)

    tf.logging.info('parsing train csv...')
    sentences, labels, _ = extract_docs(FLAGS.data)

    tf.logging.info('training...')
    w2v.fit(sentences)
    tf.logging.info('# of trained words: %d' % len(w2v.words()))

    w2v.save()

    
if __name__ == '__main__':
    tf.app.run()
