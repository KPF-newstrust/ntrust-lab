import os, sys, re
import csv

import tensorflow as tf
import numpy as np
import pickle
from konlpy.tag import Mecab


flags = tf.app.flags
flags.DEFINE_string('train', None, 'File path to train')
flags.DEFINE_string('predict', None, 'File path to predict')
flags.DEFINE_string('test', None, 'File path to test')
flags.DEFINE_string('model_path', None, 'Path to save model files')
flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
flags.DEFINE_integer('decay_steps', 1000, 'n of steps to decay learning rate')
flags.DEFINE_integer('max_vocab', 100000, 'vocab limit')
flags.DEFINE_integer('max_words_per_doc', 100, 'n words limit for a document')
FLAGS = flags.FLAGS


TAGS = ['NNP', 'NNG', 'VV', 'VA']
MAX_VOCA=FLAGS.max_vocab
MAX_WORDS_PER_DOC=FLAGS.max_words_per_doc
CATES= {
    '정치': 1, 
    '경제': 2, 
    '사회': 3, 
    '국제': 4,
    'IT 과학': 5, 
    '문화 예술': 6, 
    '교육': 7, 
    '연예': 8,
    '스포츠': 9, 
    '라이프스타일': 10, 
    '사설·칼럼': 11, 
    '기타': 12}


class Voca:
    UNK=0
    RE = re.compile(r"\b[\uAC00-\uD7AF]{2,4}", re.UNICODE)

    def __init__(self, model_file):
        self.model_file = model_file
        self.db = self._load()

    def _load(self):
        if not os.path.exists(self.model_file):
            return {}
        with open(self.model_file, 'rb') as f:
            return pickle.load(f)

    def _save(self):
        with open(self.model_file, 'wb') as f:
            pickle.dump(self.db, f, 
                    protocol=pickle.HIGHEST_PROTOCOL)

    def fit(self, docs):
        for tokens in self.tokenizer(docs):
            for c in tokens:
                if len(self.db) >= MAX_VOCA:
                    return
                if c not in self.db:
                    self.db[c] = len(self.db)

        self._save()

    def transform(self, docs):
        _docs = []
        for tokens in self.tokenizer(docs):
            _doc = []
            for c in tokens:
                if len(_doc) >= MAX_WORDS_PER_DOC:
                    break
                if c in self.db:
                    _doc.append(self.db[c])
                else:
                    _doc.append(self.UNK)
            _doc += [self.UNK] * (MAX_WORDS_PER_DOC - len(_doc))
            _docs.append(_doc)
        return _docs

    def tokenizer(self, iterator):
        for doc in iterator:
            yield self.RE.findall(doc)


def read_csv(csv_file):
    features, labels = [], []
    with open(csv_file, encoding='utf8') as f:
        reader = csv.reader(
            f, 
            delimiter='|', 
            escapechar=':', 
            quoting=csv.QUOTE_NONE, 
            skipinitialspace=True)

        for row in reader:
            if len(row) != 2: continue
            try:
                labels.append(int(row[0]))
                features.append(row[1])
            except:
                pass

    tf.logging.info('read docs: %d' % len(features))
    return features, labels


def build_model_fn(
    embedding_unit=100,
    rnn_units=32,
    max_words=MAX_WORDS_PER_DOC,
    max_voca=MAX_VOCA,
    max_cates=len(CATES),
    hidden_nodes=1024):

    def model_fn(features, labels, mode):
        def logits():
            x = features['x'] # [batch, maxchars_per_doc]

            x = tf.contrib.layers.embed_sequence(
                x, vocab_size=max_voca, embed_dim=embedding_unit)
            x = tf.unstack(x, axis=1)

            cell_fw = tf.nn.rnn_cell.GRUCell(rnn_units)
            cell_bw = tf.nn.rnn_cell.GRUCell(rnn_units)
            x, _, _ = tf.nn.static_bidirectional_rnn(
                cell_fw, cell_bw, x, dtype=tf.float32) 

            x = tf.stack(x, axis=1)
            x = tf.reshape(x, [-1, max_words * rnn_units * 2])
            # skip dropout

            x = tf.layers.dense(x, max_cates)
            return x

        def loss(logits):
            one_hot = tf.one_hot(labels, max_cates, 1., 0.) 
            return tf.losses.softmax_cross_entropy(
                    onehot_labels=one_hot,
                    logits=logits)

        def train_op(loss):
            decay_rate = tf.train.exponential_decay(
                FLAGS.learning_rate, 
                tf.train.get_global_step(), 
                FLAGS.decay_steps, 
                0.1, 
                staircase=True)
            return tf.train.AdamOptimizer(decay_rate).minimize(
                    loss, 
                    global_step=tf.train.get_global_step())

        def accu(logits):
            return tf.metrics.accuracy(
                    labels=labels, 
                    predictions=tf.argmax(logits, -1))

        logits = logits()
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={
                    'class': tf.argmax(logits, 1),
                    'prob': tf.nn.softmax(logits)
                })
        else:
            loss = loss(logits)
            train_op = train_op(loss)
            accu = accu(logits)

            return tf.estimator.EstimatorSpec(
                mode=mode, 
                loss=loss, 
                train_op=train_op,
                eval_metric_ops={'accuracy': accu})

    return model_fn


def build_input_fn(
        x, 
        labels=None, 
        voca=None,
        batch_size=50, 
        num_epochs=1, 
        shuffle=True):

    x_ = voca.transform(x)
    features = {'x': np.array(x_)}
    L = np.array(labels) if labels else None

    return tf.estimator.inputs.numpy_input_fn(
        x=features,
        y=L,
        batch_size=batch_size,
        num_epochs=num_epochs,
        shuffle=shuffle)


def train(estimator, csv_file, voca):
    docs, labels = read_csv(csv_file)

    voca.fit(docs) # preprocess
    tf.logging.info('processed voca size %d' % len(voca.db))

    input_fn = build_input_fn(
        docs, 
        labels=labels, 
        voca=voca,
        batch_size=200, 
        num_epochs=1, 
        shuffle=True)
    estimator.train(input_fn, steps=None)


def test(estimator, csv_file, voca):
    docs, labels = read_csv(csv_file)

    input_fn = build_input_fn(
        docs, 
        labels=labels, 
        voca=voca,
        batch_size=len(docs), 
        num_epochs=1, 
        shuffle=False)
    estimator.evaluate(input_fn, steps=None)


def predict(estimator, data_file, voca):
    with open(data_file) as f:
        contents = f.read()

    mecab = Mecab()
    morps = mecab.pos(contents)
    morps = [morp[0] for morp in morps if morp[1] in TAGS]

    input_fn = build_input_fn(
        [' '.join(morps)], 
        labels=None, 
        voca=voca,
        batch_size=1, 
        num_epochs=1, 
        shuffle=False)
    predict = estimator.predict(input_fn)

    cate_names = {v:k for k,v in CATES.items()}

    def second_cls(probs):
        tup = [(i, prob) for i, prob in enumerate(probs)]
        tup = sorted(tup, key=lambda x: x[1], reverse=True)
        return tup[1]

    for i, p in enumerate(predict):
        cls, probs = p['class'], p['prob']
        name, prob = cate_names[cls], probs[cls]

        cls2, prob2 = second_cls(probs)
        name2 = cate_names[cls2]
        tf.logging.info("Prediction %s: %s(%.4f), %s(%.4f)"\
            % (i+1, name, prob, name2, prob2 ))


def main(_):
    if not FLAGS.model_path:
        tf.logging.fatal('Flag --model_path must be set.')
        sys.exit(1)

    if not os.path.exists(FLAGS.model_path):
        os.makedirs(FLAGS.model_path)

    tf.logging.set_verbosity(tf.logging.INFO)

    voca_file = os.path.join(FLAGS.model_path, 'voca.pkl')
    voca = Voca(voca_file)
    model_fn = build_model_fn()

    conf = tf.estimator.RunConfig(
            model_dir=FLAGS.model_path, 
            save_checkpoints_steps=10000,
            keep_checkpoint_max=5)
    estimator = tf.estimator.Estimator(model_fn, config=conf)

    if FLAGS.train:
        train(estimator, FLAGS.train, voca=voca)

    if FLAGS.test:
        test(estimator, FLAGS.test, voca=voca)

    if FLAGS.predict:
        predict(estimator, FLAGS.predict, voca=voca)


if __name__ == '__main__':
    tf.app.run()
