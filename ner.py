import os, sys
import re

import tensorflow as tf
import numpy as np
from konlpy.tag import Mecab

import pickle
from utils.w2v import Word2Vec
from utils.idmapper import IdMapper


flags = tf.app.flags
flags.DEFINE_string('train', None, 'File path to train')
flags.DEFINE_string('predict', None, 'File path to predict')
flags.DEFINE_string('test', None, 'File path to test')
flags.DEFINE_string('w2v_path', None, 'Word2Vec Path to load')
flags.DEFINE_string('model_path', None, 'Path to save model files')
flags.DEFINE_boolean('skipgram', True, 'word2vec params')
flags.DEFINE_integer('unit', 100, 'word2vec params')
flags.DEFINE_integer('window', 10, 'word2vec params')
flags.DEFINE_integer('max_vocab', 500000, 'word2vec params')
flags.DEFINE_boolean('h_softmax', True, 'word2vec params')
flags.DEFINE_integer('batch', 1000, 'word2vec params')
flags.DEFINE_integer('tag_unit', 10, 'embedding unit for tags')
flags.DEFINE_float('learning_rate', 1e-4, 'learning rate')
flags.DEFINE_integer('decay_steps', 4500, 'n of steps to decay learning rate')
flags.DEFINE_boolean('no_crf', False, 'use softmax instead of crf')
FLAGS = flags.FLAGS

FLAGS.skipgram = int(FLAGS.skipgram)
FLAGS.h_softmax = int(FLAGS.h_softmax)


ALNUMS = [*' !.?\'\"-()[]0123456789abcdefghijklmnopqrstuvwxyz']
N_CHO=19
N_JUNG=21
N_JONG=28
MAX_CHARS=len(ALNUMS) + N_CHO + N_JUNG + N_JONG
MAX_CHARS_PER_WORD=8
MAX_WORDS_PER_SENTENCE=60
MAX_TAGS=44  # Mecab's # of tags

ALLOWED_NE=['O', 'PS', 'OG', 'LC']

rLINE_SEP = re.compile(r"([\.!\?]\s+)|$")
rCHARS  = re.compile(r"[0-9a-z'\"\(\)\[\]\uAC00-\uD7AF]", re.UNICODE)
rNA_CHARS = re.compile(r"[^0-9a-z'\"\(\)\[\]\uAC00-\uD7AF]", re.UNICODE)
rSPCS = re.compile(r"\s{2,}", re.UNICODE)

def clear_str(string):
    string = rNA_CHARS.sub(" ", string)
    string = rSPCS.sub(" ", string)
    return string.strip().lower()


def sentences_to_w2v_id(sentences, w2v):
    sentences = [[clear_str(w) for w in st] for st in sentences]

    w2v.fit(sentences)
    ids = [[w2v.wv[w] for w in st] for st in sentences]

    ids = []
    slens = []
    for st in sentences:
        _st = []
        for w in st[:MAX_WORDS_PER_SENTENCE]:
            _st.append(w2v.wv[w])

        slens.append(min(len(_st), MAX_WORDS_PER_SENTENCE))
        _st += [[0.]*FLAGS.unit] *\
                (MAX_WORDS_PER_SENTENCE - len(_st))
        ids.append(_st)
    return ids, slens


def sentences_to_id(sentences):
    _sentences = []
    _slens = []
    for sentence in sentences:
        _sentence = []
        for word in sentence:
            if len(_sentence) >= MAX_WORDS_PER_SENTENCE:
                break

            _word = []
            for ch in rCHARS.findall(clear_str(word)):
                if len(_word) > MAX_CHARS_PER_WORD-3:
                    break

                if ch in ALNUMS:
                    _word.append(ALNUMS.index(ch))
                    continue

                c = ord(ch) - 0xac00
                _word.append( int((c / N_JONG) / N_JUNG) + len(ALNUMS))
                _word.append( int((c / N_JONG) % N_JUNG) + len(ALNUMS) + N_CHO )
                _word.append( int(c % N_JONG) + len(ALNUMS) + N_CHO + N_JUNG - 1 )
            # add padding
            _word += [0] * (MAX_CHARS_PER_WORD - len(_word))
            _sentence.append(_word)

        # add padding
        _slens.append(min(len(_sentence), MAX_WORDS_PER_SENTENCE))
        _sentence += [[0]*MAX_CHARS_PER_WORD] *\
                (MAX_WORDS_PER_SENTENCE - len(_sentence))
        _sentences.append(_sentence)
    return _sentences, _slens


def labels_to_id(labels):
    ids = []
    for sentence in labels:
        word_ids = []
        for ne in sentence[:MAX_WORDS_PER_SENTENCE]:
            word_ids.append(NE_to_id(ne))
        word_ids += [0] * (MAX_WORDS_PER_SENTENCE - len(word_ids))
        ids.append(word_ids)
    return ids


def build_model_fn(
    max_chars=71,
    rnn_units=[20, 32],
    max_chars_per_word=100,
    max_words_per_sentence=5,
    w2v_unit=100,
    tag_unit=10,
    max_cates=20):

    def model_fn(features, labels, mode):
        def logits():
            x = features['x'] # [batch, maxwords, maxchars]
            seq_lens = features['seq_lens']
            x = tf.one_hot(x, max_chars, 1., 0., axis=-1)
            x = tf.reshape(x, [-1, max_chars_per_word, x.shape[-1]])

            # reshape tags to concat with encoded word
            tags = features['tags'] # [batch, maxwords]
            tags = tf.contrib.layers.embed_sequence(
                tags, vocab_size=MAX_TAGS, embed_dim=tag_unit)
            tags = tf.reshape(tags, [-1, tags.shape[-1]])

            # prepare w2v words to concat
            w2vs = features['x_w']  # [batch, maxwords, w2v_unit]
            w2vs = tf.reshape(w2vs, [-1, w2vs.shape[-1]])

            # encode word
            with tf.variable_scope('char-rnn'):
                fcell = tf.nn.rnn_cell.GRUCell(rnn_units[0])
                bcell = tf.nn.rnn_cell.GRUCell(rnn_units[0])

                _, st = tf.nn.bidirectional_dynamic_rnn(
                        fcell, 
                        bcell, 
                        x, 
                        sequence_length=None,
                        dtype=tf.float32) 
                x = tf.concat([w2vs, *st, tags], axis=-1)
                x = tf.reshape(x, [-1, max_words_per_sentence, x.shape[-1]])

            # guess label
            with tf.variable_scope('word-rnn'):
                fcell2 = tf.nn.rnn_cell.GRUCell(rnn_units[1])
                bcell2 = tf.nn.rnn_cell.GRUCell(rnn_units[1])
                x, _ = tf.nn.bidirectional_dynamic_rnn(
                        fcell2, 
                        bcell2, 
                        x, 
                        sequence_length=seq_lens,
                        dtype=tf.float32) 
                x = tf.concat(x, axis=-1)
                if mode == tf.estimator.ModeKeys.TRAIN:
                    x = tf.nn.dropout(x, .5)

            x = tf.reshape(x, [-1, 2 * rnn_units[1]])
            x = tf.layers.dense(x, max_cates, activation=tf.nn.relu)
            x = tf.reshape(x, [-1, max_words_per_sentence, max_cates])
            return x

        def loss_softmax(logits):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=labels,
                    logits=logits)
            seq_lens = features['seq_lens']
            mask = tf.sequence_mask(seq_lens)
            losses = tf.boolean_mask(losses, mask)
            return tf.reduce_mean(losses)

        def loss_crf(logits):
            seq_lens = features['seq_lens']

            log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
                    logits, labels, seq_lens)  # [batch, max_words]
            return tf.reduce_mean(-log_likelihood)

        def loss(logits):
            return loss_softmax(logits) if FLAGS.no_crf else loss_crf(logits)

        def train_op(loss):
            decay_rate = tf.train.exponential_decay(
                FLAGS.learning_rate, 
                tf.train.get_global_step(), 
                FLAGS.decay_steps, 
                0.1, 
                staircase=True)
            return tf.train.AdamOptimizer(decay_rate).minimize(
                    loss, global_step=tf.train.get_global_step())

        def div(a,b):
            a = tf.cast(a, dtype=tf.float32) + 1e-7 # to prevent div by zero
            b = tf.cast(b, dtype=tf.float32) + 1e-7
            return tf.div(a,b)

        def mask_reduce(tensor, mask):
            zeros = tensor * 0
            return tf.reduce_sum(tf.where(mask, tensor, zeros), -1)

        def eval_metrics(logits):
            y = tf.argmax(logits, -1, output_type=tf.int32)

            seq_lens = features['seq_lens'] # [batch]

            mask = tf.sequence_mask(seq_lens, maxlen=max_words_per_sentence)

            positives = tf.logical_and(tf.greater(y, 0), mask)
            trues = tf.logical_and(tf.greater(labels, 0), mask)
            true_pos = tf.logical_and(tf.equal(y, labels), trues)

            total_preds = tf.reduce_sum(tf.cast(positives, tf.int32), -1)
            total_corrects = tf.reduce_sum(tf.cast(trues, tf.int32), -1)
            correct_preds = tf.reduce_sum(tf.cast(true_pos, tf.int32), -1)
            prec = div(correct_preds, total_preds)
            recall = div(correct_preds, total_corrects)
            f1 = div(2. * prec * recall, (prec + recall))
            #f1 = div(tf.multiply(2., tf.multiply(prec, recall)), tf.add(prec, recall))

            return {'precision': tf.metrics.mean(prec),
                    'recall': tf.metrics.mean(recall),
                    'f1': tf.metrics.mean(f1)}

        logits = logits()
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={
                    'class': tf.argmax(logits, -1),
                    'prob': tf.nn.softmax(logits)
                })
        else:
            loss = loss(logits)
            train_op = train_op(loss)
            metrics = eval_metrics(logits)

            return tf.estimator.EstimatorSpec(
                mode=mode, 
                loss=loss, 
                train_op=train_op,
                eval_metric_ops=eval_metrics(logits))

    return model_fn


def train(estimator, data, w2v_data, tags, seq_lens, labels):
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': np.array(data, dtype=np.int32), 
           'x_w': np.array(w2v_data, dtype=np.float32), 
           'tags': np.array(tags, dtype=np.int32),
           'seq_lens': np.array(seq_lens, dtype=np.int32)},
        y=np.array(labels, dtype=np.int32),
        batch_size=10,
        num_epochs=1,
        shuffle=True)
    estimator.train(input_fn, steps=None)


def test(estimator, data, w2v_data, tags, seq_lens, labels):
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': np.array(data, dtype=np.int32), 
           'x_w': np.array(w2v_data, dtype=np.float32), 
           'tags': np.array(tags, dtype=np.int32),
           'seq_lens': np.array(seq_lens, dtype=np.int32)},
        y=np.array(labels, dtype=np.int32),
        batch_size=10,
        num_epochs=1,
        shuffle=False)
    estimator.evaluate(input_fn, steps=None)


def predict(estimator, data, w2v_data, tags, seq_lens, sentences):
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': np.array(data, dtype=np.int32), 
           'x_w': np.array(w2v_data, dtype=np.float32), 
           'tags': np.array(tags, dtype=np.int32),
           'seq_lens': np.array(seq_lens, dtype=np.int32)},
        batch_size=1,
        num_epochs=1,
        shuffle=False)
    predict = estimator.predict(input_fn)

    for st, pred in zip(sentences, predict):
        for word, ne in zip(st, pred['class']):
            print("{}{}".format(word, ('/' + id_to_NE(ne) if ne > 0 else '')), end=' ')
        print('')


def NE_to_id(ne):
    ne = ne.split('-')
    ne = ([''] * (2 - len(ne))) + ne
    return ALLOWED_NE.index(ne[1]) << 1 | ( 0x01 if ne[0] == 'I' else 0)


def id_to_NE(id):
    prefix = 'I-' if id & 0x01 else 'B-'
    # id 0 means 'O'
    return (prefix if id > 0 else '') + ALLOWED_NE[id >> 1]


def read_corpus(file_name):
    sentences = []  # [n_sentences, n_words]
    labels = []  # [n_sentences, n_words]
    tags = []  # [n_sentences, n_words]
    with open(file_name) as f:
        words, word_labels, word_tags = [], [], []
        for line in f:
            toks = line.rstrip().split('\t')

            if len(toks) < 3:
                sentences.append(words)
                labels.append(word_labels)
                tags.append(word_tags)
                words, word_labels, word_tags = [], [], []
                continue

            words.append(toks[0])
            word_labels.append(toks[2])
            word_tags.append(toks[1])
    return sentences, tags, labels


def read_doc(file_name):
    with open(file_name) as f:
        doc = f.read()

    tagger = Mecab()

    sentences = []  # [n_sentences, n_words]
    tags = []
    start_sentence=0
    for sep in rLINE_SEP.finditer(doc):
        sentence = doc[start_sentence:sep.start(0)]
        sentence = clear_str(sentence)
        start_sentence = sep.end(0)
        if len(sentence) < 10:
            continue
        
        poss = tagger.pos(sentence)
        sentences.append([word for word, _ in poss])
        tags.append([tag for _, tag in poss])
    return sentences, tags


def test_id_to_NE():
    nes = [('O', 0), ('B-PS', 2), ('I-PS', 3), ('B-OG', 4)]
    for ne, id in nes:
        assert NE_to_id(ne) is id, \
                "expected: %d, got: %d" % (id, NE_to_id(ne))
        assert id_to_NE(id) == ne, \
                "expected: %s, got: %s" % (ne, id_to_NE(id))


def main(_):
    if not FLAGS.model_path:
        tf.logging.fatal('Flag --model_path must be set.')
        sys.exit(1)

    if not os.path.exists(FLAGS.model_path):
        os.makedirs(FLAGS.model_path)

    tf.logging.set_verbosity(tf.logging.INFO)

    w2v_file = os.path.join(FLAGS.w2v_path, 'w2v.pkl')
    w2v = Word2Vec(
        model_file=w2v_file,
        skipgram=FLAGS.skipgram,
        unit=FLAGS.unit,
        window=FLAGS.window,
        max_vocab=FLAGS.max_vocab,
        h_softmax=FLAGS.h_softmax,
        batch=FLAGS.batch)

    tag_file = os.path.join(FLAGS.model_path, 'tag.pkl')
    tagmap = IdMapper(tag_file, vocab_size=MAX_TAGS)
    def tags_to_id(tags):
        return tagmap.fit_transform(tags, maxlen=MAX_WORDS_PER_SENTENCE)

    model_fn = build_model_fn(
        max_chars=MAX_CHARS,
        rnn_units=[24, 128],
        max_chars_per_word=MAX_CHARS_PER_WORD,
        max_words_per_sentence=MAX_WORDS_PER_SENTENCE,
        w2v_unit=FLAGS.unit,
        tag_unit=FLAGS.tag_unit,
        max_cates=(len(ALLOWED_NE)*2))

    conf = tf.estimator.RunConfig(
            model_dir=FLAGS.model_path, 
            save_checkpoints_steps=10000,
            keep_checkpoint_max=5)
    estimator = tf.estimator.Estimator(model_fn, config=conf)

    if FLAGS.train:
        sentences, tags, labels = read_corpus(FLAGS.train)
        (cid_sentences, slens) = sentences_to_id(sentences)
        (wid_sentences, _) = sentences_to_w2v_id(sentences, w2v)
        labels = labels_to_id(labels)
        tags = tags_to_id(tags)

        train(estimator, 
              cid_sentences,
              wid_sentences,
              tags,
              slens,
              labels)

    if FLAGS.test:
        sentences, tags, labels = read_corpus(FLAGS.test)
        (cid_sentences, slens) = sentences_to_id(sentences)
        (wid_sentences, _) = sentences_to_w2v_id(sentences, w2v)
        labels = labels_to_id(labels)
        tags = tags_to_id(tags)

        test(estimator, 
              cid_sentences,
              wid_sentences,
              tags,
              slens,
              labels)

    if FLAGS.predict:
        sentences, tags= read_doc(FLAGS.predict)
        (cid_sentences, slens) = sentences_to_id(sentences)
        (wid_sentences, _) = sentences_to_w2v_id(sentences, w2v)
        tags = tags_to_id(tags)

        predict(estimator,
                cid_sentences,
                wid_sentences,
                tags,
                slens,
                sentences)


if __name__ == '__main__':
    tf.app.run()
