import numpy as np
from collections import Counter


class Vocab(object):

    def __init__(self, config):
        self.config = config
        self.load_vocab()

    def load_vocab(self):
        special_tokens = [self.config.unk, self.config.pad, self.config.end]
        self.tok_to_id = load_tok_to_id(self.config.path_vocab, special_tokens)
        self.id_to_tok = {idx: tok for tok, idx in self.tok_to_id.items()}
        self.n_tok = len(self.tok_to_id)

        self.id_pad = self.tok_to_id[self.config.pad]
        self.id_end = self.tok_to_id[self.config.end]
        self.id_unk = self.tok_to_id[self.config.unk]

    @property
    def form_prepro(self):
        return get_form_prepro(self.tok_to_id, self.id_unk)


def get_form_prepro(vocab, id_unk):
    """Given a vocab, returns a lambda function word -> id

    Args:
        vocab: dict[token] = id

    Returns:
        lambda function(formula) -> list of ids

    """
    # test
    def get_token_id(token):
        return vocab[token] if token in vocab else id_unk

    return lambda formula: [get_token_id(t) for t in formula.strip().split(" ")]


def load_tok_to_id(filename, tokens=[]):
    """
    Args:
        filename: (string) path to vocab txt file one word per line
        tokens: list of token to add to vocab after reading filename

    Returns:
        dict: d[token] = id

    """
    tok_to_id = dict()
    with open(filename) as f:
        for idx, token in enumerate(f):
            token = token.strip()
            tok_to_id[token] = idx

    # add extra tokens
    for tok in tokens:
        tok_to_id[tok] = len(tok_to_id)

    return tok_to_id


def build_vocab_from_file(file_paths, min_count=10):
    """Build vocabulary from an iterable of datasets objects

    Args:
        file_paths: a list of file paths, [string,]
        min_count: (int) if token appears less times, do not include it.

    Returns:
        a set of all the words in the file in file_paths

    """
    print("Building vocab...")
    c = Counter()
    for file_path in file_paths:
        with open(file_path) as f:
            for line in f.readlines():
                formula = line.strip()
                try:
                    c.update(formula)
                except Exception:
                    print(formula)
                    raise Exception
    vocab = [tok for tok, count in c.items() if count >= min_count]
    print("- done. {}/{} tokens added to vocab.".format(len(vocab), len(c)))
    return sorted(vocab)


def build_vocab(datasets, min_count=10):
    """Build vocabulary from an iterable of datasets objects

    Args:
        datasets: a list of dataset objects
        min_count: (int) if token appears less times, do not include it.

    Returns:
        a set of all the words in the dataset

    """
    print("Building vocab...")
    c = Counter()
    for dataset in datasets:
        for _, formula in dataset:
            try:
                c.update(formula)
            except Exception:
                print(formula)
                raise Exception
    vocab = [tok for tok, count in c.items() if count >= min_count]
    print("- done. {}/{} tokens added to vocab.".format(len(vocab), len(c)))
    return sorted(vocab)


def write_vocab(vocab, filename):
    """Writes a vocab to a file

    Writes one word per line.

    Args:
        vocab: iterable that yields word
        filename: path to vocab file

    Returns:
        write a word per line

    """
    print("Writing vocab...")
    with open(filename, "w") as f:
        for i, word in enumerate(vocab):
            if i != len(vocab) - 1:
                f.write("{}\n".format(word))
            else:
                f.write(word)
    print("- done. {} tokens".format(i+1))


def pad_batch_formulas(formulas, id_pad, id_end, max_len=None):
    """Pad formulas to the max length with id_pad and adds and id_end token
    at the end of each formula

    Args:
        formulas: (list) of list of ints
        max_length: length maximal of formulas

    Returns:
        array: of shape = (batch_size, max_len) of type np.int32
        array: of shape = (batch_size) of type np.int32

    """
    if max_len is None:
        max_len = max(map(lambda x: len(x), formulas))

    batch_formulas = id_pad * np.ones([len(formulas), max_len+1], dtype=np.int32)
    formula_length = np.zeros(len(formulas), dtype=np.int32)
    for idx, formula in enumerate(formulas):
        batch_formulas[idx, :len(formula)] = np.asarray(formula, dtype=np.int32)
        batch_formulas[idx, len(formula)] = id_end
        formula_length[idx] = len(formula) + 1

    return batch_formulas, formula_length


def load_formulas(filename):
    formulas = dict()
    with open(filename) as f:
        for idx, line in enumerate(f):
            formulas[idx] = line.strip()

    print("Loaded {} formulas from {}".format(len(formulas), filename))
    return formulas
