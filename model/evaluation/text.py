import os
import sys
import numpy as np
import nltk
import distance


from ..utils.text import load_formulas
from ..utils.general import init_dir


def score_files(path_ref, path_hyp):
    """Loads result from file and score it

    Args:
        path_ref: (string) formulas of reference
        path_hyp: (string) formulas of prediction.

    Returns:
        scores: (dict)

    """
    # load formulas
    formulas_ref = load_formulas(path_ref)
    formulas_hyp = load_formulas(path_hyp)

    assert len(formulas_ref) == len(formulas_hyp)

    # tokenize
    refs = [ref.split(" ") for _, ref in formulas_ref.items()]
    hyps = [hyp.split(" ") for _, hyp in formulas_hyp.items()]

    # score
    return {
        "BLEU-4": bleu_score(refs, hyps)*100,
        "ExactMatchScore": exact_match_score(refs, hyps)*100,
        "EditDistance": edit_distance(refs, hyps)*100
    }


def exact_match_score(references, hypotheses):
    """Computes exact match scores.

    Args:
        references: list of list of tokens (one ref)
        hypotheses: list of list of tokens (one hypothesis)

    Returns:
        exact_match: (float) 1 is perfect

    """
    exact_match = 0
    for ref, hypo in zip(references, hypotheses):
        if np.array_equal(ref, hypo):
            exact_match += 1

    return exact_match / float(max(len(hypotheses), 1))


def bleu_score(references, hypotheses):
    """Computes bleu score.

    Args:
        references: list of list (one hypothesis)
        hypotheses: list of list (one hypothesis)

    Returns:
        BLEU-4 score: (float)

    """
    references = [[ref] for ref in references]  # for corpus_bleu func
    BLEU_4 = nltk.translate.bleu_score.corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))
    return BLEU_4


def edit_distance(references, hypotheses):
    """Computes Levenshtein distance between two sequences.

    Args:
        references: list of list of token (one hypothesis)
        hypotheses: list of list of token (one hypothesis)

    Returns:
        1 - levenshtein distance: (higher is better, 1 is perfect)

    """
    d_leven, len_tot = 0, 0
    for ref, hypo in zip(references, hypotheses):
        d_leven += distance.levenshtein(ref, hypo)
        len_tot += float(max(len(ref), len(hypo)))

    return 1. - d_leven / len_tot


def truncate_end(list_of_ids, id_end):
    """Removes the end of the list starting from the first id_end token"""
    list_trunc = []
    for idx in list_of_ids:
        if idx == id_end:
            break
        else:
            list_trunc.append(idx)

    return list_trunc


def write_answers(references, hypotheses, rev_vocab, dir_name, id_end):
    """Writes text answers in files.

    One file for the reference, one file for each hypotheses

    Args:
        references: list of list         (one reference)
        hypotheses: list of list of list (multiple hypotheses)
            hypotheses[0] is a list of all the first hypothesis for all the
            dataset
        rev_vocab: (dict) rev_vocab[idx] = word
        dir_name: (string) path where to write results
        id_end: (int) special id of token that corresponds to the END of
            sentence

    Returns:
        file_names: list of the created files

    """
    def ids_to_str(ids):
        ids = truncate_end(ids, id_end)
        s = [rev_vocab[idx] for idx in ids]
        return " ".join(s)

    def write_file(file_name, list_of_list):
        print("writting file ", file_name)
        with open(file_name, "w") as f:
            for l in list_of_list:
                f.write(ids_to_str(l) + "\n")

    init_dir(dir_name)
    file_names = [dir_name + "ref.txt"]
    write_file(dir_name + "ref.txt", references)  # one file for the ref
    for i in range(len(hypotheses)):              # one file per hypo
        assert len(references) == len(hypotheses[i])
        write_file(dir_name + "hyp_{}.txt".format(i), hypotheses[i])
        file_names.append(dir_name + "hyp_{}.txt".format(i))

    return file_names
