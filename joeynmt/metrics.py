# coding: utf-8

import sacrebleu
import pyter
from collections import Counter

def chrf(hypotheses, references, case_sensitive=True):
    """
    Character F-score from sacrebleu

    :param hypotheses:
    :param references:
    :return:
    """
    if not case_sensitive:
        hypotheses = [h.lower() for h in hypotheses]
        references = [r.lower() for r in references]
    return sacrebleu.corpus_chrf(
                    hypotheses=hypotheses,
                    references=references)


def ter(hypotheses, references, case_sensitive=True):
    """
    TER
    :param hypotheses:
    :param references:
    :return:
    """
    assert len(hypotheses) == len(references)
    total_edits = 0
    total_ref_tokens = 0
    for h, r in zip(hypotheses, references):
        if not case_sensitive:
            h = h.lower()
            r = r.lower()
        edit_dist = pyter.edit_distance(h, r)
        total_edits += edit_dist
        total_ref_tokens += len(r)
    #ters = ster(hypotheses, references)
    # like in tercom tool: normalize by total number of tokens in references
    return total_edits/total_ref_tokens


def ster(hypotheses, references, case_sensitive=True):
    """
    Sentence-wise computation of TER
    :param hypotheses:
    :param references:
    :return:
    """
    sters = []
    for hyp, ref in zip(hypotheses, references):
        if not case_sensitive:
            hyp = hyp.lower()
            ref = ref.lower()
        ster = pyter.ter(hyp, ref)
        sters.append(ster)
    return sters


def bleu(hypotheses, references, case_sensitive=True):
    """
    Raw corpus BLEU from sacrebleu (without tokenization)

    :param hypotheses:
    :param references:
    :return:
    """
    if not case_sensitive:
        hypotheses = [h.lower() for h in hypotheses]
        references = [r.lower() for r in references]
    return sacrebleu.raw_corpus_bleu(
                    sys_stream=hypotheses,
                    ref_streams=[references]).score


def sbleu(hypotheses, references, case_sensitive=True):
    sbleus = []
    assert len(hypotheses) == len(references)
    for hyp, ref in zip(hypotheses, references):
        if not case_sensitive:
            hyp = hyp.lower()
            ref = ref.lower()
        sbleus.append(sacrebleu.raw_corpus_bleu(
                      sys_stream=[hyp],
                      ref_streams=[[ref]]).score/100)
    return sbleus

def ngram_counts(sentence, n, lowercase, delimiter= " "):
    """Get n-grams from a sentence.
    Arguments:
        sentence: Sentence as a list of words
        n: n-gram order
        lowercase: Convert ngrams to lowercase
        delimiter: delimiter to use to create counter entries
    """

    counts = Counter()  # type: Counter

    # pylint: disable=too-many-locals
    for begin in range(len(sentence) - n + 1):
        ngram = delimiter.join(sentence[begin:begin + n])
        if lowercase:
            ngram = ngram.lower()

        counts[ngram] += 1

    return counts

def merge_max_counters(counters):
    """Merge counters using maximum values."""
    merged = Counter()  # type: Counter

    for counter in counters:
        for key in counter:
            merged[key] = max(merged[key], counter[key])

    return merged

def total_precision_recall(hypotheses,
        references_list,
        ngrams: int,
        case_sensitive: bool):
    """
    FROM NEURALMONKEY (https://github.com/ufal/neuralmonkey/blob/master/neuralmonkey/evaluators/gleu.py)
    Compute a modified n-gram precision and recall on a sentence list.
    Arguments:
        hypotheses: List of output sentences as lists of words
        references_list: List of lists of reference sentences (as lists of
            words)
        ngrams: n-gram order
        case_sensitive: Whether to perform case-sensitive computation
    """
    corpus_true_positives = 0
    corpus_generated_length = 0
    corpus_target_length = 0

    for n in range(1, ngrams + 1):
        for hypothesis, references in zip(hypotheses, references_list):
            reference_counters = []

            for reference in references:
                counter = ngram_counts(reference, n,
                                                     not case_sensitive)
                reference_counters.append(counter)

            reference_counts = merge_max_counters(
                reference_counters)
            corpus_target_length += sum(reference_counts.values())

            hypothesis_counts = ngram_counts(
                hypothesis, n, not case_sensitive)
            true_positives = 0
            for ngram in hypothesis_counts:
                true_positives += reference_counts[ngram]

            corpus_true_positives += true_positives
            corpus_generated_length += sum(hypothesis_counts.values())

        if corpus_generated_length == 0:
            return 0, 0

    return (corpus_true_positives / corpus_generated_length,
            corpus_true_positives / corpus_target_length)

def gleu(hypotheses, references, case_sensitive):
    """
    GLEU score
    :param hypotheses: list of hypotheses
    :param references: list of list of references
    :return:
    """

    prec, recall = total_precision_recall(
            hypotheses, references, ngrams=4, case_sensitive=case_sensitive)
    return min(recall, prec)


def sgleu(hypotheses, references, case_sensitive=True):
    sgleus = []
    assert len(hypotheses) == len(references)
    for hyp, ref in zip(hypotheses, references):
        sgleus.append(gleu([hyp],[[ref]], case_sensitive=case_sensitive))
    return sgleus

def token_accuracy(hypotheses, references, level="word", case_sensitive=True):
    """
    Compute the accuracy of hypothesis tokens: correct tokens / all tokens
    Tokens are correct if they appear in the same position in the reference.

    :param hypotheses:
    :param references:
    :return:
    """
    correct_tokens = 0
    all_tokens = 0
    split_char = " " if level in ["word", "bpe"] else ""
    assert len(hypotheses) == len(references)
    for h, r in zip(hypotheses, references):
        all_tokens += len(h)
        for h_i, r_i in zip(h.split(split_char), r.split(split_char)):
            # min(len(h), len(r)) tokens considered
            if not case_sensitive:
                h_i = h_i.lower()
                r_i = r_i.lower()
            if h_i == r_i:
                correct_tokens += 1
    return (correct_tokens / all_tokens)*100 if all_tokens > 0 else 0.0


def sequence_accuracy(hypotheses, references, case_sensitive=True):
    """
    Compute the accuracy of hypothesis tokens: correct tokens / all tokens
    Tokens are correct if they appear in the same position in the reference.

    :param hypotheses:
    :param references:
    :return:
    """
    assert len(hypotheses) == len(references)
    if not case_sensitive:
        correct_sequences = sum([1 for (h, r) in zip(hypotheses, references)
                                 if h.lower() == r.lower()])
    else:
        correct_sequences = sum([1 for (h, r) in zip(hypotheses, references)
                             if h == r])
    return (correct_sequences / len(hypotheses))*100 if len(hypotheses) > 0 \
        else 0.0
