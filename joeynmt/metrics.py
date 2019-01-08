# coding: utf-8

import sacrebleu


def chrf(hypotheses, references, corpus=True):
    """
    Character F-score from sacrebleu

    :param hypotheses:
    :param references:
    :param corpus: if False, mean sentence-level metric
    :return:
    """
    if corpus:
        return sacrebleu.corpus_chrf(
                        hypotheses=hypotheses,
                        references=references)
    else:
        # mean of sentence metric
        total_chrf = 0
        num_hyps = 0
        for hyp, ref in zip(hypotheses, references):
            total_chrf += sacrebleu.corpus_chrf(
                        hypotheses=[hyp],
                        references=[ref])
            num_hyps += 1
        return total_chrf / max(num_hyps, 1)


def bleu(hypotheses, references, corpus=True):
    """
    Raw corpus BLEU from sacrebleu (without tokenization)

    :param hypotheses:
    :param references:
    :param corpus: if False, mean sentence-level metric
    :return:
    """
    if corpus:
        return sacrebleu.raw_corpus_bleu(
                    sys_stream=hypotheses,
                    ref_streams=[references]).score

    else:  # mean of sentence metric
        total_sbleu = 0
        num_hyps = 0
        for hyp, ref in zip(hypotheses, references):
            total_sbleu += sacrebleu.raw_corpus_bleu(
                sys_stream=[hyp],
                ref_streams=[[ref]]).score
            num_hyps += 1
        return total_sbleu / max(num_hyps, 1)


def token_accuracy(hypotheses, references, level="word"):
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
            if h_i == r_i:
                correct_tokens += 1
    return (correct_tokens / all_tokens)*100 if all_tokens > 0 else 0.0


def sequence_accuracy(hypotheses, references):
    """
    Compute the accuracy of hypothesis tokens: correct tokens / all tokens
    Tokens are correct if they appear in the same position in the reference.

    :param hypotheses:
    :param references:
    :return:
    """
    assert len(hypotheses) == len(references)
    correct_sequences = sum([1 for (h, r) in zip(hypotheses, references)
                             if h == r])
    return (correct_sequences / len(hypotheses))*100 if len(hypotheses) > 0 \
        else 0.0
