# coding: utf-8

import numpy as np
import sys

def kenlm2weights(infile):
    """
    Read a file scored by KenLM and extract all word & sent weights
    :param infile:
    :return:
    """
    token_weights = []
    line_counter = 0
    with open(infile, "r") as f:
        for line in f:
            line_counter += 1
            if "OOVs:" in line or "Tokens:" in line or line.startswith("Perplexity "):
                continue
            # last token is OOV count: ignore it
            token_and_weights = line.strip().split("\t")[:-1]
            weights = [float(tw.split(" ")[2]) for tw in token_and_weights]
            token_weights.append(weights)
        assert len(token_weights) == line_counter-4
        sent_weights = [sum(t) for t in token_weights]
        return token_weights, sent_weights

def write_weights(weights, infile):
    """
    Write weights to file
    :param weights: (token_weights, sent_weights)
    :param infile: prefix for weights
    :return:
    """
    token_weights, sent_weights = weights
    # normalize sent_weights by sentence length
    with open(infile+".sent.weights", "w") as sentf, \
            open(infile+".token.weights", "w") as tokenf, \
            open(infile+".norm.sent.weights", "w") as sentf_norm:

        for sent_w, token_w in zip(sent_weights, token_weights):
            sent_len = len(token_w)
            norm_sent_w = sent_w/sent_len
            sentf.write("{}\n".format(sent_w))
            sentf_norm.write("{}\n".format(norm_sent_w))
            for t in token_w:
                tokenf.write("{} ".format(t))
            tokenf.write("\n")




if __name__ == "__main__":
    infile = sys.argv[1]
    weights = kenlm2weights(infile)
    write_weights(weights, infile)