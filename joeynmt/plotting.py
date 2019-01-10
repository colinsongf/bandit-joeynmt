#!/usr/bin/env python

import matplotlib

matplotlib.use('Agg')

import numpy as np
from matplotlib import rcParams
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def plot_correction(correction, reward, trg, corr_trg,
                    output_path):

    # only plot as many corrections as trg or corr_trg are long
    max_len = max(len(trg), len(corr_trg))
    abs_mean_corr = np.mean(np.abs(correction[:max_len]), axis=1, keepdims=True)
    while len(trg) < max_len:
        trg.append("#")

    labelsize = 25 * (10 / max(1, max_len))

    # font config
    rcParams['xtick.labelsize'] = labelsize
    rcParams['ytick.labelsize'] = labelsize

    fig = plt.figure(dpi=300)

    ax1 = plt.subplot(211)
    ax1.imshow(abs_mean_corr.T, cmap='viridis', origin='upper')
    ax1.set_ylabel("Correction")
    ax1.set(xlabel='Trg', ylabel='Correction')

    ax1.set_xticklabels(trg, minor=False, rotation="vertical")
    ax1.xaxis.tick_top()
    ax1.set_yticklabels([])
    ax1.set_xticks(np.arange(len(trg)) + 0, minor=False)

    #ax1.get_yaxis().set_visible(False)
    ax1.get_xaxis().set_visible(False)

    ax2 = plt.subplot(212, sharex=ax1)
    ax2.imshow(1-reward[:max_len].T, vmin=0, vmax=1, cmap='viridis',
                             origin='upper')
    #ax2.set_ylabel("Reward")
    ax2.set(xlabel='Trg', ylabel='1-Reward')
    #ax2.get_yaxis().set_visible(False)
    ax2.set_yticklabels([])
    ax2.set_xticklabels(trg, minor=False, rotation="vertical")

    # TODO add corr
    #plt.figtext(0.5, 0.01, " ".join(corr_trg),
    #            wrap=True, horizontalalignment='center',
    #            fontsize=labelsize)

    plt.tight_layout()

    if output_path.endswith(".pdf"):
        pp = PdfPages(output_path)
        pp.savefig(fig)
        pp.close()
    else:
        if not output_path.endswith(".png"):
            output_path += ".png"
        plt.savefig(output_path)

    plt.close()


def plot_attention(scores=None, column_labels=None, row_labels=None,
                   output_path="plot.png", vmin=0., vmax=1.):
    """
    Plotting function that can be used to visualize (self-)attention.
    Plots are saved if `output_path` is specified, in format that this file
    ends with ('pdf' or 'png').

    :param scores: attention scores
    :param column_labels:  labels for columns (e.g. target tokens)
    :param row_labels: labels for rows (e.g. source tokens)
    :param output_path: path to save to
    :param normalized_input: scores are normalized and between 0 and 1
    :return:
    """

    assert output_path.endswith(".png") or output_path.endswith(".pdf"), \
        "output path must have .png or .pdf extension"

    x_sent_len = len(column_labels)
    if row_labels is not None:
        y_sent_len = len(row_labels)
    else:
        y_sent_len = -1
    scores = scores[:y_sent_len, :x_sent_len]
    # check that cut off part didn't have any attention
    #assert np.sum(scores[y_sent_len:, :x_sent_len]) == 0

    # automatic label size
    x_labelsize = 25 * (10 / max(x_sent_len, y_sent_len))
    y_labelsize = 25 * (10 / max(x_sent_len, y_sent_len))


    # font config
    rcParams['xtick.labelsize'] = x_labelsize
    rcParams['ytick.labelsize'] = y_labelsize
    #rcParams['font.family'] = "sans-serif"
    #rcParams['font.sans-serif'] = ["Fira Sans"]
    #rcParams['font.weight'] = "regular"

    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
    heatmap = plt.imshow(scores, cmap='viridis', aspect='equal',
                             origin='upper', vmin=vmin, vmax=vmax)
    if vmin != 0.0 and vmax != 1.0:
        plt.colorbar()

    ax.set_xticklabels(column_labels, minor=False,
                       rotation="vertical" if row_labels is not None
                       else "horizontal")
    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(scores.shape[1]) + 0, minor=False)

    if row_labels is not None:
        ax.set_yticklabels(row_labels, minor=False)
        ax.set_yticks(np.arange(scores.shape[0]) + 0, minor=False)
    else:
        ax.get_yaxis().set_visible(False)

    plt.tight_layout()

    if output_path.endswith(".pdf"):
        pp = PdfPages(output_path)
        pp.savefig(fig)
        pp.close()
    else:
        if not output_path.endswith(".png"):
            output_path += ".png"
        plt.savefig(output_path)

    plt.close()
