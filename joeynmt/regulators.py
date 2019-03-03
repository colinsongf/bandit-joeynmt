# coding: utf-8
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from joeynmt.helpers import freeze_params
import numpy as np

"""
Various encoders
"""


# TODO make general encoder class
class Regulator(nn.Module):
    """
    Base regulator class
    """
    def __init__(self, output_size, src_emb_size): #, trg_emb_size):
        super(Regulator, self).__init__()
        self.output_size = output_size
        self.src_emb_size = src_emb_size
        #self.trg_emb_size = trg_emb_size

class RecurrentRegulator(Regulator):
    """
    Recurrent regulator model that predicts a feedback mode
    """

    # TODO use only src
    # TODO use max pool to make interpretable?
    def __init__(self,
                 output_labels,
                 type,
                 hidden_size,
                 middle_size,
                 src_emb_size,
                 #trg_emb_size,
                 num_layers,
                 bidirectional,
                 dropout,
                 feed_trg=False,
                 **kwargs):
        # mapping output indices to labels
        self.index2label = dict(enumerate(output_labels))
        # mapping labels to output indices
        self.label2index = {v: k for (k,v) in self.index2label.items()}
        self.output_size = len(output_labels)
        super(RecurrentRegulator, self).__init__(
            self.output_size, src_emb_size)#, trg_emb_size)

        rnn = nn.GRU if type == "gru" else nn.LSTM

        self.src_rnn = rnn(
            self.src_emb_size, hidden_size, num_layers, batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.)

        #self.trg_rnn = rnn(
        #    self.trg_emb_size, hidden_size, num_layers, batch_first=True,
        #    bidirectional=bidirectional,
        #    dropout=dropout if num_layers > 1 else 0.)

        self.rnn_input_dropout = torch.nn.Dropout(p=dropout, inplace=False)

        self.middle_layer = nn.Linear(in_features=self.src_rnn.hidden_size*(2 if bidirectional else 1),
                                      out_features=middle_size)

        self.output_layer = nn.Linear(
            in_features=self.middle_layer.out_features,
            #self.src_rnn.hidden_size*(2 if bidirectional else 1),
                      #  self.trg_rnn.hidden_size*(2 if bidirectional else 1),
            out_features=self.output_size
        )
        self.feed_trg = feed_trg

    def forward(self, src, src_length, hyp=None):
        """
        Read src with Bi-RNNs, take last hidden states and combine
        :param src:
        :return:
        """
        # TODO this is the same as in MT encoder

        # apply dropout ot the rnn input
        src_embedded = self.rnn_input_dropout(src)

        packed = pack_padded_sequence(src_embedded, src_length, batch_first=True)
        output, hidden = self.src_rnn(packed)

        if isinstance(hidden, tuple):
            hidden, memory_cell = hidden

        output, _ = pad_packed_sequence(output, batch_first=True)
        # hidden: dir*layers x batch x hidden
        # output: batch x max_length x directions*hidden
        batch_size = hidden.size()[1]
        # separate final hidden states by layer and direction
        hidden_layerwise = hidden.view(self.src_rnn.num_layers,
                                       2 if self.src_rnn.bidirectional else 1,
                                       batch_size, self.src_rnn.hidden_size)
        # final_layers: layers x directions x batch x hidden

        # concatenate the final states of the last layer for each directions
        # thanks to pack_padded_sequence final states don't include padding
        fwd_hidden_last = hidden_layerwise[-1:, 0]
        bwd_hidden_last = hidden_layerwise[-1:, 1]

        # final state of the top-most layer
        hidden_concat = torch.cat(
            [fwd_hidden_last, bwd_hidden_last], dim=2).squeeze(0)

        if self.feed_trg and hyp is not None:
            # also read in hyps (with same params)
            # but they are not sorted
            hyp_embedded = self.rnn_input_dropout(hyp)
            output_hyp, hidden_hyp = self.src_rnn(hyp_embedded)
            #print("output hyp", output_hyp.shape)  # batch x length x 2*rnn_size
            if isinstance(hidden_hyp, tuple):
                hidden_hyp, memory_cell_hyp = hidden_hyp
            #print("hidden hyp", hidden_hyp.shape)  # 4 x batch x rnn_size

            hidden_layerwise_hyp = hidden.view(self.src_rnn.num_layers,
                                           2 if self.src_rnn.bidirectional else 1,
                                           batch_size, self.src_rnn.hidden_size)
            #print("hidden hyp", hidden_layerwise_hyp.shape)  # layers x directions x batch x rnn_size
            fwd_hidden_last_hyp = hidden_layerwise_hyp[-1:, 0]
            bwd_hidden_last_hyp = hidden_layerwise_hyp[-1:, 1]

            # final state of the top-most layer
            hidden_concat_hyp = torch.cat(
                [fwd_hidden_last_hyp, bwd_hidden_last_hyp], dim=2).squeeze(0)

            input_to_middle = hidden_concat+hidden_concat_hyp
        else:
            input_to_middle = hidden_concat

        middle = torch.tanh(self.middle_layer(input_to_middle))

        # TODO activation function?
        output = self.output_layer(middle)

        return output



        # only feed the final state of the top-most layer to the decoder
        #hidden_concat = torch.cat(
        #    [fwd_hidden_last, bwd_hidden_last], dim=2).squeeze(0)
        # final: batch x directions*hidden
        #return output, hidden_concat

class AttentionalRegulator(Regulator):
    """
       Attentional regulator model that predicts a feedback mode
       """

    def __init__(self, src_embed_size, output_size):
        super(AttentionalRegulator, self).__init__(output_size=output_size,
                                                   src_emb_size=src_embed_size)

class ConvolutionalRegulator(Regulator):
    """
    Convolutional regulator model that predicts a feedback mode
    """

    def __init__(self, src_emb_size, output_size):
        super(ConvolutionalRegulator, self).__init__(src_emb_size=src_emb_size,
                                                     output_size=output_size)

