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
    def __init__(self, output_size, src_emb_size, trg_emb_size):
        super(Regulator, self).__init__()
        self.output_size = output_size
        self.src_emb_size = src_emb_size
        self.trg_emb_size = trg_emb_size

    def get_costs(self, pred):
        """
        Compute the cost for a prediction
        score between 0 and 1

        :param pred:
        :return:
        """
        #print("HERE")
        #print("pred", pred)
        # TODO make more sophisticated
        cost_dict = {0: 0, 1: 0, 2: 0.5, 3: 1}
        # one cost for every prediction
        total_cost = 0
        #for p in pred:
        #    if p == 0:
        #        total_cost += 0
        #    elif p == 1:
        #        total_cost += 1
        #    elif p == 2:
        #        total_cost += 2
        #    elif p == 3:
        #        total_cost += 3
        total_cost = np.array([cost_dict[p] for p in pred])
        #max_cost = 3
        #total_cost = pred/max_cost
        #print("total cost", total_cost)
        # what's the cost for each type of feedback?
        # no update: no cost
        # weak feedback: accuracy
        # post-edit: TER
        # TODO pass output here to compute cost
        # or compute cost before and just do the weighting here?
        return total_cost


class RecurrentRegulator(Regulator):
    """
    Recurrent regulator model that predicts a feedback mode
    """

    # TODO use only src
    def __init__(self,
                 output_size,
                 type,
                 hidden_size,
                 src_emb_size,
                 trg_emb_size,
                 num_layers,
                 bidirectional,
                 dropout,
                 **kwargs):
        super(RecurrentRegulator, self).__init__(
            output_size, src_emb_size, trg_emb_size)

        rnn = nn.GRU if type == "gru" else nn.LSTM

        self.src_rnn = rnn(
            self.src_emb_size, hidden_size, num_layers, batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.)

        self.trg_rnn = rnn(
            self.trg_emb_size, hidden_size, num_layers, batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.)

        self.rnn_input_dropout = torch.nn.Dropout(p=dropout, inplace=False)

        self.output_layer = nn.Linear(
            in_features=self.src_rnn.hidden_size*(2 if bidirectional else 1)+
                        self.trg_rnn.hidden_size*(2 if bidirectional else 1),
            out_features=output_size
        )

    def forward(self, src, hyp):
        """
        Read src and hyp with Bi-RNNs, take last hidden states and combine
        :param src:
        :param hyp:
        :return:
        """
        src_embedded = self.rnn_input_dropout(src)
        trg_embedded = self.rnn_input_dropout(hyp)

        src_rnn_output, src_rnn_hidden = self.src_rnn(src_embedded)
        trg_rnn_output, trg_rnn_hidden = self.trg_rnn(trg_embedded)

        if isinstance(src_rnn_hidden, tuple):
            src_rnn_hidden, src_rnn_memory_cell = src_rnn_hidden
        if isinstance(trg_rnn_hidden, tuple):
            trg_rnn_hidden, trg_rnn_memory_cell = trg_rnn_hidden

       # print("src_hidden", src_rnn_hidden.shape)  # direction*layer x batch x hidden
       # print("trg_hidden", trg_rnn_hidden.shape)

        # hidden: dir*layers x batch x hidden
        # output: batch x max_length x directions*hidden
        batch_size = src_rnn_hidden.size()[1]
        # separate final hidden states by layer and direction
        src_hidden_layerwise = src_rnn_hidden.view(self.src_rnn.num_layers,
                                       2 if self.src_rnn.bidirectional else 1,
                                       batch_size, self.src_rnn.hidden_size)
        trg_hidden_layerwise = trg_rnn_hidden.view(self.trg_rnn.num_layers,
                                                   2 if self.trg_rnn.bidirectional else 1,
                                                   batch_size,
                                                   self.trg_rnn.hidden_size)
        # final_layers: layers x directions x batch x hidden

        # concatenate the final states of the last layer for each directions
        # thanks to pack_padded_sequence final states don't include padding

        # TODO get final states without padding
        # TODO use src_lengths and hyp_lengths
        src_fw_hidden_last = src_hidden_layerwise[-1:, 0].squeeze(0)
        src_bw_hidden_last = src_hidden_layerwise[-1:, 1].squeeze(0)
        trg_fw_hidden_last = trg_hidden_layerwise[-1:, 0].squeeze(0)
        trg_bw_hidden_last = trg_hidden_layerwise[-1:, 1].squeeze(0)

        comb_states = torch.cat([src_fw_hidden_last, src_bw_hidden_last,
                                 trg_fw_hidden_last, trg_bw_hidden_last], dim=1)

        # TODO activation function?
        output = self.output_layer(comb_states)

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

    def __init__(self, emb_size, output_size):
        super(AttentionalRegulator, self).__init__(emb_size=emb_size,
                                                   output_size=output_size)

class ConvolutionalRegulator(Regulator):
    """
    Convolutional regulator model that predicts a feedback mode
    """

    def __init__(self, emb_size, output_size):
        super(ConvolutionalRegulator, self).__init__(emb_size=emb_size,
                                                     output_size=output_size)

