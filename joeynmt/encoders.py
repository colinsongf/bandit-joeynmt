# coding: utf-8
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

"""
Various encoders
"""

# TODO make general encoder class
class Encoder(nn.Module):
    """
    Base encoder class
    """
    _output_size = 0

    @property
    def output_size(self):
        return self._output_size

    pass


class RecurrentEncoder(Encoder):
    """Encodes a sequence of word embeddings"""

    def __init__(self,
                 type: str = "gru",
                 hidden_size: int = 1,
                 emb_size: int = 1,
                 num_layers: int = 1,
                 dropout: float = 0.,
                 bidirectional: bool = True,
                 lm_task: float = 0,
                 vocab_size: int = None,
                 **kwargs):
        """
        Create a new recurrent encoder.
        :param type:
        :param hidden_size:
        :param emb_size:
        :param num_layers:
        :param dropout:
        :param bidirectional:
        :param lm_task: predict next word from fwd hidden states
        :param vocab_size: only needed if lm_task
        :param kwargs:
        """

        super(RecurrentEncoder, self).__init__()

        self.rnn_input_dropout = torch.nn.Dropout(p=dropout, inplace=False)
        self.type = type
        self.num_layers = num_layers

        rnn = nn.GRU if type == "gru" else nn.LSTM
        input_size = emb_size

        self.lm_task = lm_task
        if self.lm_task > 0:
            self.projection_layer = nn.Linear(hidden_size, hidden_size,
                                              bias=True)
            self.output_layer = nn.Linear(hidden_size, vocab_size, bias=False)
            if num_layers > 1:
                # treat the 1st layer separately so we can access all hidden states
                num_layers -= 1
                self.lm_rnn = rnn(input_size, hidden_size, 1, batch_first=True,
                                  bidirectional=True)
                self.lm_dropout = torch.nn.Dropout(p=dropout, inplace=False)
                input_size = hidden_size * (2 if bidirectional else 1)

        self.rnn = rnn(
            input_size, hidden_size, num_layers, batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.)

        self._output_size = 2 * hidden_size if bidirectional else hidden_size



    def forward(self, x, x_length, mask):
        """
        Applies a bidirectional RNN to sequence of embeddings x.
        The input mini-batch x needs to be sorted by src length.
        x should have dimensions [batch, time, dim].
        The masks indicates padding areas (zeros where padding).
        :param x:
        :param x_length:
        :param mask:
        :return:
        """
        # apply dropout ot the rnn input
        x = self.rnn_input_dropout(x)

        packed = pack_padded_sequence(x, x_length, batch_first=True)

        if self.lm_task > 0 and self.num_layers > 1:
            # first pass it through first layer
            lm_rnn_output, _ = self.lm_rnn(packed)
            lm_rnn_output, _ = pad_packed_sequence(lm_rnn_output, batch_first=True)
            # (batch, seq_len, input_size)
            # then pass it through the rest of the layers
            output, hidden = self.rnn(lm_rnn_output)
        elif self.lm_task > 0 and self.num_layers == 1:
            # only one layer and lm: use standard RNN
            output, hidden = self.rnn(packed)
            output, _ = pad_packed_sequence(output, batch_first=True)
            lm_rnn_output = output
        else:
            # standard, w/o lm
            output, hidden = self.rnn(packed)
            output, _ = pad_packed_sequence(output, batch_first=True)
            # hidden: dir*layers x batch x hidden (only final time step)
            # output: batch x max_length x directions*hidden (only final layer)

        if isinstance(hidden, tuple):
            hidden, memory_cell = hidden

        batch_size = hidden.size(1)

        # separate final hidden states by layer and direction
        # FIXME in case of lm, first layer is not included here
        hidden_layerwise = hidden.view(self.rnn.num_layers,
                                       2 if self.rnn.bidirectional else 1,
                                       batch_size, self.rnn.hidden_size)
        # final_layers: layers x directions x batch x hidden

        lm_output = None
        if self.lm_task > 0:
            #output_layerwise = lm_rnn_output.view(batch_size, lm_rnn_output.size(1),
            #    2 if self.rnn.bidirectional else 1, self.rnn.hidden_size)
            # make word predictions from fwd hidden states of first layer
            # not last layer because it contains backwards info from previous layers
            #print("chunk", torch.chunk(lm_rnn_output, 2, dim=-1)[0].size())
            first_fw = torch.chunk(lm_rnn_output, 2, dim=-1)[0]
            # output_layerwise[:, :, 0, :]
            # TODO is projection needed?
            lm_projection = torch.tanh(self.projection_layer(first_fw))
            lm_output = self.output_layer(lm_projection)
            #print(lm_output)
            #print(lm_output.size())
            # batch x src_vocab_size

        # concatenate the final states of the last layer for each directions
        # thanks to pack_padded_sequence final states don't include padding
        fwd_hidden_last = hidden_layerwise[-1:, 0]
        # TODO allow unidirectional
        if self.rnn.bidirectional:
            bwd_hidden_last = hidden_layerwise[-1:, 1]
        else:
            bwd_hidden_last = fwd_hidden_last

        # only feed the final state of the top-most layer to the decoder
        hidden_concat = torch.cat(
            [fwd_hidden_last, bwd_hidden_last], dim=2).squeeze(0)
        # final: batch x directions*hidden
        return output, hidden_concat, lm_output

    def sample(self, max_output_length, initial_input, src_embeddings):
        """
        Sample src outputs from encoder hidden states
        :param max_output_length:
        :param initial_input:
        :return:
        """
        initial_symbols = initial_input.src[:, 0].view(
            initial_input.src.size(0), 1)
        embedded_input = src_embeddings(initial_symbols)
        outputs = []
        hidden = None
        for t in range(max_output_length):
            if self.num_layers > 1:
                # first pass it through first layer
                output, hidden = self.lm_rnn(embedded_input, hidden)
            else:
                # only one layer and lm: use standard RNN
                output, hidden = self.rnn(embedded_input, hidden)
            # batch x 1 x hidden_size*directions
            fw_output = output.chunk(chunks=2, dim=-1)[0]
            # batch x 1 x hidden_size
            lm_projection = torch.tanh(self.projection_layer(fw_output))
            lm_output = self.output_layer(lm_projection).squeeze(1)
            # batch x src_vocab
            predicted = torch.multinomial(
                input=torch.nn.functional.softmax(lm_output, dim=-1), num_samples=1)
            outputs.append(predicted)
            # batch x 1
            embedded_input = src_embeddings(predicted)
            # batch x 1 x embed_size
        outputs = torch.cat(outputs, dim=1)
        # TODO track logprob
        return outputs


    def __repr__(self):
        return "%s(%r)" % (self.__class__.__name__, self.rnn)

