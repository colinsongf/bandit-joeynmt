import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from joeynmt.attention import BahdanauAttention, LuongAttention, AttentionMechanism
from joeynmt.encoders import Encoder
from joeynmt.embeddings import Embeddings
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# TODO make general corrector class
class Corrector(nn.Module):
    pass


class RecurrentCorrector(Corrector):
    """A conditional RNN decoder with attention."""


    def __init__(self,
                 type: str = "gru",
                 trg_embed: Embeddings = None,
                 hidden_size: int = 0,
                 num_layers: int = 0,
                 dropout: float = 0.,
                 bridge: bool = False,
                 bidirectional: bool = False,
                 decoder_size: int = 0,
                 activation: str = "tanh",
                 **kwargs):

        super(RecurrentCorrector, self).__init__()

        self.trg_embed = trg_embed
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_input_dropout = torch.nn.Dropout(p=dropout, inplace=False)
        self.bridge = bridge
        self.type = type
        self.output_size = decoder_size

        rnn = nn.GRU if type == "gru" else nn.LSTM

        embed_size = trg_embed.embedding_dim

        # backwards RNN (could also be bidirectional)
        # reads in the produced words
        self.rnn = rnn(embed_size, hidden_size, num_layers,
                       batch_first=True,
                       dropout=dropout if num_layers > 1 else 0.)

        # output layer: receives previous prediction as input
        # 1 layer RNN on concatenated decoder and bw states
        self.output_rnn = rnn(decoder_size+hidden_size+decoder_size,
                              hidden_size, num_layers=1, batch_first=True)

        self.output_layer = nn.Linear(hidden_size, decoder_size)

        if activation == "tanh":
            self.activation = torch.tanh
        elif activation == "relu":
            self.activation = F.relu
        elif activation == "leakyrelu":
            self.activation = F.leaky_relu


    def forward(self, y, y_length, mask, y_states):
        """
        Reads the decoder output backwards,
        combines it with decoder hidden states
        and predicts edits to hidden states.

        :param y: embedded, reversed decoder predictions
        :param y_length:
        :param mask:
        :return:
        """
        # TODO make use of length and mask!
        # apply dropout ot the rnn input (embedded decoder predictions)
        x = self.rnn_input_dropout(y) # batch x time x embed
        # run through rnn (backwards)
        hidden = None
        rnn_outputs = []
        for t in range(x.shape[1]):
            x_i = x[:,t,:].unsqueeze(1)
            rnn_output, hidden = self.rnn(x_i, hx=hidden)  # batch x 1 x hidden
            rnn_outputs.append(rnn_output.squeeze(1))
        rnn_outputs = torch.stack(rnn_outputs, dim=1)
        #print("bw rnn output", rnn_outputs.shape)  # batch x time x hidden

        # concat with y_states
        comb_states = torch.cat([y_states, rnn_outputs], dim=2)  # batch x time x decoder.hidden_size+hidden

        # now make a prediction for every time step with rnn
        hidden = None
        outputs = []
        with torch.no_grad():
            prev_pred = comb_states.new_zeros(comb_states.shape[0], 1,
                                              self.output_size)
            #print(prev_pred.shape)
        for t in range(x.shape[1]):
            comb_i = comb_states[:, t, :].unsqueeze(1)
            #print(comb_i.shape)
            # feed in both previous prediction and combination of states
            input_i = torch.cat([comb_i, prev_pred], dim=2)
            #print("inpi", input_i.shape)
            # TODO might add layers here to make rnn smaller
            rnn_output, hidden = self.output_rnn(input_i, hx=hidden)
            prev_pred = self.activation(self.output_layer(rnn_output))
            outputs.append(prev_pred.squeeze(1))
        outputs = torch.stack(outputs, dim=1)
        #print("outputs", outputs.shape)

        # read in the translation backwards
        # bidirectional only works if sorted!
        #packed = pack_padded_sequence(y, y_length, batch_first=True)
        #rnn_output, hidden = self.rnn(packed)

        #if isinstance(hidden, tuple):
        #    hidden, memory_cell = hidden

        #rnn_output, _ = pad_packed_sequence(rnn_output, batch_first=True)
        # hidden: dir*layers x batch x hidden
        # output: batch x max_length x directions*hidden
        #batch_size = hidden.size()[1]

        # separate final hidden states by layer and direction
        #hidden_layerwise = hidden.view(self.rnn.num_layers,
        #                               2 if self.rnn.bidirectional else 1,
        #                               batch_size, self.rnn.hidden_size)
        # final_layers: layers x directions x batch x hidden

        # concatenate the final states of the last layer for each directions
        # thanks to pack_padded_sequence final states don't include padding
        #fwd_hidden_last = hidden_layerwise[-1:, 0]
        #bwd_hidden_last = hidden_layerwise[-1:, 1]

        # only feed the final state of the top-most layer to the decoder
        #hidden_concat = torch.cat(
        #    [fwd_hidden_last, bwd_hidden_last], dim=2).squeeze(0)
        # final: batch x directions*hidden

        # combine with decoder hidden states and predict vector for each position
        # make RNN as well: previous correction should influence future correction!
        #output = self.activation(self.output_layer(comb_states))

        return outputs




