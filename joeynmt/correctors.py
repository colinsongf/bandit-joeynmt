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
                 hidden_dropout: float = 0.,
                 bridge: bool = False,
                 bidirectional: bool = False,
                 decoder_size: int = 0,
                 corr_activation: str = "tanh",
                 reward_activation: str = "sigmoid",
                 freeze: bool = False,
                 attention: str = "bahdanau",
                 encoder: Encoder = None,
                 reward_coeff: float = 0,
                 shift_rewards: bool = False,
                 **kwargs):

        super(RecurrentCorrector, self).__init__()

        self.trg_embed = trg_embed
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_input_dropout = torch.nn.Dropout(p=dropout, inplace=False)
        self.bridge = bridge
        self.type = type
        self.output_size = decoder_size
        self.reward_coeff = reward_coeff
        self.shift_rewards = shift_rewards

        rnn = nn.GRU if type == "gru" else nn.LSTM

        embed_size = trg_embed.embedding_dim

        # backwards RNN (could also be bidirectional)
        # reads in the produced words
        self.rnn = rnn(embed_size, hidden_size, num_layers,
                       batch_first=True,
                       dropout=dropout if num_layers > 1 else 0.)

        # output layer: receives previous prediction as input
        # 1 layer RNN on concatenated decoder and bw states
        self.output_rnn = rnn(
            decoder_size+hidden_size+decoder_size+hidden_size,
            hidden_size, num_layers=1, batch_first=True)

        self.corr_output_layer = nn.Linear(hidden_size, decoder_size)

        # combine output with context vector before output layer (Luong-style)
        self.att_vector_layer = nn.Linear(
            hidden_size + encoder.output_size, hidden_size, bias=True)

        # predict a reward
        # TODO binary?
        self.reward_output_layer = nn.Linear(hidden_size, 2)

        if corr_activation == "tanh":
            self.corr_activation = torch.tanh
        elif corr_activation == "relu":
            self.corr_activation = F.relu
        elif corr_activation == "leakyrelu":
            self.corr_activation = F.leaky_relu

        if reward_activation == "tanh":
            self.reward_activation = torch.tanh
        elif reward_activation == "relu":
            self.reward_activation = F.relu
        elif reward_activation == "leakyrelu":
            self.reward_activation = F.leaky_relu
        elif reward_activation == "sigmoid":
            self.reward_activation = torch.sigmoid

        # TODO integrate src attention
        if attention == "bahdanau":
            self.attention = BahdanauAttention(hidden_size=hidden_size,
                                               key_size=encoder.output_size,
                                               query_size=hidden_size)
        elif attention == "luong":
            self.attention = LuongAttention(hidden_size=hidden_size,
                                            key_size=encoder.output_size)
        else:
            raise ValueError("Unknown attention mechanism: %s" % attention)

        if freeze:
            for n, p in self.named_parameters():
                print("Not training {}".format(n))
                p.requires_grad = False

        self.hidden_dropout = torch.nn.Dropout(p=hidden_dropout, inplace=False)


    def _apply_rnn(self, input):
        """
        RNN to read a sequence of embedded symbols (input is reversed)
        :param input:
        :return: hidden states, batch x input.shape(1) x rnn_size
        """
        # apply dropout ot the rnn input (embedded decoder predictions)
        x = self.rnn_input_dropout(input)  # batch x time x embed
        # run through rnn (backwards)
        hidden = None
        rnn_outputs = []
        for t in range(x.shape[1]):
            x_i = x[:, t, :].unsqueeze(1)
            rnn_output, hidden = self.rnn(x_i, hx=hidden)  # batch x 1 x hidden
            rnn_outputs.append(rnn_output.squeeze(1))
        rnn_outputs = torch.stack(rnn_outputs, dim=1)
        return rnn_outputs

    def forward(self, reversed_input, y_length, mask, y_states,
                encoder_output, src_mask, gold_rewards=None):
        """
        Reads the decoder output backwards,
        combines it with decoder hidden states
        and predicts edits to hidden states.

        :param reversed_input: embedded, reversed decoder predictions
        :param y_length:
        :param mask:
        :param y_states: MT decoder hidden states
        :param encoder_output: encoder outputs, keys/values for attention
        :param src_mask: mask on the encoder outputs
        :return:
        """
        # TODO make use of length and mask?
        # TODO could do without flip by modifying iteratior in apply_rnn
        rnn_outputs = self._apply_rnn(input=reversed_input)
        #print("bw rnn output", rnn_outputs.shape)  # batch x time x hidden

        # Flip order back
        rnn_outputs = torch.flip(rnn_outputs, dims=[1])

        # pre-compute projected encoder outputs
        # (the "keys" for the attention mechanism)
        # this is only done for efficiency
        if hasattr(self.attention, "compute_proj_keys"):
            self.attention.compute_proj_keys(encoder_output)

        # concat with y_states
        comb_states = torch.cat([y_states, rnn_outputs], dim=2)  # batch x time x decoder.hidden_size+hidden

        # now make a prediction for every time step with rnn
        hidden = None
        corr_outputs = []
        reward_outputs = []
        attention_probs = []
        batch_size = comb_states.shape[0]
        with torch.no_grad():
            corr_prev_pred = comb_states.new_zeros(batch_size, 1,
                                                   self.output_size)
            prev_att_vector = comb_states.new_zeros(
                [batch_size, 1, self.hidden_size])

        for t in range(reversed_input.shape[1]):
            comb_i = comb_states[:, t, :].unsqueeze(1)
            #print(comb_i.shape)
            # feed in both previous prediction and combination of states
            # and previous attention vector
            # rnn_input = torch.cat([prev_embed, prev_att_vector], dim=2)
            input_i = torch.cat([comb_i, corr_prev_pred, prev_att_vector],
                                dim=2)
            # TODO might add layers here to make rnn smaller
            rnn_output, hidden = self.output_rnn(input_i, hx=hidden)
            corr_prev_pred = self.corr_activation(
                self.corr_output_layer(rnn_output))
            #reward_prev_pred = self.reward_activation(
            #    self.reward_output_layer(rnn_output))
            reward_output = self.reward_output_layer(rnn_output)

            reward_pred = reward_output.argmax(-1).float().unsqueeze(-1)

            # use new (top) decoder layer as attention query
            if isinstance(hidden, tuple):
                query = hidden[0][-1].unsqueeze(1)
            else:
                query = hidden[-1].unsqueeze(1)  # [#layers, B, D] -> [B, 1, D]

            # compute context vector using attention mechanism
            # only use last layer for attention mechanism
            # key projections are pre-computed
            context, att_probs = self.attention(
                query=query, values=encoder_output, mask=src_mask)

            # return attention vector (Luong)
            # combine context with decoder hidden state before prediction
            att_vector_input = torch.cat([query, context], dim=2)
            att_vector_input = self.hidden_dropout(att_vector_input)

            # batch x 1 x 2*enc_size+hidden_size
            prev_att_vector = torch.tanh(self.att_vector_layer(att_vector_input))
            # TODO could also feed correct reward as history
            # during training use gold reward
            if gold_rewards is not None:
                curr_reward = gold_rewards[:, t].unsqueeze(-1)
            # during validation use predicted reward
            else:
                curr_reward = reward_pred

            corr_prev_pred = torch.mul(corr_prev_pred, 1-curr_reward)
            corr_outputs.append(corr_prev_pred.squeeze(1))
            reward_outputs.append(reward_output.squeeze(1))
            attention_probs.append(att_probs.squeeze(1))
        corr_outputs = torch.stack(corr_outputs, dim=1)
        reward_outputs = torch.stack(reward_outputs, dim=1)
        attention_probs = torch.stack(attention_probs, dim=1)
        #print("corr_outputs", corr_outputs.shape)

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

        return corr_outputs, reward_outputs, attention_probs




