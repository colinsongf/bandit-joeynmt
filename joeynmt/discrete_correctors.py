# coding: utf-8
import torch
import torch.nn as nn
from torch import Tensor
from joeynmt.attention import BahdanauAttention, LuongAttention, AttentionMechanism
from joeynmt.encoders import Encoder
from joeynmt.decoders import RecurrentDecoder

# TODO make general decoder class
class Decoder(nn.Module):
    pass


class RecurrentDiscreteCorrector(RecurrentDecoder):
    """A conditional RNN decoder with attention."""

    def __init__(self,
                 type: str = "gru",
                 emb_size: int = 0,
                 hidden_size: int = 0,
                 encoder: Encoder = None,
                 attention: str = "bahdanau",
                 num_layers: int = 0,
                 vocab_size: int = 0,
                 dropout: float = 0.,
                 hidden_dropout: float = 0.,
                 bridge: bool = False,
                 input_feeding: bool = True,
                 freeze: bool = False,
                 prev_hidden_size: int = 0,
                 bidirectional: bool = True,
                 reward_coeff: float = 0,
                 shift_rewards: bool = False,
                 **kwargs):
        """
        Create a recurrent decoder.
        If `bridge` is True, the decoder hidden states are initialized from a
        projection of the encoder states, else they are initialized with zeros.

        :param type:
        :param emb_size:
        :param hidden_size:
        :param encoder:
        :param attention:
        :param num_layers:
        :param vocab_size:
        :param dropout:
        :param hidden_dropout:
        :param bridge:
        :param input_feeding:
        :param freeze:
        :param kwargs:
        """
        super(RecurrentDiscreteCorrector, self).__init__(
                type=type,
                emb_size=emb_size,
                hidden_size=hidden_size,
                encoder=encoder,
                attention=attention,
                num_layers=num_layers,
                vocab_size=vocab_size,
                dropout=dropout,
                hidden_dropout=hidden_dropout,
                bridge=bridge,
                input_feeding=input_feeding,
                freeze=freeze,
                add_input_size=prev_hidden_size+hidden_size,
                **kwargs)
        # additional input:
        # - hidden state from previous decoder
        # - backwards rnn hidden state

        # what's the difference? additional rnn to read in produced seq backwards
        # also receives hidden states from first decoder
        # maybe: additional attention over first decoder's hidden states
        # predict reward in every step

        # backwards RNN (could also be bidirectional)
        # reads in the produced words
        rnn = nn.GRU if type == "gru" else nn.LSTM

        self.bw_rnn = rnn(emb_size, hidden_size//(2 if bidirectional else 1),
                          num_layers,
                          batch_first=True, bidirectional=bidirectional,
                          dropout=dropout if num_layers > 1 else 0.)
        # predict a reward
        # TODO binary?
        self.reward_output_layer = nn.Linear(hidden_size, 1)

        self.shift_rewards = shift_rewards
        self.reward_coeff = reward_coeff

    def _read_rnn(self, input):
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
        for t in range(x.shape[1]-1, -1, -1):  # backwards iteration
            x_i = x[:, t, :].unsqueeze(1)
            rnn_output, hidden = self.bw_rnn(x_i, hx=hidden)  # batch x 1 x hidden
            rnn_outputs.append(rnn_output.squeeze(1))
        rnn_outputs = torch.stack(rnn_outputs, dim=1).flip(1)
        return rnn_outputs

    def decoder_bridge(self, prev_states, prev_outputs):
        """
        Bridge to previous decoder

        :param prev_states: hidden states of previous decoder
        :param prev_outputs: embedded outputs of previous decoder
        :return:
        """
        # read in decoded seq with backwards RNN
        rnn_output = self._read_rnn(prev_outputs)
        # concatenate with decoder hidden states
        comb_states = torch.cat([prev_states, rnn_output],
                                dim=2)  # batch x time x decoder.hidden_size+hidden
        return comb_states

    def forward(self, trg_embed, encoder_output, encoder_hidden,
                comb_states,
                src_mask, unrol_steps, hidden=None, prev_att_vector=None):
        """
         Unroll the decoder one step at a time for `unrol_steps` steps.

        :param trg_embed:
        :param encoder_output:
        :param prev_decoder_hidden: decoder hidden states
        :param decoder_seq: decoder output sequence (embedded)
        :param encoder_hidden:
        :param src_mask:
        :param unrol_steps:
        :param hidden:
        :param prev_att_vector:
        :return:
        """

        # run a normal decoder, but with additional input in every time step
        outputs, hidden, att_probs, att_vectors = \
            super(RecurrentDiscreteCorrector, self).forward(
                trg_embed=trg_embed, encoder_output=encoder_output,
                encoder_hidden=encoder_hidden, src_mask=src_mask,
                unrol_steps=unrol_steps, hidden=hidden,
                prev_att_vector=prev_att_vector, add_inputs=comb_states)

        rewards = self.reward_output_layer(att_vectors)
        return outputs, hidden, att_probs, att_vectors, rewards

    def __repr__(self):
        return "RecurrentDiscreteCorrector(rnn=%r, attention=%r)" % (
            self.rnn, self.attention)
