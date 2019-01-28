# coding: utf-8
import torch
import torch.nn as nn
from torch import Tensor
from joeynmt.attention import BahdanauAttention, LuongAttention, AttentionMechanism
from joeynmt.encoders import Encoder
from joeynmt.decoders import RecurrentDecoder
from joeynmt.binary import BinarySigmoidLayer

# TODO make general decoder class
class Decoder(nn.Module):
    pass


class CoattentiveDiscreteCorrector(Decoder):
    """
    A conditional RNN decoder with co-attention
    """
    def __init__(self,
                 type: str = "gru",
                 emb_size: int = 0,
                 hidden_size: int = 0,
                 num_layers: int = 0,
                 vocab_size: int = 0,
                 dropout: float = 0.,
                 freeze: bool = False,
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
        super(CoattentiveDiscreteCorrector, self).__init__()
        #super(CoattentiveRecurrentDiscreteCorrector, self).__init__(
        #        type=type,
        #        emb_size=emb_size,
        #        hidden_size=hidden_size,
        #        encoder=encoder,
        #        attention=None,
        #        num_layers=num_layers,
        #        vocab_size=vocab_size,
        #        dropout=dropout,
        #        hidden_dropout=hidden_dropout,
        #        bridge=bridge,
        #        input_feeding=input_feeding,
        #        freeze=freeze,
        #        add_input_size=2*(hidden_size*(2 if bidirectional else 1)), #prev_hidden_size+hidden_size,
        #        **kwargs)
        # additional input:
        # - hidden state from previous decoder
        # - backwards rnn hidden state

        # what's the difference? additional rnn to read in produced seq backwards
        # also receives hidden states from first decoder
        # maybe: additional attention over first decoder's hidden states
        # predict reward in every step
        self.rnn_input_dropout = torch.nn.Dropout(p=dropout, inplace=False)

        # RNN to read in the produced sequence
        # reads in the produced words
        rnn = nn.GRU if type == "gru" else nn.LSTM
        self.hyp_rnn = rnn(emb_size, hidden_size,
                           num_layers,
                           batch_first=True, bidirectional=bidirectional,
                           dropout=dropout if num_layers > 1 else 0.)

        self.output_rnn = rnn(self.hyp_rnn.hidden_size*2*(2 if bidirectional else 1),
                              hidden_size, num_layers,
                              batch_first=True, bidirectional=bidirectional,
                              dropout=dropout if num_layers > 1 else 0.)
        # make classification in every step (non-recurrent)
        self.corr_output_layer = nn.Linear(hidden_size*(2 if bidirectional else 1), vocab_size, bias=True)
        # TODO use an rnn instead

        # predict a reward
        self.reward_output_layer = nn.Linear(hidden_size*(2 if bidirectional else 1), 1, bias=True)
        self.binary_layer = BinarySigmoidLayer()

        self.shift_rewards = shift_rewards
        self.reward_coeff = reward_coeff

        if freeze:
            for n, p in self.named_parameters():
                print("Not training {}".format(n))
                p.requires_grad = False

    def _apply_rnn(self, rnn, inputs):
        # Bi-RNN without length-sorted inputs
        # apply dropout ot the rnn input (embedded decoder predictions)
        x = self.rnn_input_dropout(inputs)  # batch x time x embed
        # run through rnn (backwards)
        rnn_outputs, _ = rnn(x)
        #hidden = None
        #rnn_outputs = []
        #for t in range(x.shape[1] - 1, -1, -1):  # backwards iteration
        #for t in range(x.shape[1]):
        #    x_i = x[:, t, :].unsqueeze(1)
        #    rnn_output, hidden = rnn(x_i, hx=hidden)  # batch x 1 x hidden
        #    rnn_outputs.append(rnn_output.squeeze(1))
        #rnn_outputs = torch.stack(rnn_outputs, dim=1) #.flip(1)
        return rnn_outputs

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
            rnn_output, hidden = self.hyp_rnn(x_i, hx=hidden)  # batch x 1 x hidden
            rnn_outputs.append(rnn_output.squeeze(1))
        rnn_outputs = torch.stack(rnn_outputs, dim=1).flip(1)
        return rnn_outputs

    def _coattention(self, encoder_outputs, prev_outputs, src_mask, trg_mask):
        """
        Bridge to previous decoder

        :param prev_outputs: embedded outputs of previous decoder
        :return:
        """
        # read in decoded seq with backwards RNN
        #rnn_output = self._read_rnn(prev_outputs)
        rnn_output = self._apply_rnn(self.hyp_rnn, prev_outputs)
        # concatenate with decoder hidden states
        # instead with co-attention
        #print("rnn_output", rnn_output.shape)  # batch x len x hidden
        #print("encoder_outputs", encoder_outputs.shape)  # batch x len x encoder_hidden
        l = torch.bmm(encoder_outputs, rnn_output.transpose(dim1=1, dim0=2))  # batch x src_len x trg_len
        #print("L", l.shape)
        # mask out invalid positions by filling the masked out parts with -inf, both for src and trg
        l_masked_s = torch.where(src_mask, l.transpose(dim0=1, dim1=2), l.new_full([1], float('-inf')))
        l_masked_t = torch.where(trg_mask, l, l.new_full([1], float('-inf')))
        a_s = torch.softmax(l_masked_t, dim=2)  # normalize over trg dim -> batch x src_len x trg_len
        a_t = torch.softmax(l_masked_s, dim=2)  # normalize over src dim -> batch x trg_len x src_len
        #print("a_s", a_s)
        #print("a_t", a_t)
       # c_t = torch.bmm(a_t, encoder_outputs)
        c_s = torch.bmm(a_s, rnn_output)
        #print(c_s.shape, c_t.shape)  # batch x len x hidden
        #print("cat", torch.cat([encoder_outputs, c_s], dim=2).shape)  # batch x len x 2*hidden
        c_t = torch.bmm(a_t, torch.cat([encoder_outputs, c_s], dim=2))
        assert c_t.size(0) == rnn_output.size(0)
        assert c_t.size(1) == rnn_output.size(1)
        assert c_t.size(2) == 2*rnn_output.size(2)
        #print(c_t.shape) # batch x 2*emb x trg_len
        #comb_states = torch.cat([prev_states, rnn_output],
        #                        dim=2)  # batch x time x decoder.hidden_size+hidden
        comb_states = c_t
        return comb_states, a_s, a_t  # batch x len x 2*hidden

    def forward(self, encoder_states, decoder_outputs, src_mask, trg_mask):
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
        #outputs, hidden, att_probs, att_vectors = \
        #    super(CoattentiveRecurrentDiscreteCorrector, self).forward(
        #        trg_embed=trg_embed, encoder_output=encoder_output,
        #        encoder_hidden=encoder_hidden, src_mask=src_mask,
        #        unrol_steps=unrol_steps, hidden=hidden,
        #        prev_att_vector=prev_att_vector, add_inputs=comb_states)
        comb_states, a_s, a_t = self._coattention(
            encoder_outputs=encoder_states,
            prev_outputs=decoder_outputs,
            src_mask=src_mask, trg_mask=trg_mask)
        # pass co-attentive to birnn
        rnn_states = self._apply_rnn(self.output_rnn, comb_states)
        # two output layers: one for corrections, one for rewards
        corr_logits = self.corr_output_layer(rnn_states)
        reward_logits = self.reward_output_layer(rnn_states)
        # prob of being 1: sigmoid(output)
        # prob of being 0: 1-sigmoid(output)
        slope = 1
        # TODO modify slope
        rewards = self.binary_layer((reward_logits, slope))
        return corr_logits, a_s, a_t, rewards, reward_logits

    def __repr__(self):
        # TODO make more informative
        return "CoattentiveDiscreteCorrector"




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
        #
        self.reward_output_layer = nn.Linear(hidden_size, 1, bias=True)
        self.binary_layer = BinarySigmoidLayer()

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
        reward_logits = self.reward_output_layer(att_vectors)
        # prob of being 1: sigmoid(output)
        # prob of being 0: 1-sigmoid(output)
        slope = 1
        # TODO modify slope
        rewards = self.binary_layer((reward_logits, slope))
        return outputs, hidden, att_probs, att_vectors, rewards, reward_logits

    def __repr__(self):
        return "RecurrentDiscreteCorrector(rnn=%r, attention=%r)" % (
            self.rnn, self.attention)
