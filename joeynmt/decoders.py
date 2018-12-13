# coding: utf-8
import torch
import torch.nn as nn
from torch import Tensor
from joeynmt.attention import BahdanauAttention, LuongAttention, AttentionMechanism
from joeynmt.encoders import Encoder
from joeynmt.maxout import Maxout

# TODO make general decoder class
class Decoder(nn.Module):
    pass


class RecurrentDecoder(Decoder):
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
                 output_layer_type: str = "luong",
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
        :param kwargs:
        :param output_layer_type: type of output layer. Options: luong, weiss, simple, deep, maxout
        """

        super(RecurrentDecoder, self).__init__()
        print("OUTPUT LAYER", output_layer_type)

        self.rnn_input_dropout = torch.nn.Dropout(p=dropout, inplace=False)
        self.type = type
        self.hidden_dropout = torch.nn.Dropout(p=hidden_dropout, inplace=False)
        self.hidden_size = hidden_size

        rnn = nn.GRU if type == "gru" else nn.LSTM

        self.input_feeding = input_feeding
        if self.input_feeding: # Luong-style
            # combine embedded prev word +attention vector before feeding to rnn
            self.rnn_input_size = emb_size + hidden_size
        else:
            # just feed prev word embedding
            self.rnn_input_size = emb_size

        # the decoder RNN
        self.rnn = rnn(self.rnn_input_size, hidden_size, num_layers,
                       batch_first=True,
                       dropout=dropout if num_layers > 1 else 0.)
        self.output_layer_type = output_layer_type
        # we use the same output layer input size for all models (except for weiss)
        self.output_layer_input_size = hidden_size

        self.att_vector_layer = nn.Linear(
            hidden_size + encoder.output_size, hidden_size, bias=True)

        if self.output_layer_type.lower() == "luong":
            # combine output with context vector before output layer (Luong-style)
            # W_o \text{tanh}(W_i[c_t, s_t])
            # no additional parameters needed, already used for attention vector
            pass

        elif self.output_layer_type.lower() == "simple":
            # W_o s_t
            # no additional parameters needed
            pass
        elif self.output_layer_type.lower() in "maxout":
            # maxout: W_o\, \text{maxout}(W_i[s_{t-1}, Ey_{t-1}, c_t])
            # create maxout layer
            self.maxout_layer = Maxout(
                d_in=hidden_size + emb_size + encoder.output_size,
                d_out=hidden_size, pool_size=2)

        elif self.output_layer_type.lower() == "deep":
            # deep: W_o\, \text{tanh}(W_i[s_t, Ey_{t-1}, c_t])
            self.pre_output_layer = nn.Linear(
                hidden_size + emb_size + encoder.output_size,
                hidden_size, bias=True)

        elif self.output_layer_type.lower() == "weiss":
            # W_o [o_t , c_t]
            # no additional params needed, but output layer has to get increased
            self.output_layer_input_size = hidden_size + encoder.output_size

        self.output_layer = nn.Linear(self.output_layer_input_size,
                                      vocab_size, bias=False)
        self.output_size = vocab_size

        if attention == "bahdanau":
            self.attention = BahdanauAttention(hidden_size=hidden_size,
                                               key_size=encoder.output_size,
                                               query_size=hidden_size)
        elif attention == "luong":
            self.attention = LuongAttention(hidden_size=hidden_size,
                                            key_size=encoder.output_size)
        else:
            raise ValueError("Unknown attention mechanism: %s" % attention)

        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # to initialize from the final encoder state of last layer
        self.bridge = bridge
        if self.bridge:
            self.bridge_layer = nn.Linear(
                encoder.output_size, hidden_size, bias=True)

    def _forward_step(self,
                      prev_embed: Tensor = None,
                      prev_att_vector: Tensor = None,  # context or att vector
                      encoder_output: Tensor = None,
                      src_mask: Tensor = None,
                      hidden: Tensor = None):
        """
        Perform a single decoder step (1 word)

        :param prev_embed:
        :param prev_att_vector:
        :param encoder_output:
        :param src_mask:
        :param hidden:
        :return:
        """

        # loop:
        # 1. rnn input = concat(prev_embed, prev_output [possibly empty])
        # 2. update RNN with rnn_input
        # 3. calculate attention and context/attention vector
        # 4. repeat

        # update rnn hidden state
        if self.input_feeding:
            rnn_input = torch.cat([prev_embed, prev_att_vector], dim=2)
        else:
            rnn_input = prev_embed

        rnn_input = self.rnn_input_dropout(rnn_input)

        # rnn_input: batch x 1 x emb+2*enc_size
        _, hidden = self.rnn(rnn_input, hidden)

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
        att_vector = torch.tanh(self.att_vector_layer(att_vector_input))

        # output: batch x 1 x dec_size
        return att_vector, hidden, att_probs, context

    def forward(self, trg_embed, encoder_output, encoder_hidden,
                src_mask, unrol_steps, hidden=None, prev_att_vector=None):
        """
         Unroll the decoder one step at a time for `unrol_steps` steps.

        :param trg_embed:
        :param encoder_output:
        :param encoder_hidden:
        :param src_mask:
        :param unrol_steps:
        :param hidden:
        :param prev_att_vector:
        :return:
        """

        # here we store all intermediate attention vectors (used for prediction)
        att_vectors = []
        att_probs = []
        hidden_vectors = []
        att_contexts = []

        # initialize decoder hidden state from final encoder hidden state
        if hidden is None:
            hidden = self.init_hidden(encoder_hidden)

        if self.output_layer_type == "maxout":
            # use previous hidden state, not current hidden state for prediction
            if isinstance(hidden, tuple):
                hidden_vectors.append(hidden[0][-1].unsqueeze(1))
            else:
                hidden_vectors.append(hidden[-1].unsqueeze(1))


        # pre-compute projected encoder outputs
        # (the "keys" for the attention mechanism)
        # this is only done for efficiency
        if hasattr(self.attention, "compute_proj_keys"):
            self.attention.compute_proj_keys(encoder_output)



        batch_size = encoder_output.size(0)

        if prev_att_vector is None:
            with torch.no_grad():
                prev_att_vector = encoder_output.new_zeros(
                    [batch_size, 1, self.hidden_size])

        # unroll the decoder RN N for max_len steps
        for i in range(unrol_steps):
            prev_embed = trg_embed[:, i].unsqueeze(1)  # batch, 1, emb
            prev_att_vector, hidden, att_prob, att_context = self._forward_step(
                prev_embed=prev_embed,
                prev_att_vector=prev_att_vector,
                encoder_output=encoder_output,
                src_mask=src_mask,
                hidden=hidden)
            att_vectors.append(prev_att_vector)
            att_probs.append(att_prob)
            if isinstance(hidden, tuple):
                hidden_vectors.append(hidden[0][-1].unsqueeze(1))
            else:
                hidden_vectors.append(hidden[-1].unsqueeze(1))
            att_contexts.append(att_context)

        att_vectors = torch.cat(att_vectors, dim=1)
        att_probs = torch.cat(att_probs, dim=1)
        hidden_vectors = torch.cat(hidden_vectors, dim=1)
        att_contexts = torch.cat(att_contexts, dim=1)
        # att_probs: batch, max_len, src_length

        if self.output_layer_type == "luong":
            outputs = self.output_layer(att_vectors)

        elif self.output_layer_type == "simple":
            outputs = self.output_layer(hidden_vectors)

        elif self.output_layer_type == "maxout":
            # TODO they also use the previous hidden state in attention, not the current one
            outputs = self.output_layer(
                        self.maxout_layer( # remove last hidden state (shifted)
                                torch.cat([hidden_vectors[:, :-1, :],
                                           trg_embed, att_contexts],
                                          dim=-1)
                        )
                    )

        elif self.output_layer_type == "deep":
            # TODO during testing, what's trg_embed? need to be predictions!
            outputs = self.output_layer(
                torch.tanh(
                    self.pre_output_layer(
                        torch.cat([hidden_vectors, trg_embed, att_contexts],
                                  dim=-1)
                    )))

        elif self.output_layer_type == "weiss":
            outputs = self.output_layer(
                torch.cat([hidden_vectors, att_contexts], dim=-1))
        # outputs: batch, max_len, vocab_size
        return outputs, hidden, att_probs, att_vectors

    def init_hidden(self, encoder_final):
        """
        Returns the initial decoder state,
        conditioned on the final encoder state of the last encoder layer.

        :param encoder_final:
        :return:
        """
        batch_size = encoder_final.size(0)

        # for multiple layers: is the same for all layers
        if self.bridge and encoder_final is not None:
            h = torch.tanh(
                self.bridge_layer(encoder_final)).unsqueeze(0).repeat(
                self.num_layers, 1, 1)  # num_layers x batch_size x hidden_size

        else:  # initialize with zeros
            with torch.no_grad():
                h = encoder_final.new_zeros(self.num_layers, batch_size,
                                            self.hidden_size)

        return (h, h) if isinstance(self.rnn, nn.LSTM) else h

    def __repr__(self):
        return "RecurrentDecoder(rnn=%r, attention=%r)" % (
            self.rnn, self.attention)
