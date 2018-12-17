# coding: utf-8
import torch
import torch.nn as nn
from torch import Tensor
from joeynmt.attention import BahdanauAttention, LuongAttention, AttentionMechanism
from joeynmt.encoders import Encoder


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
                 freeze: bool = False,
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
        """

        super(RecurrentDecoder, self).__init__()

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

        # combine output with context vector before output layer (Luong-style)
        self.att_vector_layer = nn.Linear(
            hidden_size + encoder.output_size, hidden_size, bias=True)

        self.output_layer = nn.Linear(hidden_size, vocab_size, bias=False)
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

        if freeze:
            for n, p in self.named_parameters():
                print("Not training {}".format(n))
                p.requires_grad=False

    def _forward_step(self,
                      prev_embed: Tensor = None,
                      prev_att_vector: Tensor = None,  # context or att vector
                      encoder_output: Tensor = None,
                      src_mask: Tensor = None,
                      hidden: Tensor = None):
        """Perform a single decoder step (1 word)"""

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
            query=query, keys=encoder_output, mask=src_mask)

        # return attention vector (Luong)
        # combine context with decoder hidden state before prediction
        att_vector_input = torch.cat([query, context], dim=2)
        att_vector_input = self.hidden_dropout(att_vector_input)

        # batch x 1 x 2*enc_size+hidden_size
        att_vector = torch.tanh(self.att_vector_layer(att_vector_input))

        # output: batch x 1 x dec_size
        return att_vector, hidden, att_probs

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

        # initialize decoder hidden state from final encoder hidden state
        if hidden is None:
            hidden = self.init_hidden(encoder_hidden)

        # pre-compute projected encoder outputs
        # (the "keys" for the attention mechanism)
        # this is only done for efficiency
        if hasattr(self.attention, "compute_proj_keys"):
            self.attention.compute_proj_keys(encoder_output)

        # here we store all intermediate attention vectors (used for prediction)
        att_vectors = []
        att_probs = []
        hidden_vectors = []

        # FIXME support not-input feeding
        batch_size = encoder_output.size(0)

        if prev_att_vector is None:
            with torch.no_grad():
                prev_att_vector = encoder_output.new_zeros(
                    [batch_size, 1, self.hidden_size])

        # unroll the decoder RN N for max_len steps
        for i in range(unrol_steps):
            prev_embed = trg_embed[:, i].unsqueeze(1)  # batch, 1, emb
            prev_att_vector, hidden, att_prob = self._forward_step(
                prev_embed=prev_embed,
                prev_att_vector=prev_att_vector,
                encoder_output=encoder_output,
                src_mask=src_mask,
                hidden=hidden)
            att_vectors.append(prev_att_vector)
            att_probs.append(att_prob)
            if isinstance(hidden, tuple):  # lstm
                hidden_vector = hidden[0][-1].unsqueeze(1)
            else:
                hidden_vector = hidden[-1].unsqueeze(1)  # [#layers, B, D] -> [B, 1, D]
            hidden_vectors.append(hidden_vector)

        att_vectors = torch.cat(att_vectors, dim=1)
        att_probs = torch.cat(att_probs, dim=1)
        hidden_vectors = torch.cat(hidden_vectors, dim=1)
        # att_probs: batch, max_len, src_length
        outputs = self.output_layer(att_vectors)
        # outputs: batch, max_len, vocab_size
        return outputs, hidden, att_probs, att_vectors, hidden_vectors

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


class RecurrentDeliberationDecoder(Decoder):
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
                 **kwargs):
        """
        Create a recurrent deliberation decoder.
        In contrast to the standard 1st decoder, this one receives the output
        from the 1st decoder as input as well as the encoder outputs and has a
        joint attention over both.

        input to the attention: [hidden state of previous decoder, predicted word]
        input to rnn: [previous prediction, source context vector, target context vector], prev state
        att vector feeding: add target context to it

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
        :param freeze: do not update the parameters of this decoder
        :param kwargs:
        """

        super(RecurrentDeliberationDecoder, self).__init__()

        self.rnn_input_dropout = torch.nn.Dropout(p=dropout, inplace=False)
        self.type = type
        self.hidden_dropout = torch.nn.Dropout(p=hidden_dropout, inplace=False)
        self.hidden_size = hidden_size

        rnn = nn.GRU if type == "gru" else nn.LSTM

        # learns to combine src and trg attention contexts
        self.context_comb_layer = nn.Linear(
            encoder.output_size + hidden_size + emb_size, encoder.output_size,
        bias=True)

        self.input_feeding = input_feeding
        if self.input_feeding:
            # combine hidden state and attentional context before feeding to rnn
            #self.comb_att_vector_layer = nn.Linear(
            #    hidden_size + encoder.output_size + hidden_size + emb_size,
            #    hidden_size, bias=True)
            self.comb_att_vector_layer = nn.Linear(
                hidden_size + encoder.output_size, hidden_size, bias=True
            )
            self.rnn_input_size = emb_size + hidden_size
        else: # TODO does this make sense?
            # just feed prev word embedding
            self.rnn_input_size = emb_size



        # the decoder RNN
        self.rnn = rnn(self.rnn_input_size, hidden_size, num_layers,
                       batch_first=True,
                       dropout=dropout if num_layers > 1 else 0.)

        self.output_layer = nn.Linear(hidden_size, vocab_size, bias=False)
        self.output_size = vocab_size

        if attention == "bahdanau":
            self.src_attention = BahdanauAttention(hidden_size=hidden_size,
                                               key_size=encoder.output_size,
                                               query_size=hidden_size)
            self.d1_attention = BahdanauAttention(hidden_size=hidden_size,
                                               key_size=hidden_size+emb_size,
                                               query_size=hidden_size)
        elif attention == "luong":
            self.src_attention = LuongAttention(hidden_size=hidden_size,
                                            key_size=encoder.output_size)
            self.d1_attention = LuongAttention(hidden_size=hidden_size,
                                            key_size=hidden_size+emb_size)
        else:
            raise ValueError("Unknown attention mechanism: %s" % attention)

        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # to initialize from the final encoder state of last layer
        self.bridge = bridge
        if self.bridge:
            self.bridge_layer = nn.Linear(
                encoder.output_size, hidden_size, bias=True)

        if freeze:
            for n, p in self.named_parameters():
                print("Not training {}".format(n))
                p.requires_grad=False

    def _forward_step(self,
                      prev_embed: Tensor = None,
                      prev_comb_att_vector: Tensor = None,
                      encoder_output: Tensor = None,
                      d1_states: Tensor = None,
                      d1_predictions: Tensor = None,
                      src_mask: Tensor = None,
                      trg_mask: Tensor = None,
                      hidden: Tensor = None):
        """Perform a single decoder step (1 word)"""

        # FIXME trying new loop:
        # 1. rnn input = concat(prev_embed, prev_output [possibly empty])
        # 2. update RNN with rnn_input
        # 3. calculate attention and context/attention vector
        # 4. repeat

        # update rnn hidden state
        # if using input feeding, prev_context is the previous attention vector
        # otherwise prev_context is the previous context vector
        # FIXME if not input feeding, do not input context here
        if self.input_feeding:
            rnn_input = torch.cat([prev_embed, prev_comb_att_vector], dim=2)
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
        #print("src_mask", src_mask.shape)
        src_context, src_att_probs = self.src_attention(
            query=query, values=encoder_output, mask=src_mask)
        #print("SRC MASK", src_mask)
        #print("SRC PROB", src_att_probs)

        d1_att_values = torch.cat([d1_states, d1_predictions], dim=-1)
        #print("d1_att_values", d1_att_values.shape)
        #print("trg_mask", trg_mask.shape)
        d1_context, d1_att_probs = self.d1_attention(query=query,
                                                     values=d1_att_values,
                                                     mask=trg_mask)
        #print("p1_att_probs", d1_att_probs.shape)

        # return attention vector
        # combine context with decoder hidden state before prediction
        # d2: combine both attention contexts
        #print("query", query.shape)
        #print("src_context", src_context.shape)
        #print("d1_context", d1_context.shape)
        # query: batch x 1 x hidden
        # src_context: batch x 1 x 2*hidden
        # d1_context: batch x 1 x (hidden+emb)

        # learn to combine src and d1 context
        comb_context = torch.tanh(
            self.context_comb_layer(
                torch.cat([src_context, d1_context], dim=2)))

        #comb_att_vector_input = torch.cat([query, src_context, d1_context],
        #                                  dim=2)
        comb_att_vector_input = torch.cat([query, comb_context], dim=2)
        #print("comb_att_vector_input", comb_att_vector_input.shape)
        comb_att_vector_input = self.hidden_dropout(comb_att_vector_input)

        comb_att_vector = torch.tanh(
            self.comb_att_vector_layer(comb_att_vector_input))

        # output: batch x 1 x dec_size
        return comb_att_vector, hidden, src_att_probs, d1_att_probs

    def forward(self, trg_embed, d1_predictions, d1_states,
                encoder_output, encoder_hidden,
                src_mask, trg_mask, unrol_steps, hidden=None, prev_comb_att_vector=None):
        """
         Unroll the decoder one step at a time for `unrol_steps` steps.

        :param trg_embed:
        :param d1_predictions: predictions of the previous decoder (embedded)
        :param d1_states: hidden states of the previous decoder
        :param encoder_output:
        :param encoder_hidden:
        :param src_mask:
        :param trg_mask: mask out parts of decoder1 output
        :param unrol_steps:
        :param hidden:
        :param prev_comb_att_vector: includes both attentions
        :return:
        """

        # initialize decoder hidden state from final encoder hidden state
        if hidden is None:
            hidden = self.init_hidden(encoder_hidden)

        # pre-compute projected encoder outputs
        # (the "keys" for the attention mechanism)
        # this is only done for efficiency
        if hasattr(self.src_attention, "compute_proj_keys"):
            self.src_attention.compute_proj_keys(encoder_output)

        # attention to previous decoder states and predictions
        if hasattr(self.d1_attention, "compute_proj_keys"):
            att_keys = torch.cat([d1_predictions, d1_states], dim=-1)
            self.d1_attention.compute_proj_keys(att_keys)

        # here we store all intermediate attention vectors (used for prediction)
        comb_att_vectors = []
        src_att_probs = []
        d1_att_probs = []

        batch_size = encoder_output.size(0)

        if prev_comb_att_vector is None:
            with torch.no_grad():
                prev_comb_att_vector = encoder_output.new_zeros(
                    [batch_size, 1, self.hidden_size])

        # unroll the decoder RN N for max_len steps
        # TODO compute attention values before (fix them)
        for i in range(unrol_steps):
            prev_embed = trg_embed[:, i].unsqueeze(1)  # batch, 1, emb
            prev_comb_att_vector, hidden, src_att_prob,\
            d1_att_prob = self._forward_step(
                prev_embed=prev_embed,
                prev_comb_att_vector=prev_comb_att_vector,
                encoder_output=encoder_output,
                d1_states=d1_states,
                d1_predictions=d1_predictions,
                src_mask=src_mask,
                trg_mask=trg_mask,
                hidden=hidden)
            comb_att_vectors.append(prev_comb_att_vector)
            src_att_probs.append(src_att_prob)
            d1_att_probs.append(d1_att_prob)

        comb_att_vectors = torch.cat(comb_att_vectors, dim=1)
        src_att_probs = torch.cat(src_att_probs, dim=1)
        d1_att_probs = torch.cat(d1_att_probs, dim=1)

        # combined att vector:
        # affine transformation of [state, src_context, d1_context]
        # att_probs: batch, max_len, src_length
        outputs = self.output_layer(comb_att_vectors)
        # outputs: batch, max_len, vocab_size
        return outputs, hidden, src_att_probs, d1_att_probs, comb_att_vectors

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
