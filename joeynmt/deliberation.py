# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from joeynmt.encoders import Encoder
from joeynmt.decoders import Decoder, RecurrentDeliberationDecoder
from joeynmt.embeddings import Embeddings
from joeynmt.vocabulary import Vocabulary
from joeynmt.constants import PAD_TOKEN, EOS_TOKEN, BOS_TOKEN
from joeynmt.helpers import arrays_to_sentences
from joeynmt.search import greedy, beam_search

class DeliberationModel(nn.Module):
    """
    Deliberation Model class
    """

    def __init__(self,
                 name: str = "my_model",
                 encoder: Encoder = None,
                 decoder1: Decoder = None,
                 decoder2: RecurrentDeliberationDecoder = None,
                 src_embed: Embeddings = None,
                 trg_embed: Embeddings = None,
                 src_vocab: Vocabulary = None,
                 trg_vocab: Vocabulary = None):
        """
        Create a new encoder-decoder-decoder (deliberation) model
        The first decoder attends the encoder,
        the second attends the first decoder and the encoder
        :param name:
        :param encoder:
        :param decoder:
        :param src_embed:
        :param trg_embed:
        :param src_vocab:
        :param trg_vocab:
        """
        super(DeliberationModel, self).__init__()

        self.name = name
        self.src_embed = src_embed
        self.trg_embed = trg_embed
        self.encoder = encoder
        self.decoder1 = decoder1
        self.decoder2 = decoder2
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.bos_index = self.trg_vocab.stoi[BOS_TOKEN]
        self.pad_index = self.trg_vocab.stoi[PAD_TOKEN]
        self.eos_index = self.trg_vocab.stoi[EOS_TOKEN]

    def forward(self, src, trg_input, src_mask, src_lengths):
        """
        Take in and process masked src and target sequences.
        Use the encoder hidden state to initialize the decoder
        The encoder outputs are used for attention
        :param src:
        :param trg_input:
        :param src_mask:
        :param src_lengths:
        :return: decoder outputs
        """
        encoder_output, encoder_hidden, lm_output = self.encode(src=src,
                                                     src_length=src_lengths,
                                                     src_mask=src_mask)
        unrol_steps = trg_input.size(1)
        dec1_outputs, dec1_hidden, dec1_att_probs, dec1_att_vectors = \
            self.decode1(encoder_output=encoder_output,
                           encoder_hidden=encoder_hidden,
                           src_mask=src_mask, trg_input=trg_input,
                           unrol_steps=unrol_steps)
        # TODO pass on outputs or hidden? paper says hidden states
        # what if we use attentional vectors instead? (would also contain context)
        d1_states = dec1_att_vectors.detach()  # don't backprop from d2 through d1
        # TODO original paper uses beam seach with k=2 here, let's use greedy search for efficiency
        d1_greedy = torch.argmax(dec1_outputs, dim=-1).detach()  # don't backprop through here
        d1_predictions = self.trg_embed(d1_greedy) # TODO backprop through embeddings though?
        # TODO is there a target mask?? pro: ignore after </s>, con: full knowledge
        # for now: just use ones
        # TODO shapes
        trg_mask = d1_states.new_ones(d1_states.shape[0], 1, d1_states.shape[1]).byte()

        dec2_outputs = self.decode2(encoder_output=encoder_output,
                                    encoder_hidden=encoder_hidden,
                                    src_mask=src_mask, trg_input=trg_input,
                                    unrol_steps=unrol_steps,
                                    trg_mask=trg_mask,
                                    d1_states=d1_states,
                                    d1_predictions=d1_predictions)

        return dec1_outputs, dec2_outputs, d1_greedy, lm_output

    def encode(self, src, src_length, src_mask):
        """
        Encodes the source sentence.
        TODO adapt to transformer

        :param src:
        :param src_length:
        :param src_mask:
        :return:
        """
        return self.encoder(self.src_embed(src), src_length, src_mask)

    def decode1(self, encoder_output, encoder_hidden, src_mask, trg_input,
               unrol_steps, decoder_hidden=None):
        """
        Decode, given an encoded source sentence.
        # TODO adapt to transformer

        :param encoder_output:
        :param encoder_hidden:
        :param src_mask:
        :param trg_input:
        :param unrol_steps:
        :param decoder_hidden:
        :return: decoder outputs
        """
        return self.decoder1(trg_embed=self.trg_embed(trg_input),
                            encoder_output=encoder_output,
                            encoder_hidden=encoder_hidden,
                            src_mask=src_mask,
                            unrol_steps=unrol_steps,
                            hidden=decoder_hidden)

    def decode2(self, encoder_output, encoder_hidden, src_mask, trg_input,
                unrol_steps, d1_predictions, d1_states, trg_mask,
                decoder_hidden=None):
        return self.decoder2(trg_embed=self.trg_embed(trg_input),
                             encoder_output=encoder_output,
                             encoder_hidden=encoder_hidden,
                             d1_predictions=d1_predictions,
                             d1_states=d1_states,
                             trg_mask=trg_mask,
                             src_mask=src_mask,
                             unrol_steps=unrol_steps,
                             hidden=decoder_hidden)

    def get_loss_for_batch(self, batch, criterion):
        """
        Compute non-normalized loss and number of tokens for a batch
        :param batch:
        :param criterion:
        :return:
        """
        d1_output, d2_output, d1_predictions, lm_output = self.forward(
            src=batch.src, trg_input=batch.trg_input,
            src_mask=batch.src_mask, src_lengths=batch.src_lengths)

        #print(len(d1_output))
        #d1_out, d1_hidden, d1_att_probs, _ = d1_output
        # 2nd decoder has additional attention on 1st decoder's output
        d2_out, d2_hidden, d2_src_att_probs, d2_trg_att_probs, _ = d2_output

        # https://pytorch.org/docs/stable/notes/autograd.html
        # 1. d1: stop_grad(log p_d2(y|y_d1, x)) * grad(log p_d1(y_d1|x))
        # 2. d2: grad(log p_d2(y|y_d1, x))
        # 3. shared encoder: sum of both
        # grad( log p_d2(y| y_d1, x)) + grad(log p_d1(y_d1 | x) * no_grad (log p_d2(y | y_d1, x))

        # d1
        d1_log_probs = F.log_softmax(d1_output, dim=-1)
        # pretend targets are sampled
        d1_pred_logprobs = criterion(input=d1_log_probs.contiguous().view(-1, d1_log_probs.size(-1)), target=d1_predictions.contiguous().view(-1))

        # d2
        d2_log_probs = F.log_softmax(d2_out, dim=-1)
        # standard xent
        d2_ref_logprobs = criterion(input=d2_log_probs.contiguous().view(-1, d2_log_probs.size(-1)), target=batch.trg.contiguous().view(-1))

        # don't backprop through d2 for d1's parameters
        d1_loss = d1_pred_logprobs * d2_ref_logprobs.detach()
        d2_loss = d2_ref_logprobs

        batch_loss = d1_loss + d2_loss

        # add lm loss (optional)
        if lm_output is not None:
            predictions = arrays_to_sentences(arrays=torch.argmax(lm_output[:, :-1, :], dim=-1),
                                                vocabulary=self.src_vocab,
                                                cut_at_eos=False)  # batch x length
            correct = arrays_to_sentences(arrays=batch.src[:, 1:],
                                                vocabulary=self.src_vocab,
                                                cut_at_eos=False)
            print("input",arrays_to_sentences(arrays=batch.src,
                                                vocabulary=self.src_vocab,
                                                cut_at_eos=False)[0] )
            print("correct", correct[0])
            print("predicted", predictions[0])
            lm_logprobs = F.log_softmax(lm_output, dim=-1)
            # shift inputs to the left for loss targets, ignore last hidden state
            lm_loss = criterion(
                input=lm_logprobs[:, :-1].contiguous().view(-1,
                                                            lm_logprobs.size(
                                                                -1)),
                target=batch.src[:, 1:].contiguous().view(-1))
            #print("MT loss:", batch_loss.data.cpu().numpy(), "LM loss:",
            #      lm_loss.data.cpu().numpy(), "weighted LM loss:", self.encoder.lm_task*lm_loss.data.cpu().numpy())
            batch_loss += self.encoder.lm_task * lm_loss  # weighted

        return batch_loss


    def run_batch(self, batch, max_output_length, beam_size, beam_alpha):
        """
        Get outputs and attentions scores for a given batch
        :param batch:
        :param max_output_length:
        :param beam_size:
        :param beam_alpha:
        :return:
        """

        encoder_output, encoder_hidden, lm_output = self.encode(
            batch.src, batch.src_lengths,
            batch.src_mask)

        # if maximum output length is not globally specified, adapt to src len
        if max_output_length is None:
            max_output_length = int(max(batch.src_lengths.cpu().numpy()) * 1.5)

        # TODO for inference use first decoder? (yes for our model, but no for deliberation
        # TODO make greedy and beam search handle multiple decoders
        # TODO get outputs for both decoders!
        # greedy decoding
        if beam_size == 0:
            stacked_output1, stacked_attention_scores1 = greedy(
                encoder_hidden=encoder_hidden, encoder_output=encoder_output,
                src_mask=batch.src_mask, embed=self.trg_embed,
                bos_index=self.bos_index, decoder=self.decoder1,
                max_output_length=max_output_length)
            # batch, time, max_src_length
        else:  # beam size
            stacked_output1, stacked_attention_scores1 = \
                beam_search(size=beam_size, encoder_output=encoder_output,
                            encoder_hidden=encoder_hidden,
                            src_mask=batch.src_mask, embed=self.trg_embed,
                            max_output_length=max_output_length,
                            alpha=beam_alpha, eos_index=self.eos_index,
                            pad_index=self.pad_index, bos_index=self.bos_index,
                            decoder=self.decoder1)

        return stacked_output1, stacked_attention_scores1, stacked_output2, stacked_attention_scores2

    def __repr__(self):
        """
        String representation: a description of encoder, decoder and embeddings
        :return:
        """
        return "%s(\n" \
               "\tencoder=%r,\n" \
               "\tdecoder=%r,\n" \
               "\tsrc_embed=%r,\n" \
               "\ttrg_embed=%r)" % (
                   self.__class__.__name__, str(self.encoder),
                   str(self.decoder),
                   self.src_embed, self.trg_embed)

    def log_parameters_list(self, logging_function):
        """
        Write all parameters (name, shape) to the log.
        :param logging_function:
        :return:
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        n_params = sum([np.prod(p.size()) for p in model_parameters])
        logging_function("Total params: %d" % n_params)
        for name, p in self.named_parameters():
            if p.requires_grad:
                logging_function("%s : %s" % (name, list(p.size())))
