# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from joeynmt.embeddings import Embeddings
from joeynmt.encoders import Encoder, RecurrentEncoder
from joeynmt.decoders import Decoder, RecurrentDecoder, \
    RecurrentDeliberationDecoder
from joeynmt.constants import PAD_TOKEN, EOS_TOKEN, BOS_TOKEN
from joeynmt.search import beam_search, greedy
from joeynmt.vocabulary import Vocabulary
from joeynmt.helpers import arrays_to_sentences
from joeynmt.deliberation import DeliberationModel
from joeynmt.initialization import initialize_model

def build_model(cfg: dict = None,
                src_vocab: Vocabulary = None,
                trg_vocab: Vocabulary = None):
    src_padding_idx = src_vocab.stoi[PAD_TOKEN]
    trg_padding_idx = trg_vocab.stoi[PAD_TOKEN]

    src_embed = Embeddings(
        **cfg["encoder"]["embeddings"], vocab_size=len(src_vocab),
        padding_idx=src_padding_idx)

    # if more 2 decoders specified: deliberation network
    decoders = [k for k in cfg.keys() if "decoder" in k]

    if cfg.get("tied_embeddings", False) \
        and src_vocab.itos == trg_vocab.itos:
        # share embeddings for src and trg
        trg_embed = src_embed
    else:
        # embeddings for all decoders
        trg_embed = Embeddings(
            **cfg[decoders[0]]["embeddings"], vocab_size=len(trg_vocab),
            padding_idx=trg_padding_idx)

    encoder = RecurrentEncoder(**cfg["encoder"], vocab_size=len(src_vocab),
                               emb_size=src_embed.embedding_dim)

    if len(decoders) == 1:
        # standard NMT
        d = cfg[decoders[0]]
        decoder = RecurrentDecoder(**d, encoder=encoder,
                                   vocab_size=len(trg_vocab),
                                   emb_size=trg_embed.embedding_dim)


        model = Model(encoder=encoder, decoder=decoder,
                      src_embed=src_embed, trg_embed=trg_embed,
                      src_vocab=src_vocab, trg_vocab=trg_vocab)

    elif len(decoders) == 2:
        # deliberation network
        print("DELIBERATION NETWORK")
        decoder1 = RecurrentDecoder(**cfg[decoders[0]], encoder=encoder,
                                   vocab_size=len(trg_vocab),
                                   emb_size=trg_embed.embedding_dim)
        decoder2 = RecurrentDeliberationDecoder(**cfg[decoders[1]], encoder=encoder,
                                   vocab_size=len(trg_vocab),
                                   emb_size=trg_embed.embedding_dim)

        model = DeliberationModel(encoder=encoder, decoder1=decoder1,
                                  decoder2=decoder2, src_embed=src_embed,
                                  trg_embed=trg_embed, src_vocab=src_vocab,
                                  trg_vocab=trg_vocab)

    # custom initialization of model parameters
    initialize_model(model, cfg, src_padding_idx, trg_padding_idx)

    return model


class Model(nn.Module):
    """
    Base Model class
    """

    def __init__(self,
                 name: str = "my_model",
                 encoder: Encoder = None,
                 decoder: Decoder = None,
                 src_embed: Embeddings = None,
                 trg_embed: Embeddings = None,
                 src_vocab: Vocabulary = None,
                 trg_vocab: Vocabulary = None):
        """
        Create a new encoder-decoder model
        :param name:
        :param encoder:
        :param decoder:
        :param src_embed:
        :param trg_embed:
        :param src_vocab:
        :param trg_vocab:
        """
        super(Model, self).__init__()

        self.name = name
        self.src_embed = src_embed
        self.trg_embed = trg_embed
        self.encoder = encoder
        self.decoder = decoder
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
        return self.decode(encoder_output=encoder_output,
                           encoder_hidden=encoder_hidden,
                           src_mask=src_mask, trg_input=trg_input,
                           unrol_steps=unrol_steps), lm_output

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

    def decode(self, encoder_output, encoder_hidden, src_mask, trg_input,
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
        return self.decoder(trg_embed=self.trg_embed(trg_input),
                            encoder_output=encoder_output,
                            encoder_hidden=encoder_hidden,
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
        decoder_output, lm_output = self.forward(
            src=batch.src, trg_input=batch.trg_input,
            src_mask=batch.src_mask, src_lengths=batch.src_lengths)

        out, hidden, att_probs, _ = decoder_output

        # compute log probs
        log_probs = F.log_softmax(out, dim=-1)

        # compute batch loss
        batch_loss = criterion(
            input=log_probs.contiguous().view(-1, log_probs.size(-1)),
            target=batch.trg.contiguous().view(-1))
        # return batch loss = sum over all elements in batch that are not pad

        # add lm loss
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
            batch_loss += self.encoder.lm_task*lm_loss  # weighted

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

        # greedy decoding
        if beam_size == 0:
            stacked_output, stacked_attention_scores = greedy(
                encoder_hidden=encoder_hidden, encoder_output=encoder_output,
                src_mask=batch.src_mask, embed=self.trg_embed,
                bos_index=self.bos_index, decoder=self.decoder,
                max_output_length=max_output_length)
            # batch, time, max_src_length
        else:  # beam size
            stacked_output, stacked_attention_scores = \
                beam_search(size=beam_size, encoder_output=encoder_output,
                            encoder_hidden=encoder_hidden,
                            src_mask=batch.src_mask, embed=self.trg_embed,
                            max_output_length=max_output_length,
                            alpha=beam_alpha, eos_index=self.eos_index,
                            pad_index=self.pad_index, bos_index=self.bos_index,
                            decoder=self.decoder)

        return stacked_output, stacked_attention_scores

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
