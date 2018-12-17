# coding: utf-8
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from joeynmt.initialization import initialize_model
from joeynmt.embeddings import Embeddings
from joeynmt.encoders import Encoder, RecurrentEncoder
from joeynmt.decoders import Decoder, RecurrentDecoder
from joeynmt.correctors import RecurrentCorrector, Corrector
from joeynmt.constants import PAD_TOKEN, EOS_TOKEN, BOS_TOKEN
from joeynmt.search import beam_search, greedy
from joeynmt.vocabulary import Vocabulary
from joeynmt.helpers import arrays_to_sentences


def build_model(cfg: dict = None,
                src_vocab: Vocabulary = None,
                trg_vocab: Vocabulary = None):
    src_padding_idx = src_vocab.stoi[PAD_TOKEN]
    trg_padding_idx = trg_vocab.stoi[PAD_TOKEN]

    src_embed = Embeddings(
        **cfg["encoder"]["embeddings"], vocab_size=len(src_vocab),
        padding_idx=src_padding_idx)

    if cfg.get("tied_embeddings", False) \
        and src_vocab.itos == trg_vocab.itos:
        # share embeddings for src and trg
        trg_embed = src_embed
    else:
        trg_embed = Embeddings(
            **cfg["decoder"]["embeddings"], vocab_size=len(trg_vocab),
            padding_idx=trg_padding_idx)

    encoder = RecurrentEncoder(**cfg["encoder"],
                               emb_size=src_embed.embedding_dim)
    decoder = RecurrentDecoder(**cfg["decoder"], encoder=encoder,
                               vocab_size=len(trg_vocab),
                               emb_size=trg_embed.embedding_dim)
    corrector = RecurrentCorrector(**cfg["corrector"],
                                   trg_embed=trg_embed,
                                   decoder_size=decoder.hidden_size)

    model = Model(encoder=encoder, decoder=decoder, corrector=corrector,
                  src_embed=src_embed, trg_embed=trg_embed,
                  src_vocab=src_vocab, trg_vocab=trg_vocab)

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
                 corrector: Corrector = None,
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
        self.corrector = corrector
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.bos_index = self.trg_vocab.stoi[BOS_TOKEN]
        self.pad_index = self.trg_vocab.stoi[PAD_TOKEN]
        self.eos_index = self.trg_vocab.stoi[EOS_TOKEN]

    def forward(self, src, trg_input, src_mask, src_lengths, correct=False):
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
        encoder_output, encoder_hidden = self.encode(src=src,
                                                     src_length=src_lengths,
                                                     src_mask=src_mask)
        unrol_steps = trg_input.size(1)
        decoder_output = self.decode(encoder_output=encoder_output,
                           encoder_hidden=encoder_hidden,
                           src_mask=src_mask, trg_input=trg_input,
                           unrol_steps=unrol_steps)

        if correct:
            outputs, hidden, att_probs, att_vectors = decoder_output
            # corrector predicts perturbation of hidden state that needs correction
            # input: attention vector of forwards RNN, hidden state of backwards RNN
            # R_corr: c_t = RNN([fw, bw, o_t-1], c_t-1), o_t = tanh(Linear(c_t))
            # loss: -log(P(r|NMT,o_t))
            # TODO decoder predictions: what if beam search?
            greedy_pred = torch.argmax(outputs, dim=-1).cpu().numpy()  # batch x length
            #print("pred", greedy_pred.shape, greedy_pred)
            # print("flipped", np.flip(greedy_pred, axis=1))
            # compute mask with numpy
            eos = np.where(greedy_pred == self.trg_vocab.stoi[EOS_TOKEN], 1, 0)
            #print("np eos", eos)
            # mark everything after first eos with 1, even if non-eos
            eos_filled = np.where(np.cumsum(eos, axis=1)>=1, 1, 0)
            # then create mask with 0s after first eos
            mask = np.where(np.cumsum(eos_filled, axis=1)>1, 0, 1)
            #print("np mask", mask)
            rev_pred_mask = outputs.new_tensor(np.flip(mask, axis=1).copy(),
                                               dtype=torch.uint8)
            # reverse the prediction to read it in backwards
            rev_predicted = outputs.new_tensor(
                np.flip(greedy_pred, axis=1).copy(), dtype=torch.long)  # flip in time (copy is needed for contiguity)
            #print("EOS ID", self.trg_vocab.stoi[EOS_TOKEN])
            #eos = torch.where(rev_predicted.eq(self.trg_vocab.stoi[EOS_TOKEN]),
            #                  outputs.new_full([1], 1),
            #                  outputs.new_full([1], 0))
            #print("eos?", eos)
            #rev_pred_mask = torch.where(torch.cumsum(eos, dim=1).lt(1),  # lt since flipped
            #                            outputs.new_full([1], 1),
            #                            outputs.new_full([1], 0)).unsqueeze(1).byte()
            #rint("mask", rev_pred_mask)
            # TODO is mask even needed?
            pred_length = rev_pred_mask.sum(1)
            #print("len", pred_length)

            # predict corrections
            corrections = self.correct(
                y=rev_predicted, y_length=pred_length,
                mask=rev_pred_mask, y_states=att_vectors)

            # run decoder again with corrections
            corr_outputs, corr_hidden, corr_att_probs, corr_att_vectors =\
                self.decode(encoder_output=encoder_output,
                        encoder_hidden=encoder_hidden,
                        src_mask=src_mask, trg_input=trg_input,
                        unrol_steps=unrol_steps,
                        corrections=corrections)
            return greedy_pred, corrections, \
                   corr_outputs, corr_hidden, corr_att_probs, corr_att_vectors

        return decoder_output

    def correct(self, y, y_length, mask, y_states):
        """
        Run the corrector to predict corrections for hidden states
        :param y:
        :param y_length:
        :param mask:
        :param y_states:
        :return:
        """
        return self.corrector(self.trg_embed(y), y_length, mask, y_states)

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
               unrol_steps, decoder_hidden=None, corrections=None):
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
                            hidden=decoder_hidden,
                            corrections=corrections)

    def get_xent_loss_for_batch(self, batch, criterion):
        """
        Compute non-normalized loss and number of tokens for a batch

        :param batch:
        :param criterion:
        :return:
        """
        out, hidden, att_probs, _ = self.forward(
            src=batch.src, trg_input=batch.trg_input,
            src_mask=batch.src_mask, src_lengths=batch.src_lengths)

        # compute log probs
        log_probs = F.log_softmax(out, dim=-1)

        # compute batch loss
        batch_loss = criterion(
            input=log_probs.contiguous().view(-1, log_probs.size(-1)),
            target=batch.trg.contiguous().view(-1))
        # return batch loss = sum over all elements in batch that are not pad
        return batch_loss

    def get_corr_loss_for_batch(self, batch, criterion):
        # TODO
        greedy_pred, corrections, \
        corr_outputs, corr_hidden, corr_att_probs, corr_att_vectors = self.forward(
            src=batch.src, trg_input=batch.trg_input, correct=True,
            src_mask=batch.src_mask, src_lengths=batch.src_lengths)

        print("before corr", " ".join(arrays_to_sentences(greedy_pred, vocabulary=self.trg_vocab)[0]))
        corr_pred = torch.argmax(corr_outputs, dim=2).cpu().numpy()
        print("after corr", " ".join(arrays_to_sentences(corr_pred, vocabulary=self.trg_vocab)[0]))
        print("ref", " ".join(arrays_to_sentences(batch.trg, vocabulary=self.trg_vocab)[0]))
        # loss for correction is log likelihood of ref

        # compute log probs of correction
        # TODO should be related to xent loss? should be penalized if even higher!
        # diff = corr_loss/xent_loss ?
        log_probs = F.log_softmax(corr_outputs, dim=-1)

        # compute batch loss
        batch_loss = criterion(
            input=log_probs.contiguous().view(-1, log_probs.size(-1)),
            target=batch.trg.contiguous().view(-1))
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
        encoder_output, encoder_hidden = self.encode(
            batch.src, batch.src_lengths,
            batch.src_mask)

        # if maximum output length is not globally specified, adapt to src len
        if max_output_length is None:
            max_output_length = int(max(batch.src_lengths.cpu().numpy()) * 1.5)

        # TODO add corrector!
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
