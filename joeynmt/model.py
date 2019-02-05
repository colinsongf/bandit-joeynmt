# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from joeynmt.initialization import initialize_model
from joeynmt.embeddings import Embeddings
from joeynmt.encoders import Encoder, RecurrentEncoder
from joeynmt.decoders import Decoder, RecurrentDecoder
from joeynmt.regulators import Regulator, RecurrentRegulator
from joeynmt.constants import PAD_TOKEN, EOS_TOKEN, BOS_TOKEN
from joeynmt.search import beam_search, greedy, sample
from joeynmt.vocabulary import Vocabulary
from torch.distributions import Categorical
from joeynmt.metrics import sbleu, ster
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
        #reg_trg_embed = Embeddings(
        #    **cfg["regulator"]["embeddings"], vocab_size=len(trg_vocab),
        #    padding_idx=trg_padding_idx)

    encoder = RecurrentEncoder(**cfg["encoder"],
                               emb_size=src_embed.embedding_dim)
    decoder = RecurrentDecoder(**cfg["decoder"], encoder=encoder,
                               vocab_size=len(trg_vocab),
                               emb_size=trg_embed.embedding_dim)
    if cfg.get("regulator") is not None:
        reg_src_embed = Embeddings(
            **cfg["regulator"]["embeddings"], vocab_size=len(src_vocab),
            padding_idx=src_padding_idx)
        regulator = RecurrentRegulator(**cfg["regulator"],
                                   src_emb_size=reg_src_embed.embedding_dim)
                                 #  trg_emb_size=reg_trg_embed.embedding_dim)
    else:
        regulator = None
        reg_src_embed = None

    model = Model(encoder=encoder, decoder=decoder, regulator=regulator,
                  src_embed=src_embed, trg_embed=trg_embed,
                  reg_src_embed=reg_src_embed, #reg_trg_embed=reg_trg_embed,
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
                 regulator: Regulator = None,
                 src_embed: Embeddings = None,
                 trg_embed: Embeddings = None,
                 reg_src_embed: Embeddings = None,
                 reg_trg_embed: Embeddings = None,
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
        self.reg_src_embed = reg_src_embed
        self.reg_trg_embed = reg_trg_embed
        self.encoder = encoder
        self.decoder = decoder
        self.regulator = regulator
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.bos_index = self.trg_vocab.stoi[BOS_TOKEN]
        self.pad_index = self.trg_vocab.stoi[PAD_TOKEN]
        self.eos_index = self.trg_vocab.stoi[EOS_TOKEN]
        self.rewards = []

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
        encoder_output, encoder_hidden = self.encode(src=src,
                                                     src_length=src_lengths,
                                                     src_mask=src_mask)
        unrol_steps = trg_input.size(1)
        decoder_output = self.decode(encoder_output=encoder_output,
                           encoder_hidden=encoder_hidden,
                           src_mask=src_mask, trg_input=trg_input,
                           unrol_steps=unrol_steps)
        return encoder_output, encoder_hidden, decoder_output

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

    def regulate(self, src, src_length): #, hyp):
        """

        :param src:
        :param hyp:
        :return:
        """
        return self.regulator(src=self.reg_src_embed(src), src_length=src_length)
                            #  hyp=self.reg_trg_embed(hyp))

    # TODO split batch according to regulator prediction
    # then for each part of the batch compute parts
    # then sum loss
    # -> still mini-batching, but smaller

    def _self_sup_loss(self, selection, encoder_out, encoder_hidden, src_mask, max_output_length, criterion, beam_size=10, beam_alpha=1.0, entropy=False):
        """
        Compute the loss for self-supervised training for the given selection
        of indices of the batch

        loss: -log_p(beam_search_hyp | x)

        :param selection:
        :param encoder_out:
        :param encoder_hidden:
        :param src_mask:
        :return:
        """
        selected_encoder_out = torch.index_select(encoder_out, index=selection,
                                              dim=0)
        selected_encoder_hidden = torch.index_select(encoder_hidden, index=selection,
                                                 dim=0)
        selected_src_mask = torch.index_select(src_mask, index=selection, dim=0)
        bs_hyp, _ = beam_search(size=beam_size,
                                       encoder_output=selected_encoder_out,
                                       encoder_hidden=selected_encoder_hidden,
                                       src_mask=selected_src_mask,
                                       embed=self.trg_embed,
                                       max_output_length=max_output_length,
                                       alpha=beam_alpha,
                                       eos_index=self.eos_index,
                                       pad_index=self.pad_index,
                                       bos_index=self.bos_index,
                                       decoder=self.decoder)
        selected_batch_size = selection.shape[0]
        selected_tokens = bs_hyp.size
        bs_hyp_pad = np.full(shape=(selected_batch_size, max_output_length),
                                  fill_value=self.pad_index)
        for i, row in enumerate(bs_hyp):
            for j, col in enumerate(row):
                bs_hyp_pad[i, j] = col
        # print("padded", bs_hyp_pad)
        bos_array = np.full(shape=(selected_batch_size, 1),
                            fill_value=self.bos_index)
        # prepend bos but cut off one bos
        bs_hyp_pad_bos = np.concatenate((bos_array, bs_hyp_pad),
                                             axis=1)[:, :-1]
        # print("with bos", bs_hyp_pad_bos)

        # treat bs output as target for forced decoding to get log likelihood of bs output
        bs_out, _, _, _ = self.decode(encoder_output=selected_encoder_out,
                                           encoder_hidden=selected_encoder_hidden,
                                           trg_input=src_mask.new(
                                               bs_hyp_pad_bos).long(),
                                           src_mask=selected_src_mask,
                                           unrol_steps=bs_hyp_pad_bos.shape[1])
        bs_log_probs = F.log_softmax(bs_out, dim=-1)
        # greedy_log_prob = self.force_decode(encoder_output=encoder_out,
        #                                 encoder_hidden=encoder_hidden,
        #                                 trg_input=batch.trg.new(greedy_hyp).long(),
        #                                 src_mask=batch.src_mask)

        bs_target = src_mask.new(bs_hyp_pad).long()

        bs_nll = criterion(
            input=bs_log_probs.contiguous().view(-1,bs_log_probs.size(-1)),
            target=bs_target.view(-1))
        self_sup_loss = bs_nll.view(selected_batch_size, -1).sum(
            -1)  # batch

        if entropy:
            entropy = (-torch.exp(bs_log_probs) * bs_log_probs).sum(
                -1).mean(1)
            # print("entropy", entropy)
            self_sup_loss = self_sup_loss - entropy  # *confidence.detach()
            # print("weighted", self_sup_loss)
        # TODO logprob selection can actually be done for all, just return chosen hyp and reward
        # then logprobs are selected and multiplied by reward
        assert self_sup_loss.size(0) == selected_batch_size
        return self_sup_loss.sum(), selected_tokens, selected_batch_size

    def _weak_sup_loss(self, selection, encoder_out, encoder_hidden, src_mask, max_output_length, chunk_type, criterion, target, level, weak_baseline=True, temperature=1.0):
        """
        Compute weakly-supervised loss for selected inputs

        loss: -log_p(sampled_hyp | x) * -reward

        Reward is either token- or sequence-based

        :param selection:
        :return:
        """
        selected_encoder_out = torch.index_select(encoder_out, index=selection,
                                                  dim=0)
        selected_encoder_hidden = torch.index_select(encoder_hidden,
                                                     index=selection,
                                                     dim=0)
        selected_src_mask = torch.index_select(src_mask, index=selection, dim=0)
        selected_trg = torch.index_select(target, index=selection, dim=0)
        trg_np = selected_trg.detach().cpu().numpy()

        sample_hyp, _ = sample(encoder_output=selected_encoder_out,
                             encoder_hidden=selected_encoder_hidden,
                             src_mask=selected_src_mask, embed=self.trg_embed,
                             max_output_length=max_output_length,
                             bos_index=self.bos_index,
                             decoder=self.decoder, temperature=temperature)

        selected_batch_size = selection.shape[0]
        sample_hyp_pad = np.full(shape=(selected_batch_size, max_output_length),
                             fill_value=self.pad_index)
        for i, row in enumerate(sample_hyp):
            for j, col in enumerate(row):
                sample_hyp_pad[i, j] = col
        # print("padded", bs_hyp_pad)
        bos_array = np.full(shape=(selected_batch_size, 1),
                            fill_value=self.bos_index)
        # prepend bos but cut off one bos
        sample_hyp_pad_bos = np.concatenate((bos_array, sample_hyp_pad),
                                        axis=1)[:, :-1]
        # print("with bos", bs_hyp_pad_bos)

        # treat bs output as target for forced decoding to get log likelihood of bs output
        sample_out, _, _, _ = self.decode(encoder_output=selected_encoder_out,
                                      encoder_hidden=selected_encoder_hidden,
                                      trg_input=src_mask.new(
                                          sample_hyp_pad_bos).long(),
                                      src_mask=selected_src_mask,
                                      unrol_steps=sample_hyp_pad_bos.shape[1])
        sample_log_probs = F.log_softmax(sample_out, dim=-1)

        sample_target = src_mask.new(sample_hyp_pad).long()

        sample_nll = criterion(
            input=sample_log_probs.contiguous().view(-1, sample_log_probs.size(-1)),
            target=sample_target.view(-1))

        if chunk_type == "marking":
            # in case of markings: "chunk-based" feedback: nll of bs weighted by 0/1
            # 1 if correct, 0 if incorrect
            # fill curr_hyp with padding, since different length
            # print("bs", bs_hyp_pad)
            # print("trg", trg_np)
            # padding area is zero
            markings = np.zeros_like(sample_hyp_pad, dtype=float)
            for i, row in enumerate(sample_hyp):
                for j, val in enumerate(row):
                    try:
                        if trg_np[i, j] == val:
                            markings[i, j] = 1.
                    except IndexError:  # BS is longer than trg
                        continue
            chunk_loss = (sample_nll.view(selected_batch_size, -1) * src_mask.new(
                markings).float()).sum(1)
            selected_tokens = markings.sum()

        else:
            # use same reward for all the tokens
            if chunk_type == "sbleu":
                # decode hypothesis and target
                join_char = " " if level in ["word", "bpe"] else ""
                bs_hyp_decoded = [join_char.join(t) for t in
                                  arrays_to_sentences(arrays=sample_hyp,
                                                      vocabulary=self.trg_vocab,
                                                      cut_at_eos=True)]
                trg_np_decoded = [join_char.join(t) for t in
                                  arrays_to_sentences(arrays=trg_np,
                                                      vocabulary=self.trg_vocab,
                                                      cut_at_eos=True)]
                assert len(trg_np_decoded) == len(bs_hyp_decoded)
                # compute sBLEUs
                sbleus = np.array(sbleu(bs_hyp_decoded, trg_np_decoded))
                rewards = 1 - sbleus

            elif chunk_type == "ster":
                # decode hypothesis and target
                bs_hyp_decoded_list = arrays_to_sentences(arrays=sample_hyp,
                                                          vocabulary=self.trg_vocab,
                                                          cut_at_eos=True)
                trg_np_decoded_list = arrays_to_sentences(arrays=trg_np,
                                                          vocabulary=self.trg_vocab,
                                                          cut_at_eos=True)
                assert len(trg_np_decoded_list) == len(bs_hyp_decoded_list)
                sters = np.array(ster(bs_hyp_decoded_list, trg_np_decoded_list))
                rewards = sters

            if weak_baseline:
                if len(self.rewards) > 0:
                    # subtract mean from reward
                    new_rewards = rewards - np.mean(self.rewards)
                else:
                    new_rewards = rewards
                #update baseline
                self.rewards.extend(rewards)
            else:
                new_rewards = rewards

            selected_tokens = sample_hyp.size
            # make update with baselined rewards
            chunk_loss = sample_nll.sum(-1) * src_mask.new(new_rewards).float()

        assert chunk_loss.size(0) == selected_batch_size
        return chunk_loss.sum(), selected_tokens, selected_batch_size

    def _full_sup_loss(self, selection, decoder_out, criterion, target):
        """
        Compute the loss for fully-supervised training for the given selection
        of indices of the batch

        loss: -log_p(reference | x)

        :param selection:
        :param src_mask:
        :return:
        """
        selected_target = torch.index_select(target, index=selection, dim=0)
        selected_decoder_out = torch.index_select(decoder_out, index=selection,
                                                  dim=0)
        selected_batch_size = selection.shape[0]
        selected_tokens = (selected_target != self.pad_index).sum().cpu().numpy()

        # compute log probs of teacher-forced decoder for fully-supervised training
        tf_log_probs = F.log_softmax(selected_decoder_out, dim=-1)
        # in case of full supervision (teacher forcing with post-edit)
        full_sup_loss = criterion(
            input=tf_log_probs.contiguous().view(-1, tf_log_probs.size(-1)),
            target=selected_target.contiguous().view(-1)).view(selected_batch_size, -1).sum(-1)

        assert full_sup_loss.size(0) == selected_batch_size
        return full_sup_loss.sum(), selected_tokens, selected_batch_size

    def get_loss_for_batch(self, batch, criterion, regulate=False, pred=False,
                           max_output_length=100, chunk_type="marking", level="word",
                           entropy=False, search="beam", weak_baseline=True):
        """
        Compute non-normalized loss and number of tokens for a batch

        :param batch:
        :param criterion:
        :return:
        """
        # TODO if there is no regutor, predictions of it are always 3 -> generalize code
        encoder_out, encoder_hidden, decoder_out = \
            self.forward(src=batch.src, trg_input=batch.trg_input,
                         src_mask=batch.src_mask, src_lengths=batch.src_lengths)
        out, hidden, att_probs, _ = decoder_out
        batch_size = batch.src.size(0)

        if not regulate:
            log_probs = F.log_softmax(decoder_out, dim=-1)
            batch_loss = criterion(
                input=log_probs.contiguous().view(-1, log_probs.size(-1)),
                target=batch.trg.contiguous().view(-1)).sum()
            batch_tokens = batch.ntokens
            batch_seqs = batch.nseqs
            assert batch_loss.size(0) == 1
            reg_pred = None
            reg_log_probs = None

        else:
            # with regulator
            regulator_out = self.regulate(batch.src,
                                          batch.src_lengths)  # bs_target)
            reg_log_probs = F.log_softmax(regulator_out, dim=-1)

            # sample an output
            reg_dist = Categorical(logits=regulator_out)
            reg_pred = reg_dist.sample()

            # heuristic: always choose one type of supervision
            if pred is not False:
                if pred == "random":
                    # random choice
                    fill_value = np.random.randint(0, 4, size=batch_size)
                else:
                    fill_value = pred
                reg_pred = torch.from_numpy(
                    np.full(shape=(batch_size), fill_value=fill_value)).to(
                    regulator_out.device).long()
                # print("regulator prediction", reg_pred)

            batch_loss = 0
            batch_tokens = 0
            batch_seqs = 0

            # split up the batch and sum the individual losses
            if 0 in reg_pred:
                # skip those: no loss
                zeros = torch.eq(reg_pred, 0)
                zeros_idx = zeros.nonzero().squeeze(1)
            if 1 in reg_pred:
                ones = torch.eq(reg_pred, 1)
                ones_idx = ones.nonzero().squeeze(1)
                # compute self-train loss for those (smaller batch)
                self_sup_loss_selected, tokens, seqs = self._self_sup_loss(
                    selection=ones_idx, encoder_out=encoder_out, entropy=entropy,
                    encoder_hidden=encoder_hidden, src_mask=batch.src_mask,
                    max_output_length=max_output_length, criterion=criterion)
                #print("self sup selected", self_sup_loss_selected)
                batch_loss += self_sup_loss_selected.sum()
                batch_tokens += tokens
                batch_seqs += seqs

            if 2 in reg_pred:
                # compute weak-sup. loss for those
                twos = torch.eq(reg_pred, 2)
                twos_idx = twos.nonzero().squeeze(1)
                weak_sup_loss_selected, tokens, seqs = self._weak_sup_loss(
                    selection=twos_idx, encoder_out=encoder_out,
                    encoder_hidden=encoder_hidden, src_mask=batch.src_mask,
                    max_output_length=max_output_length, criterion=criterion,
                    chunk_type=chunk_type, level=level,
                    target=batch.trg, temperature=1.0,
                    weak_baseline=weak_baseline)
                #print("weak sup selected", weak_sup_loss_selected)
                batch_loss += weak_sup_loss_selected.sum()
                batch_tokens += tokens
                batch_seqs += seqs

            if 3 in reg_pred:
                # compute fully-sup. loss for those
                threes = torch.eq(reg_pred, 3)
                threes_idx = threes.nonzero().squeeze(1)
                full_sup_loss_selected, tokens, seqs = self._full_sup_loss(
                    selection=threes_idx, decoder_out=out,
                    criterion=criterion, target=batch.trg)
               # print("full sup selected", full_sup_loss_selected)
                batch_loss += full_sup_loss_selected
                batch_tokens += tokens
                batch_seqs += seqs

            if type(batch_loss) == int:  # no update for batch
                batch_loss = None

        return batch_loss, reg_log_probs, reg_pred, batch_tokens, batch_seqs

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
