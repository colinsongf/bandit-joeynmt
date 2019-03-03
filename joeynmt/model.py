# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Set
from joeynmt.initialization import initialize_model
from joeynmt.embeddings import Embeddings
from joeynmt.encoders import Encoder, RecurrentEncoder
from joeynmt.decoders import Decoder, RecurrentDecoder
from joeynmt.regulators import Regulator, RecurrentRegulator
from joeynmt.constants import PAD_TOKEN, EOS_TOKEN, BOS_TOKEN
from joeynmt.search import beam_search, greedy, sample
from joeynmt.vocabulary import Vocabulary
from torch.distributions import Categorical
from joeynmt.metrics import sbleu, ster, sgleu
from joeynmt.helpers import arrays_to_sentences, sentences_to_arrays
import pyter
from difflib import SequenceMatcher
from string import whitespace


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
        self.bleus = []
        self.entropies = []
        if self.regulator is not None:
            self.rewards_per_output = {i: [] for i in self.regulator.index2label.keys()}
            self.costs_per_output = {i: [] for i in self.regulator.index2label.keys()}

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
               unrol_steps, decoder_hidden=None, attention_drop=0.0):
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
                            attention_drop=attention_drop)

    def regulate(self, src, src_length, hyp):
        """

        :param src:
        :param hyp:
        :return:
        """
        return self.regulator(src=self.reg_src_embed(src), src_length=src_length,
                              hyp=self.reg_src_embed(hyp) if hyp is not None else None)

    # TODO split batch according to regulator prediction
    # then for each part of the batch compute parts
    # then sum loss
    # -> still mini-batching, but smaller

    def _self_sup_loss(self, selection, encoder_out, encoder_hidden, src_mask,
                       max_output_length, criterion, target, level, beam_size=10,
                       beam_alpha=1.0, entropy=False, logger=None,
                       attention_drop=0.0, hyps=None, hyp_inputs=None, hyp_masks=None):
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
        selected_batch_size = selection.shape[0]

        if hyps is not None and hyp_inputs is not None and hyp_masks is not None:
            selected_hyps = torch.index_select(hyps, index=selection, dim=0)
            selected_hyp_inputs = torch.index_select(hyp_inputs, index=selection, dim=0)
            selected_hyp_masks = torch.index_select(hyp_masks, index=selection, dim=0)
            #logger.info("SELF USING LOGGED HYP INSTEAD OF SAMPLE: {}".format(selected_hyps))
            # instead of sampling from the current model, take the log hypothesis
            bs_hyp = selected_hyps
            bs_hyp_mask = selected_hyp_masks
            bs_hyp_inputs = selected_hyp_inputs
            bs_target = bs_hyp
            selected_tokens = selected_hyp_masks.sum().cpu().numpy()
        else:
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
            bs_hyp_pad = np.full(shape=(selected_batch_size, max_output_length),
                                  fill_value=self.pad_index)
            # print("padded", bs_hyp_pad)
            bos_array = np.full(shape=(selected_batch_size, 1),
                                fill_value=self.bos_index)
            # prepend bos but cut off one bos
            bs_hyp_pad_bos = np.concatenate((bos_array, bs_hyp_pad),
                                            axis=1)[:, :-1]

            for i, row in enumerate(bs_hyp):
                for j, col in enumerate(row):
                    bs_hyp_pad[i, j] = col

            bs_hyp_inputs = src_mask.new(bs_hyp_pad_bos).long()
            bs_target = src_mask.new(bs_hyp_pad).long()

            selected_tokens = bs_hyp.size

        # print("with bos", bs_hyp_pad_bos)

        # treat bs output as target for forced decoding to get log likelihood of bs output
        bs_out, _, _, _ = self.decode(encoder_output=selected_encoder_out,
                                           encoder_hidden=selected_encoder_hidden,
                                           trg_input=bs_hyp_inputs,
                                           src_mask=selected_src_mask,
                                           unrol_steps=bs_hyp_inputs.shape[1],
                                      attention_drop=attention_drop)
        bs_log_probs = F.log_softmax(bs_out, dim=-1)
        # greedy_log_prob = self.force_decode(encoder_output=encoder_out,
        #                                 encoder_hidden=encoder_hidden,
        #                                 trg_input=batch.trg.new(greedy_hyp).long(),
        #                                 src_mask=batch.src_mask)


        bs_nll = criterion(
            input=bs_log_probs.contiguous().view(-1, bs_log_probs.size(-1)),
            target=bs_target.view(-1))
        self_sup_loss = bs_nll.view(selected_batch_size, -1).sum(
            -1)  # batch

        if entropy:
            logger.info("before entropy subtraction: {}".format(self_sup_loss))
            entropy = (-torch.exp(bs_log_probs) * bs_log_probs).sum(
                -1).mean(1)
            logger.info("entropy: {}".format(entropy))
            self_sup_loss = self_sup_loss - entropy #.detach()  # *confidence.detach()
            logger.info("entropy-weighted: {}".format(self_sup_loss))
        # TODO logprob selection can actually be done for all, just return chosen hyp and reward
        # then logprobs are selected and multiplied by reward

        if logger is not None:
            logger.info("Examples from self-supervision:")
            join_char = " " if level in ["word", "bpe"] else ""
            decoded_bs_hyp = [join_char.join(t) for t in arrays_to_sentences(bs_hyp[:3], vocabulary=self.trg_vocab)]
            decoded_ref = [join_char.join(t) for t in arrays_to_sentences(torch.index_select(target, index=selection, dim=0)[:3], vocabulary=self.trg_vocab)]
            for hyp, ref, logprob in zip(decoded_bs_hyp, decoded_ref, -self_sup_loss[:3]):
                logger.info("\tSelf-supervision: {} ({:.3f})".format(hyp, logprob))
                logger.info("\tReference: {}".format(ref))


        assert self_sup_loss.size(0) == selected_batch_size
        return self_sup_loss.sum(), selected_tokens, selected_batch_size

    def _weak_sup_loss(self, selection, src, encoder_out, encoder_hidden, src_mask,
                       max_output_length, chunk_type, criterion, target, level,
                       weak_baseline=True, weak_temperature=1.0,
                       weak_search="sample",
                       beam_size=10, beam_alpha=1.0, logger=None,
                       case_sensitive=True,
                       hyps=None, hyp_inputs=None, hyp_masks=None):
        """
        Compute weakly-supervised loss for selected inputs

        loss: -log_p(sampled_hyp | x) * -reward

        Reward is either token- or sequence-based

        :param selection:
        :return:
        """
        selected_srcs = torch.index_select(src, dim=0, index=selection)
        selected_encoder_out = torch.index_select(encoder_out, index=selection,
                                                  dim=0)
        selected_encoder_hidden = torch.index_select(encoder_hidden,
                                                     index=selection,
                                                     dim=0)
        selected_src_mask = torch.index_select(src_mask, index=selection, dim=0)
        selected_trg = torch.index_select(target, index=selection, dim=0)
        trg_np = selected_trg.detach().cpu().numpy()
        selected_batch_size = selection.shape[0]

        if hyps is not None and hyp_inputs is not None and hyp_masks is not None \
                and weak_search == "offline":
            selected_hyps = torch.index_select(hyps, index=selection, dim=0)
            selected_hyp_inputs = torch.index_select(hyp_inputs, index=selection, dim=0)
            selected_hyp_masks = torch.index_select(hyp_masks, index=selection, dim=0)
            #logger.info("WEAK USING LOGGED HYP INSTEAD OF SAMPLE: {}".format(selected_hyps))
            # instead of sampling from the current model, take the log hypothesis
            sample_hyp = selected_hyps
            sample_hyp_mask = selected_hyp_masks
            sample_hyp_inputs = selected_hyp_inputs
            sample_target = sample_hyp
            selected_tokens = sample_hyp_mask.sum().cpu().numpy()
        else:

            if weak_search == "sample":
                sample_hyp, _ = sample(encoder_output=selected_encoder_out,
                                     encoder_hidden=selected_encoder_hidden,
                                     src_mask=selected_src_mask, embed=self.trg_embed,
                                     max_output_length=max_output_length,
                                     bos_index=self.bos_index,
                                     decoder=self.decoder, temperature=weak_temperature)

            elif weak_search == "beam":
                sample_hyp, _ = beam_search(size=beam_size,
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
            else:  # greedy
                sample_hyp, _ = greedy(encoder_output=selected_encoder_out,
                                       encoder_hidden=selected_encoder_hidden,
                                       src_mask=selected_src_mask,
                                       embed=self.trg_embed,
                                       max_output_length=max_output_length,
                                       bos_index=self.bos_index,
                                       decoder=self.decoder)

            sample_hyp_mask = np.not_equal(sample_hyp, self.pad_index).astype(int)

            #sample_hyp_pad = np.full(shape=(selected_batch_size, max_output_length),
            #                     fill_value=self.pad_index)
            #for i, row in enumerate(sample_hyp):
            #    for j, col in enumerate(row):
            #        sample_hyp_pad[i, j] = col
            # print("padded", bs_hyp_pad)
            bos_array = np.full(shape=(selected_batch_size, 1),
                                fill_value=self.bos_index)
            # prepend bos but cut off one pad
            sample_hyp_pad_bos = np.concatenate((bos_array, sample_hyp),
                                            axis=1)[:, :-1]
            sample_hyp_inputs = src_mask.new(sample_hyp_pad_bos).long()
            sample_target = src_mask.new(sample_hyp).long()
            selected_tokens = sample_hyp.size




        # print("with bos", bs_hyp_pad_bos)

        # treat bs output as target for forced decoding to get log likelihood of bs output
        sample_out, _, _, _ = self.decode(encoder_output=selected_encoder_out,
                                      encoder_hidden=selected_encoder_hidden,
                                      trg_input=sample_hyp_inputs,
                                      src_mask=selected_src_mask,
                                      unrol_steps=sample_hyp_inputs.shape[1])
        sample_log_probs = F.log_softmax(sample_out, dim=-1)


        sample_nll = criterion(
            input=sample_log_probs.contiguous().view(-1, sample_log_probs.size(-1)),
            target=sample_target.view(-1)).view(selected_batch_size, -1)

        # decode reference
        join_char = " " if level in ["word", "bpe"] else ""
        refs_np_decoded_list = arrays_to_sentences(arrays=trg_np,
                                              vocabulary=self.trg_vocab,
                                              cut_at_eos=True)
        refs_np_decoded = [join_char.join(t) for t in refs_np_decoded_list]

        # decode hypothesis
        hyps_decoded_list = arrays_to_sentences(arrays=sample_hyp,
                                              vocabulary=self.trg_vocab,
                                              cut_at_eos=True)
        hyps_decoded = [join_char.join(t) for t in hyps_decoded_list]

        # decode source
        decoded_srcs = [join_char.join(t) for t in
                        arrays_to_sentences(selected_srcs,
                                            vocabulary=self.src_vocab)]

        # post-process for BPE
        if level == "bpe":
            # merge byte pairs
            hyps_decoded = [t.replace("@@ ", "") for t in hyps_decoded]
            refs_np_decoded = [t.replace("@@ ", "") for t in
                               refs_np_decoded]
            hyps_decoded_list = [t.split(" ") for t in hyps_decoded]
            refs_np_decoded_list = [t.split(" ") for t in refs_np_decoded]

        if chunk_type == "marking":
            # in case of markings: "chunk-based" feedback: nll of bs weighted by 0/1
            # 1 if correct, 0 if incorrect
            # fill curr_hyp with padding, since different length
            # print("bs", bs_hyp_pad)
            # print("trg", trg_np)
            # padding area is zero
            # TODO use case sensitivity
            if type(sample_hyp) is torch.Tensor:
                sample_hyp = sample_hyp.cpu().numpy()
                sample_hyp_mask = sample_hyp_mask.cpu().numpy()
            markings = np.zeros_like(sample_hyp, dtype=float)
            # for baseline we don't track padding rewards since zeros
            valid_rewards = []
            for i, row in enumerate(sample_hyp):
                for j, val in enumerate(row):
                    try:
                        if trg_np[i, j] == val:
                            markings[i, j] = 1.
                            if sample_hyp_mask[i, j]:
                                valid_rewards.append(1)
                        elif sample_hyp_mask[i, j]:
                            valid_rewards.append(0)
                    except IndexError:  # BS is longer than trg
                        continue
            if weak_baseline:
                if len(self.rewards) > 0:
                    # subtract mean from reward
                    new_markings = (markings - np.mean(self.rewards))*sample_hyp_mask
                else:
                    new_markings = markings
                #update baseline
               # print("valid", valid_rewards)
                self.rewards.extend(valid_rewards)
            else:
                new_markings = markings
            #print("final", new_markings*sample_mask)
           # if weak_baseline:
           #     # baseline over time
           #     # TODO over batch?
           #     markings -= np.mean(markings, axis=1, keepdims=True)
            chunk_loss = (sample_nll * src_mask.new(new_markings*sample_hyp_mask).float()).sum(1)
            costs = markings.sum(-1)

            logger.info("Examples from weak supervision:")
            for hyp, ref, src, logprob, mark, cost in zip(hyps_decoded_list[:3],
                                            refs_np_decoded_list[:3], decoded_srcs[:3],
                                            -sample_nll[:3].sum(1),
                                            new_markings[:3], costs[:3]):
                logger.info("\tSource: {}".format(src))
                logger.info(
                    "\t{} for weak: {} ({:.3f})".format(weak_search, hyp,
                                                        logprob))
                logger.info("\tReference: {}".format(ref))
                logger.info(
                    "\tMarkings {}".format([(h, m) for h, m in zip(hyp, mark)]))
                logger.info("\tCost {}".format(cost))
            logger.info("Current BL: {}".format(np.mean(self.rewards)))

        elif chunk_type == "match":
            # 1 if occurs in reference, 0 if it doesn't
            # position-independent
            # TODO include case sensitivity
            if type(sample_hyp) is torch.Tensor:
                sample_hyp = sample_hyp.cpu().numpy()
                sample_hyp_mask = sample_hyp_mask.cpu().numpy()
            valid_matches = []
            matches = np.zeros_like(sample_hyp, dtype=float)
            for i, row in enumerate(sample_hyp):
                for j, val in enumerate(row):
                    # skip to next row if pad reached
                    if val == self.pad_index:
                        break
                    try:
                        if val in trg_np[i]:
                            matches[i, j] = 1.
                            valid_matches.append(1)
                        else:
                            valid_matches.append(0)
                    except IndexError:  # BS is longer than trg
                        continue
                    # skip to next row if eos has been marked
                    if val == self.eos_index:
                        break

            if weak_baseline:
                if len(self.rewards) > 0:
                    # subtract mean from reward
                    new_matches = (matches - np.mean(self.rewards))*sample_hyp_mask
                else:
                    new_matches = matches
                # update baseline
                self.rewards.extend(valid_matches)
            else:
                new_matches = matches
#            print("hyp", sample_hyp_pad)
#            print("ref", trg_np)
#            print("matches", matches)
            #if weak_baseline:
            #    # baseline over time
            #    # TODO over batch?
            #    matches -= np.mean(matches, axis=1, keepdims=True)
            chunk_loss = (sample_nll * src_mask.new(new_matches*sample_hyp_mask).float()).sum(1)

            costs = matches.sum(-1)


            logger.info("Examples from weak supervision:")
            for hyp, ref, src, logprob, mark, cost in zip(hyps_decoded_list[:3],
                                               refs_np_decoded_list[:3],
                                                    decoded_srcs[:3],
                                                    -sample_nll[:3].sum(1),
                                               new_matches[:3],
                                                          costs[:3]):
                logger.info("\tSource: {}".format(src))

                logger.info(
                    "\t{} (cased: {}) for weak: {} ({:.3f})".format(weak_search, case_sensitive, hyp,
                                                        logprob))
                logger.info("\tReference: {}".format(ref))
                logger.info(
                    "\tMatching {}".format([(h, m) for h, m in zip(hyp, mark)]))
                logger.info("\tCost: {}".format(cost))
            logger.info("Current BL: {}".format(np.mean(self.rewards)))

        elif chunk_type == "lcs":
            #def token_lcs_reward(gold, pred):
                # based on https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Longest_common_substring#Python
                # idea from http://www.aclweb.org/anthology/P18-2052
                # TODO adapt to all longest substrings
            def longest_common_substring_rewards(pred, gold):
                m = [[0] * (1 + len(gold)) for i in range(1 + len(pred))]
                longest, x_longest = 0, 0
                rewards = np.zeros(len(pred))
                for x in range(1, 1 + len(pred)):
                    for y in range(1, 1 + len(gold)):
                        # prevent padding area from getting rewarded
                        if pred[x - 1] == gold[y - 1] and pred[x-1] != self.pad_index and gold[y-1] != self.pad_index:
                            m[x][y] = m[x - 1][y - 1] + 1
                            if m[x][y] > longest:
                                longest = m[x][y]
                            x_longest = x
                        else:
                            m[x][y] = 0
                rewards[x_longest - longest: x_longest] = 1
                #return pred[x_longest - longest: x_longest]
                return rewards

            if type(sample_hyp) is torch.Tensor:
                sample_hyp = sample_hyp.cpu().numpy()
                sample_hyp_mask = sample_hyp_mask.cpu().numpy()

            all_rewards = np.zeros_like(sample_hyp, dtype=float)
            for j, (g, p) in enumerate(zip(trg_np, sample_hyp)):  # iterate over batch
                r = longest_common_substring_rewards(p, g)
                for i, r_i in enumerate(r):
                    all_rewards[j, i] = r_i  # r does cover padding area
            if weak_baseline:
                if len(self.rewards) > 0:
                    # subtract mean from reward
                    new_rewards = (all_rewards - np.mean(self.rewards))*sample_hyp_mask
                else:
                    new_rewards = all_rewards
                # update baseline
                # only with non-padding areas
                valid_rewards = []
                for r, m in zip(all_rewards, sample_hyp_mask):
                    for r_i, m_i in zip(r, m):
                        if m_i:
                            valid_rewards.append(r_i)
                self.rewards.extend(valid_rewards)
            else:
                new_rewards = all_rewards
            #if weak_baseline:
            #    # baseline over time
            #    # TODO over batch?
            #    all_rewards -= np.mean(all_rewards, axis=1, keepdims=True)

            chunk_loss = (sample_nll * src_mask.new(new_rewards*sample_hyp_mask).float()).sum(1)

            costs = all_rewards.sum(-1)


            logger.info("Examples from weak supervision:")
            for hyp, ref, src, logprob, mark, cost in zip(hyps_decoded_list[:3],
                                               refs_np_decoded_list[:3],
                                               decoded_srcs[:3],
                                               -sample_nll[:3].sum(1),
                                               new_rewards[:3],
                                                          costs[:3]):
                logger.info("\tSrc: {}".format(src))
                logger.info(
                    "\t{} (cased: {}) for weak: {} ({:.3f})".format(weak_search, case_sensitive, hyp,
                                                        logprob))
                logger.info("\tReference: {}".format(ref))
                logger.info(
                    "\tLCS {}".format([(h, m) for h, m in zip(hyp, mark)]))
                logger.info("\tCost {}".format(cost))
            logger.info("Current BL: {}".format(np.mean(self.rewards)))

        elif chunk_type == "lcs-all":
            #def token_lcs_reward(gold, pred):
                # based on https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Longest_common_substring#Python
                # idea from http://www.aclweb.org/anthology/P18-2052
                # adapted to all longest substrings
            def all_longest_common_substring_rewards(pred, gold):
                m = [[0] * (1 + len(gold)) for i in range(1 + len(pred))]
                # collect all length of all longest spans here
                len_start = {}
                longest, x_longest = 0, 0
                rewards = np.zeros(len(pred))
                for x in range(1, 1 + len(pred)):
                    for y in range(1, 1 + len(gold)):
                        if pred[x - 1] == gold[y - 1] and pred[x-1] != self.pad_index and gold[y-1] != self.pad_index:
                            m[x][y] = m[x - 1][y - 1] + 1
                            if m[x][y] >= longest:
                                old = len_start.get(m[x][y], set())
                                old.add(x)
                                len_start[m[x][y]] = old
                            if m[x][y] > longest:
                                longest = m[x][y]
                            x_longest = x

                        else:
                            m[x][y] = 0
                #rewards[x_longest - longest: x_longest] = 1
                if len(len_start) > 0:  # skipped if no overlap found
                    for pos in len_start[longest]:
                        rewards[pos - longest: pos] = 1
                    #return pred[x_longest - longest: x_longest]
                return rewards

            all_rewards = np.zeros_like(sample_hyp, dtype=float)
            if type(sample_hyp) is torch.Tensor:
                sample_hyp = sample_hyp.cpu().numpy()
                sample_hyp_mask = sample_hyp_mask.cpu().numpy()
            for j, (g, p) in enumerate(zip(trg_np, sample_hyp)):  # iterate over batch
                r = all_longest_common_substring_rewards(p, g)
                for i, r_i in enumerate(r):
                    all_rewards[j, i] = r_i
            # TODO remove rewards for duplications: only reinforce as often as in reference

            if weak_baseline:
                if len(self.rewards) > 0:
                    # subtract mean from reward
                    new_rewards = (all_rewards - np.mean(self.rewards))*sample_hyp_mask
                else:
                    new_rewards = all_rewards
                # update baseline
                # only with non-padding areas
                valid_rewards = []
                for r, m in zip(all_rewards, sample_hyp_mask):
                    for r_i, m_i in zip(r, m):
                        if m_i:
                            valid_rewards.append(r_i)
                self.rewards.extend(valid_rewards)
            else:
                new_rewards = all_rewards

            #if weak_baseline:
            #    # baseline over time
            #    # TODO over batch?
            #    all_rewards -= np.mean(all_rewards, axis=1, keepdims=True)

            chunk_loss = (sample_nll * src_mask.new(new_rewards*sample_hyp_mask).float()).sum(1)

            costs = all_rewards.sum(-1)

            logger.info("Examples from weak supervision:")
            for hyp, ref, src, logprob, mark, cost in zip(hyps_decoded_list[:3],
                                               refs_np_decoded_list[:3],
                                                decoded_srcs[:3],
                                               -sample_nll[:3].sum(1),
                                               new_rewards[:3],
                                                            costs[:3]):
                logger.info("\tSrc: {}".format(src))
                logger.info(
                    "\t{} (cased: {}) for weak: {} ({:.3f})".format(weak_search, case_sensitive, hyp,
                                                        logprob))
                logger.info("\tReference: {}".format(ref))
                logger.info(
                    "\tLCS-ALL: {}".format([(h, m) for h, m in zip(hyp, mark)]))
                logger.info("\tCost: {}".format(cost))
            logger.info("Current BL: {}".format(np.mean(self.rewards)))

        else:
            # use same reward for all the tokens
            if chunk_type == "sbleu":
                assert len(refs_np_decoded) == len(hyps_decoded)
                # compute sBLEUs
                sbleus = np.array(sbleu(hyps_decoded, refs_np_decoded, case_sensitive=case_sensitive))
                rewards = sbleus

            elif chunk_type == "sgleu":
                assert len(refs_np_decoded) == len(hyps_decoded)
                # compute sGLEUs
                sgleus = np.array(sgleu(hyps_decoded, refs_np_decoded,
                                        case_sensitive=case_sensitive))
                rewards = sgleus

            elif chunk_type == "ster":
                sters = np.array(ster(hyps_decoded_list, refs_np_decoded_list, case_sensitive=case_sensitive))
                rewards = 1-sters

            # TODO or constant cost?
            # TODO make this more sophisticated
            # cost is length of hyp
            costs = [len(h) for h in hyps_decoded]
            # print("COSTS", costs)

            if logger is not None:
                logger.info("Examples from weak supervision:")
                for hyp, ref, src, logprob, r, cost in zip(hyps_decoded[:3],
                                                refs_np_decoded[:3],
                                                decoded_srcs[:3],
                                                -sample_nll[:3].sum(1),
                                                rewards[:3],
                                                costs[:3]):
                    logger.info("\tSrc: {}".format(src))
                    logger.info(
                        "\t{} (cased: {}) for weak: {} ({:.3f})".format(weak_search, case_sensitive, hyp,
                                                            logprob))
                    logger.info("\tReference: {}".format(ref))
                    logger.info(
                        "\tReward: {:.3f} {} (BL: {:.3f})".format(r, chunk_type,
                                                                  np.mean(
                                                                      self.rewards)))
                    logger.info("\tCost: {}".format(cost))
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

            # make update with baselined rewards
            chunk_loss = sample_nll.sum(1) * src_mask.new(new_rewards).float()


        # compute cost
        # word-based -> need to decode
        # how many words occur do not occur in the ref? = #markings
        # ratio: #markings/hyp_len
        # or rather absolute?
        # maybe add 1 as constant since "accept" if not any error
        #costs = []
        #print("refs", refs_np_decoded)
        #print("hyps", hyps_decoded)
        # TODO doesn't really work for sentence-level markings
        # now: max(1, min(neg_markings, pos_markings))
       # for ref_decoded, hyp_decoded in zip(refs_np_decoded, hyps_decoded):
       #     missing = 0
       #     not_missing = 0
       #     for hyp_token in hyp_decoded.split(" "):
       #         if hyp_token not in ref_decoded.split(" "):
       #             missing += 1
       #         else:
       #             not_missing += 1
      #      costs.append(max(1, min(missing, not_missing)))
        #print(costs)

        # other way around: how many words occur in ref?
        # or: how many got a reward of 1?
        # or min(#1s, #0s)
        # how many were selected?

        assert len(costs) == selected_batch_size

        assert chunk_loss.size(0) == selected_batch_size
        return chunk_loss.sum(), selected_tokens, selected_batch_size, costs

    def _full_sup_loss(self, selection, decoder_out, criterion, target,
                       batch_src_mask, max_output_length, encoder_hidden,
                       encoder_out, level,
                       beam_size=10, beam_alpha=1.0, logger=None, pe_ratio=1.0,
                       hyps=None,
                       hyp_inputs=None,
                       hyp_masks=None):
        """
        Compute the loss for fully-supervised training for the given, selection
        of indices of the batch

        loss: -log_p(reference | x)

        :param selection:
        :param src_mask:
        :return:
        """
        selected_target = torch.index_select(target, index=selection, dim=0)
        selected_decoder_out = torch.index_select(decoder_out, index=selection,
                                                  dim=0)
        selected_encoder_hidden = torch.index_select(encoder_hidden,
                                                     index=selection, dim=0)
        selected_encoder_out = torch.index_select(encoder_out, index=selection,
                                                  dim=0)
        selected_batch_size = selection.shape[0]
        selected_src_mask = torch.index_select(batch_src_mask,
                                               index=selection, dim=0)
        selected_tokens = (selected_target != self.pad_index).sum().cpu().numpy()

        # compute log probs of teacher-forced decoder for fully-supervised training
        tf_log_probs = F.log_softmax(selected_decoder_out, dim=-1)
        # in case of full supervision (teacher forcing with post-edit)
        full_sup_loss = criterion(
            input=tf_log_probs.contiguous().view(-1, tf_log_probs.size(-1)),
            target=selected_target.contiguous().view(-1)).view(selected_batch_size, -1).sum(-1)

        if hyps is not None and hyp_inputs is not None and hyp_masks is not None:
            # use logged hyp instead of bs output
            selected_hyps = torch.index_select(hyps, index=selection, dim=0)
            selected_hyp_inputs = torch.index_select(hyp_inputs,
                                                     index=selection, dim=0)
            selected_hyp_masks = torch.index_select(hyp_masks, index=selection,
                                                    dim=0)
            # logger.info("WEAK USING LOGGED HYP INSTEAD OF SAMPLE: {}".format(selected_hyps))
            # instead of sampling from the current model, take the log hypothesis
            selected_output = selected_hyps
            #sample_hyp_mask = selected_hyp_masks
            #sample_hyp_inputs = selected_hyp_inputs
            #sample_target = sample_hyp
            #selected_tokens = sample_hyp_mask.sum().cpu().numpy()

        else:
            # decode
            # post-edit: reference
            # hyp: beam search
            selected_output, _ = \
                beam_search(size=beam_size, encoder_output=selected_encoder_out,
                            encoder_hidden=selected_encoder_hidden,
                            src_mask=selected_src_mask, embed=self.trg_embed,
                            max_output_length=max_output_length,
                            alpha=beam_alpha, eos_index=self.eos_index,
                            pad_index=self.pad_index, bos_index=self.bos_index,
                            decoder=self.decoder)

        join_char = " " if level in ["word", "bpe"] else ""

        decoded_refs = [join_char.join(t) for t in arrays_to_sentences(selected_target,
                                          vocabulary=self.trg_vocab)]
        decoded_ref_list = [t.split(" ") for t in decoded_refs]
        #print("decoded_refs", decoded_refs)
        decoded_hyps = [join_char.join(t) for t in arrays_to_sentences(selected_output,
                                          vocabulary=self.trg_vocab)]
        decoded_hyp_list = [t.split(" ") for t in decoded_hyps]

        # compute cost: character edit distance? = number of characters to type
        chr_edit_distances = [pyter.edit_distance(list(h), list(r)) for h, r in
                              zip(decoded_hyps, decoded_refs)]
        costs = chr_edit_distances

        # TODO problematic: character edits -> might not be found in vocab
        if pe_ratio < 1.0:
            logger.info("PE ratio {}".format(pe_ratio))
            # white spaces and BPE marker are not considered for edit distance
            s = SequenceMatcher(lambda x: x in whitespace+"@", autojunk=False)
            costs = []
            post_edits = []
            for h, r in zip(decoded_hyps, decoded_refs):
                logger.info("hyp: {}".format(h))
                logger.info("ref: {}".format(r))
                #print("ref", r)
                #s.set_seq1(r)
                #s.set_seq2(h)
                #print(s.get_opcodes())
                #print([g for g in s.get_grouped_opcodes()])
                def edit(a, b, ratio=0.5):
                    # ratio: ratio of edit operations to perform (of those needed to reach reference)
                    i = 0
                    j = 0
                    cost = 0 # number of characters touched
                    s.set_seq1(a)
                    s.set_seq2(b)
                    codes = s.get_opcodes()
                    n = len([c[0] for c in codes if c[0] != "equal" ])*ratio
                    while i < n and j < len(codes):
                        #print("a", a)
                        #print("b", b)
                        #print("code", codes[j])
                        to_execute = codes[j]
                        if to_execute[0] == "equal":
                            j += 1
                            continue
                        elif to_execute[0] == "replace":
                            #print(a[to_execute[1]:to_execute[2]])
                            #print(b[to_execute[3]:to_execute[4]])
                            #print("a before replacement", a)
                            rep_len = len(b[to_execute[3]: to_execute[4]].replace("@", ""))
                            #print("rep len", rep_len)
                            a = a[:to_execute[1]] + b[to_execute[3]:to_execute[4]] + a[to_execute[2]:]
                            #print("after replacement", a)
                            #print("replace cost", max(to_execute[2]-to_execute[1], to_execute[4]-to_execute[3]))
                            cost += rep_len #max(to_execute[2]-to_execute[1], to_execute[4]-to_execute[3])
                            i += 1
                            s.set_seq1(a)
                            codes = s.get_opcodes()
                        elif to_execute[0] == "insert":
                            #print("a before insertion", a)
                            #print(a[to_execute[1]:to_execute[2]])
                            #print(b[to_execute[3]:to_execute[4]])
                            insert_len = len(b[to_execute[3]:to_execute[4]].replace("@", ""))
                            #print("insert len", insert_len)
                            a = a[:to_execute[2]] + b[to_execute[3]:to_execute[4]] + a[to_execute[2]:]
                            #print("a after insertion", a)
                            #print("insert cost", max(to_execute[2]-to_execute[1], to_execute[4]-to_execute[3]))
                            cost += insert_len #max(to_execute[2]-to_execute[1], to_execute[4]-to_execute[3])
                            i += 1
                            s.set_seq1(a)
                            codes = s.get_opcodes()
                        elif to_execute[0] == "delete":
                            #print(a[to_execute[1]:to_execute[2]])
                            #print(b[to_execute[3]:to_execute[4]])
                            # deduce the number of @s in the edited sequence to get correct cost
                            del_len = len(a[to_execute[1]:to_execute[2]].replace("@", ""))
                            #print("DEL LEN", del_len)
                            #print("before deletion", a)
                            a = a[:to_execute[1]] + a[to_execute[2]:]
                            #print("after deletion", a)
                            #print(a[:to_execute[1]])#.replace(" ", ""))
                            #print(a[:to_execute[2]]) #.replace(" ", ""))

                            #print("delete cost", max(to_execute[2]-to_execute[1], to_execute[4]-to_execute[3]))

                            cost += del_len #max(to_execute[2]-to_execute[1], to_execute[4]-to_execute[3])
                            i += 1
                            s.set_seq1(a)
                            codes = s.get_opcodes()
                    return a, cost

                post_edited_h, edit_cost = edit(h, r, ratio=0.6)
                logger.info("PE: {}".format(post_edited_h))
                logger.info("PE cost {}".format(edit_cost))
                post_edits.append(post_edited_h)
                costs.append(edit_cost)


            #print("edit distances", chr_edit_distances, sum(chr_edit_distances))

            #pe_edit_distances = [pyter.edit_distance(list(p), list(r)) for p, r in zip(post_edits, decoded_refs)]
            #print("after pe", pe_edit_distances, sum(pe_edit_distances))

            # now represent as indices
            pe_indices = sentences_to_arrays([p.split(" ") for p in post_edits], vocabulary=self.trg_vocab, pad_index=self.pad_index, max_length=selected_target.shape[1])
                                             #max_length=max_output_length)

            # add BOS
            bos_array = np.full(shape=(selected_batch_size, 1),
                                fill_value=self.bos_index)
            # prepend bos but cut off one bos
            pe_pad_bos = np.concatenate((bos_array, pe_indices),
                                                axis=1)[:, :-1]

            #print("PE INPUT", pe_pad_bos)
            #print("PE TRG", pe_indices)
            # teacher-force with PE now
            pe_out, _, _, _ = self.decode(encoder_output=selected_encoder_out,
                                              encoder_hidden=selected_encoder_hidden,
                                              trg_input=selected_src_mask.new(
                                                  pe_pad_bos).long(),
                                              src_mask=selected_src_mask,
                                              unrol_steps=pe_pad_bos.shape[
                                                  1])

            pe_log_probs = F.log_softmax(pe_out, dim=-1)

            # TODO what happens with BPEs?
            #print(arrays_to_sentences(pe_indices, vocabulary=self.trg_vocab, cut_at_eos=False))
            pe_sup_loss = criterion(
                input=pe_log_probs.contiguous().view(-1, pe_log_probs.size(-1)),
                target=tf_log_probs.new(pe_indices).long().contiguous().view(-1)).view(
                selected_batch_size, -1).sum(-1)
            full_sup_loss = pe_sup_loss
        #print("PE loss", pe_sup_loss)
        #print("full sup loss", full_sup_loss)


        # compute cost: TER*ref_len -> absolute number of edits
        #print("decoded_hyp", decoded_hyp)
        #ters = ster(hypotheses=decoded_hyp_list, references=decoded_ref_list)
        #print("ters", ters)
        #ref_lens = [len(t) for t in decoded_ref_list]
        #print("ref lens", ref_lens)
        #costs = [r*c for r, c in zip(ters, ref_lens)]
        #print("#edits", costs)
        assert full_sup_loss.size(0) == selected_batch_size
        return full_sup_loss.sum(), selected_tokens, selected_batch_size, costs
        #return pe_sup_loss.sum(), selected_tokens, selected_batch_size, costs

    def get_loss_for_batch(self, batch, criterion, regulate=False, pred=False,
                           max_output_length=100, chunk_type="marking", level="word",
                           entropy=False, weak_search="sample", weak_baseline=True,
                           weak_temperature=1.0, logger=None, case_sensitive=True,
                           pe_ratio=1.0, beam_size=10, beam_alpha=1.,
                           self_attention_drop=0.0, epsilon=0.5, regulator_sample=True):
        """
        Compute non-normalized loss and number of tokens for a batch

        :param batch:
        :param criterion:
        :return:
        """
        encoder_out, encoder_hidden, decoder_out = \
            self.forward(src=batch.src, trg_input=batch.trg_input,
                         src_mask=batch.src_mask, src_lengths=batch.src_lengths)
        out, hidden, att_probs, _ = decoder_out
        batch_size = batch.src.size(0)

        if not regulate:
            log_probs = F.log_softmax(out, dim=-1)
            batch_loss = criterion(
                input=log_probs.contiguous().view(-1, log_probs.size(-1)),
                target=batch.trg.contiguous().view(-1)).sum()
            batch_tokens = batch.ntokens
            batch_seqs = batch.nseqs
            reg_pred = None
            reg_log_probs = None
            batch_costs = None
            individual_losses = None

        else:
            # with regulator

            if self.regulator.feed_trg:
                hyp, _ = beam_search(size=beam_size,
                                            encoder_output=encoder_out,
                                            encoder_hidden=encoder_hidden,
                                            src_mask=batch.src_mask,
                                            embed=self.trg_embed,
                                            max_output_length=max_output_length,
                                            alpha=beam_alpha,
                                            eos_index=self.eos_index,
                                            pad_index=self.pad_index,
                                            bos_index=self.bos_index,
                                            decoder=self.decoder)
                hyp = batch.src.new(hyp).long()
            else:
                hyp = None
            #bos_array = np.full(shape=(batch_size, 1),
            #                    fill_value=self.bos_index)
            # prepend bos but cut off one bos
            #sample_hyp_pad_bos = np.concatenate((bos_array, bs_hyp),
            #                                    axis=1)[:, :-1]
            # print("with bos", bs_hyp_pad_bos)

            # treat bs output as target for forced decoding to get log likelihood of bs output
            #sample_out, _, _, _ = self.decode(
            #    encoder_output=encoder_out,
            #    encoder_hidden=encoder_hidden,
            #    trg_input=batch.src_mask.new(
            #        sample_hyp_pad_bos).long(),
            #    src_mask=batch.src_mask,
            #    unrol_steps=sample_hyp_pad_bos.shape[1])
            #sample_log_probs = F.log_softmax(sample_out, dim=-1)

            regulator_out = self.regulate(batch.src,
                                          batch.src_lengths,
                                          hyp=hyp)  # bs_target)
            reg_log_probs = F.log_softmax(regulator_out, dim=-1)

            # sample an output
            if regulator_sample:
                reg_dist = Categorical(logits=regulator_out)
                reg_pred = reg_dist.sample()
            else:
                reg_pred = torch.argmax(reg_log_probs, dim=-1)
                #logger.info("GREEDY OPTION {}".format(reg_pred))
                #logger.info("log probs {}".format(reg_log_probs))

            # heuristic: always choose one type of supervision
            if pred is not False:
                if pred == "random":
                    # random choice
                    fill_value = np.random.randint(0, self.regulator.output_size,
                                                   size=batch_size)
                    reg_pred = torch.from_numpy(
                        np.full(shape=(batch_size), fill_value=fill_value)).to(
                        regulator_out.device).long()
                elif pred == "uniform":
                    # fixed choice: [0, 1, 2, 3, 0, 1, 2, 3, 4 ... ]
                    reg_pred = torch.from_numpy(
                        (np.array([i for i in range(0, self.regulator.output_size)]
                         *batch_size)[:batch_size])).to(regulator_out.device).long()
                elif pred == "epsilon":
                    # epsilon-greedy
                    # draw random number between 0 and 1
                    exploit = np.random.uniform(low=0.0, high=1.0, size=(batch_size)) > epsilon
                    reg_pred = np.zeros(shape=(batch_size))
                    # if larger than epsilon: pick best action so far
                    for k, e in enumerate(exploit):
                        if e:
                            #print("REWARD STATS",self.rewards_per_output)
                            #print("COST STATS", self.costs_per_output)
                            stats = [np.mean(np.array(self.rewards_per_output[i])-np.array(self.costs_per_output[i])) if len(self.rewards_per_output[i]) > 0 else 0 for i in range(self.regulator.output_size)]
                            #print("EXPLOIT", stats, np.argmax(stats))
                            fill_value = np.argmax(stats)
                        # otherwise: pick one uniformly
                        else:
                            #print("EXPLORE")
                            fill_value = np.random.randint(0, high=self.regulator.output_size)
                        reg_pred[k] = fill_value
                    reg_pred = torch.from_numpy(reg_pred).to(
                        regulator_out.device).long()
                elif pred == "random-batch":
                    # random choice for whole batch
                    fill_value = np.random.randint(0,
                                                   self.regulator.output_size,
                                                   size=1)
                    reg_pred = torch.from_numpy(
                        np.full(shape=(batch_size), fill_value=fill_value)).to(
                        regulator_out.device).long()
                elif pred == "epsilon-batch":
                    # epsilon-greedy
                    # draw random number between 0 and 1
                    exploit = np.random.uniform(low=0.0, high=1.0) > epsilon
                    reg_pred = np.zeros(shape=(batch_size))
                    # if larger than epsilon: pick best action so far
                    if exploit:
                        #print("REWARD STATS",self.rewards_per_output)
                        #print("COST STATS", self.costs_per_output)
                        stats = [np.mean(np.array(self.rewards_per_output[i])-np.array(self.costs_per_output[i])) if len(self.rewards_per_output[i]) > 0 else 0 for i in range(self.regulator.output_size)]
                        #print("EXPLOIT", stats, np.argmax(stats))
                        fill_value = np.argmax(stats)
                    # otherwise: pick one uniformly
                    else:
                        #print("EXPLORE")
                        fill_value = np.random.randint(0, high=self.regulator.output_size)
                    reg_pred = torch.from_numpy(
                        np.full(shape=(batch_size), fill_value=fill_value)).to(
                        regulator_out.device).long()
                elif pred == "user":
                    # choose according to HTER: TODO never none
                    # if perfect: do self-training
                    # if great: weak feedback
                    # if okay: full feedback
                    sample_hyp, _ = beam_search(size=beam_size,
                                                encoder_output=encoder_out,
                                                encoder_hidden=encoder_hidden,
                                                src_mask=batch.src_mask,
                                                embed=self.trg_embed,
                                                max_output_length=max_output_length,
                                                alpha=beam_alpha,
                                                eos_index=self.eos_index,
                                                pad_index=self.pad_index,
                                                bos_index=self.bos_index,
                                                decoder=self.decoder)
                    decoded_hyps = arrays_to_sentences(sample_hyp, vocabulary=self.trg_vocab)
                    decoded_refs = arrays_to_sentences(batch.trg, vocabulary=self.trg_vocab)
                    #if level == "bpe":
                    #    decoded_hyps = [d.replace("@@ ", "") for d in decoded_hyps]
                    join_char = "" if level == "char" else " "
                    bleus = sbleu(hypotheses=[join_char.join(d).replace("@@ ", "") for d in decoded_hyps],
                                  references=[join_char.join(r).replace("@@ ", "") for r in decoded_refs])
                    # collect all ters
                    self.bleus.extend(bleus)
                    good = np.percentile(a=self.bleus, q=10, axis=0)
                    very_good = np.percentile(a=self.bleus, q=20, axis=0)
                    fill_value = np.zeros(shape=(batch_size))
                    for i, ter in enumerate(bleus):
                        if ter > very_good:
                            # self-supervision
                            label = "self"
                        elif ter > good:
                            # weak feedback
                            label = "weak"
                        else:  # full feedback
                            label = "full"
                        # TODO no skipping
                        fill_value[i] = self.regulator.label2index[label]
                    reg_pred = torch.from_numpy(fill_value).to(
                        regulator_out.device).long()

                elif pred == "uncertainty":
                    sample_hyp, _ = beam_search(size=beam_size,
                                                encoder_output=encoder_out,
                                                encoder_hidden=encoder_hidden,
                                                src_mask=batch.src_mask,
                                                embed=self.trg_embed,
                                                max_output_length=max_output_length,
                                                alpha=beam_alpha,
                                                eos_index=self.eos_index,
                                                pad_index=self.pad_index,
                                                bos_index=self.bos_index,
                                                decoder=self.decoder)
                    bos_array = np.full(shape=(batch_size, 1),
                                        fill_value=self.bos_index)
                    # prepend bos but cut off one bos
                    sample_hyp_pad_bos = np.concatenate((bos_array, sample_hyp),
                                                        axis=1)[:, :-1]
                    # print("with bos", bs_hyp_pad_bos)

                    # treat bs output as target for forced decoding to get log likelihood of bs output
                    sample_out, _, _, _ = self.decode(
                        encoder_output=encoder_out,
                        encoder_hidden=encoder_hidden,
                        trg_input=batch.src_mask.new(
                            sample_hyp_pad_bos).long(),
                        src_mask=batch.src_mask,
                        unrol_steps=sample_hyp_pad_bos.shape[1])
                    sample_log_probs = F.log_softmax(sample_out, dim=-1)

                    #print("sample log probs", sample_log_probs.shape)  # batch x time x vocab
                    avg_sample_entropy = -torch.sum(sample_log_probs*torch.exp(sample_log_probs), dim=2).mean(dim=1).detach().cpu().numpy()
                    #print("entropy", avg_sample_entropy)

                    self.entropies.extend(avg_sample_entropy)
                    #okay = np.percentile(a=self.entropies, q=50, axis=0)
                    #good = np.percentile(a=self.entropies, q=15, axis=0)
                    #very_good = np.percentile(a=self.entropies, q=5, axis=0)
                    uncertain = np.percentile(a=self.entropies, q=90, axis=0)
                    certain = np.percentile(a=self.entropies, q=80, axis=0)
                    fill_value = np.zeros(shape=(batch_size))
                    for i, ent in enumerate(avg_sample_entropy):
                        if ent < certain:
                            label = "self"
                        elif ent > uncertain:
                            label = "full"
                        else:
                            label = "weak"
                        #if ent < very_good:
                            # no supervision if very certain
                        #    label = "none"
                        #elif ent < good:
                            # self supervision when still certain
                        #    label = "self"
                        #elif ent < okay:
                            # weak supervision if not really certain
                        #    label = "weak"
                        #else:  # full feedback if uncertain
                        #    label = "full"
                        fill_value[i] = self.regulator.label2index[label]
                    reg_pred = torch.from_numpy(fill_value).to(
                        regulator_out.device).long()

                else:
                    fill_value = pred
                    reg_pred = torch.from_numpy(
                        np.full(shape=(batch_size), fill_value=fill_value)).to(
                        regulator_out.device).long()
                # print("regulator prediction", reg_pred)

            batch_loss = 0
            batch_tokens = 0
            batch_seqs = 0
            batch_costs = regulator_out.new_zeros(batch_size)
            individual_losses = regulator_out.new_zeros(batch_size)

            # split up the batch and sum the individual losses
            none_index = self.regulator.label2index.get("none", -1)
            self_index = self.regulator.label2index.get("self", -1)
            weak_index = self.regulator.label2index.get("weak", -1)
            full_index = self.regulator.label2index.get("full", -1)

            if none_index in reg_pred:
                # skip those: no loss
                zeros = torch.eq(reg_pred, none_index)
                zeros_idx = zeros.nonzero().squeeze(1)
                selected_srcs = torch.index_select(batch.src, dim=0,
                                                   index=zeros_idx)
                join_char = " " if level in ["word", "bpe"] else ""
                decoded_srcs = [join_char.join(t) for t in
                               arrays_to_sentences(selected_srcs,
                                                   vocabulary=self.src_vocab)]

                logger.info("Examples for no supervision:")
                for src in decoded_srcs[:3]:
                    logger.info("\tSkipping {}".format(src))
                # no cost
            if self_index in reg_pred:
                ones = torch.eq(reg_pred, self_index)
                ones_idx = ones.nonzero().squeeze(1)
                # compute self-train loss for those (smaller batch)
                self_sup_loss_selected, tokens, seqs = self._self_sup_loss(
                    selection=ones_idx, encoder_out=encoder_out, entropy=entropy,
                    encoder_hidden=encoder_hidden, src_mask=batch.src_mask,
                    max_output_length=max_output_length, criterion=criterion,
                    logger=logger, target=batch.trg, level=level,
                    beam_size=beam_size, beam_alpha=beam_alpha,
                    attention_drop=self_attention_drop,
                    hyps=batch.hyp if hasattr(batch, 'hyp') else None,
                    hyp_inputs=batch.hyp_input if hasattr(batch, 'hyp_input') else None,
                    hyp_masks=batch.hyp_mask if hasattr(batch, 'hyp_mask') else None)
                #print("self sup selected", self_sup_loss_selected)
                batch_loss += self_sup_loss_selected.sum()
                batch_tokens += tokens
                batch_seqs += seqs
                # no cost
                individual_losses[ones_idx] = self_sup_loss_selected

            if weak_index in reg_pred:
                # compute weak-sup. loss for those
                twos = torch.eq(reg_pred, weak_index)
                twos_idx = twos.nonzero().squeeze(1)
                weak_sup_loss_selected, tokens, seqs, costs = self._weak_sup_loss(
                    selection=twos_idx, src=batch.src, encoder_out=encoder_out,
                    encoder_hidden=encoder_hidden, src_mask=batch.src_mask,
                    max_output_length=max_output_length, criterion=criterion,
                    chunk_type=chunk_type, level=level,
                    target=batch.trg, weak_temperature=weak_temperature,
                    weak_search=weak_search, beam_size=beam_size, beam_alpha=beam_alpha,
                    weak_baseline=weak_baseline, logger=logger, case_sensitive=case_sensitive,
                    hyps=batch.hyp if hasattr(batch, 'hyp') else None,
                    hyp_inputs=batch.hyp_input if hasattr(batch, 'hyp_input') else None,
                    hyp_masks=batch.hyp_mask if hasattr(batch, 'hyp_mask') else None)
                #print("weak sup selected", weak_sup_loss_selected)
                batch_loss += weak_sup_loss_selected.sum()
                batch_tokens += tokens
                batch_seqs += seqs
                # get cost: number of words that need to be marked (incorrect)
                # write at the right position of cost vector
                batch_costs[twos_idx] = batch_costs.new(costs)
                individual_losses[twos_idx] = weak_sup_loss_selected

            if full_index in reg_pred:
                # compute fully-sup. loss for those
                threes = torch.eq(reg_pred, full_index)
                threes_idx = threes.nonzero().squeeze(1)
                full_sup_loss_selected, tokens, seqs, costs = self._full_sup_loss(
                    selection=threes_idx, decoder_out=out,
                    criterion=criterion, target=batch.trg,
                    encoder_hidden=encoder_hidden, level=level,
                    beam_size=beam_size, beam_alpha=beam_alpha,
                    encoder_out=encoder_out, max_output_length=max_output_length,
                    batch_src_mask=batch.src_mask, logger=logger, pe_ratio=pe_ratio,
                    hyps=batch.hyp if hasattr(batch, 'hyp') else None,
                    hyp_inputs=batch.hyp_input if hasattr(batch, 'hyp_input') else None,
                    hyp_masks=batch.hyp_mask if hasattr(batch, 'hyp_mask') else None)
               # print("full sup selected", full_sup_loss_selected)
                batch_loss += full_sup_loss_selected
                batch_tokens += tokens
                batch_seqs += seqs
                # get cost: # edits needed in hyp
                # write at the right position of cost vector
                batch_costs[threes_idx] = batch_costs.new(costs)
                individual_losses[threes_idx] = full_sup_loss_selected

                selected_srcs = torch.index_select(batch.src, dim=0,
                                                   index=threes_idx)
                join_char = " " if level in ["word", "bpe"] else ""
                decoded_srcs = [join_char.join(t) for t in
                                arrays_to_sentences(selected_srcs,
                                                    vocabulary=self.src_vocab)]

                logger.info("Examples for full supervision:")
                for src in decoded_srcs[:3]:
                    logger.info("\tFull supervision for: {}".format(src))

            if type(batch_loss) == int:  # no update for batch
                batch_loss = None

        return batch_loss, reg_log_probs, reg_pred, batch_tokens, batch_seqs, \
               batch_costs, individual_losses

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
