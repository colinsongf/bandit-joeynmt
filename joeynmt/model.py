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
from joeynmt.search import beam_search, greedy
from joeynmt.vocabulary import Vocabulary
from torch.distributions import Categorical
from joeynmt.metrics import sbleu
from joeynmt.helpers import arrays_to_sentences



def build_model(cfg: dict = None,
                src_vocab: Vocabulary = None,
                trg_vocab: Vocabulary = None):
    src_padding_idx = src_vocab.stoi[PAD_TOKEN]
    trg_padding_idx = trg_vocab.stoi[PAD_TOKEN]

    src_embed = Embeddings(
        **cfg["encoder"]["embeddings"], vocab_size=len(src_vocab),
        padding_idx=src_padding_idx)
    reg_src_embed = Embeddings(
        **cfg["regulator"]["embeddings"], vocab_size=len(src_vocab),
        padding_idx=src_padding_idx)

    if cfg.get("tied_embeddings", False) \
        and src_vocab.itos == trg_vocab.itos:
        # share embeddings for src and trg
        trg_embed = src_embed
        reg_trg_embed = reg_src_embed
    else:
        trg_embed = Embeddings(
            **cfg["decoder"]["embeddings"], vocab_size=len(trg_vocab),
            padding_idx=trg_padding_idx)
        reg_trg_embed = Embeddings(
            **cfg["regulator"]["embeddings"], vocab_size=len(trg_vocab),
            padding_idx=trg_padding_idx)

    encoder = RecurrentEncoder(**cfg["encoder"],
                               emb_size=src_embed.embedding_dim)
    decoder = RecurrentDecoder(**cfg["decoder"], encoder=encoder,
                               vocab_size=len(trg_vocab),
                               emb_size=trg_embed.embedding_dim)
    regulator = RecurrentRegulator(**cfg["regulator"],
                                   src_emb_size=reg_src_embed.embedding_dim,
                                   trg_emb_size=reg_trg_embed.embedding_dim)

    model = Model(encoder=encoder, decoder=decoder, regulator=regulator,
                  src_embed=src_embed, trg_embed=trg_embed,
                  reg_src_embed=reg_src_embed, reg_trg_embed=reg_trg_embed,
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
        # TODO include regulator
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

    def regulate(self, src, hyp):
        """

        :param src:
        :param hyp:
        :return:
        """
        return self.regulator(src=self.reg_src_embed(src),
                              hyp=self.reg_trg_embed(hyp))

    def get_loss_for_batch(self, batch, criterion, regulate=False, pred=False, max_output_length=100, chunk_type="marking", level="word"):
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

        # compute log probs of teacher-forced decoder for fully-supervised training
        tf_log_probs = F.log_softmax(out, dim=-1)
        batch_size = batch.trg.size(0)
        # in case of full supervision (teacher forcing with post-edit)
        pe_loss = criterion(
            input=tf_log_probs.contiguous().view(-1, tf_log_probs.size(-1)),
            target=batch.trg.contiguous().view(-1)).view(batch_size, -1).sum(-1)
        assert pe_loss.size(0) == batch_size
        #print("pe_loss", pe_loss)
        # batch*length

        if regulate:

            # compute outputs that are presented to user (BS)
           # max_output_length = int(max(batch.src_lengths.cpu().numpy()) * 1.5)
            beam_size = 10
            beam_alpha = 1.0
            bs_hyp, _ = beam_search(size=beam_size, encoder_output=encoder_out,
                                encoder_hidden=encoder_hidden,
                                src_mask=batch.src_mask, embed=self.trg_embed,
                                max_output_length=max_output_length,
                                alpha=beam_alpha, eos_index=self.eos_index,
                                pad_index=self.pad_index, bos_index=self.bos_index,
                                decoder=self.decoder)

            # padded beam search target
            trg_np = batch.trg.detach().cpu().numpy()
            #print("bs hyp", bs_hyp)
            bs_hyp_pad = np.full(shape=(batch_size,max_output_length),
                                 fill_value=self.pad_index)
            for i, row in enumerate(bs_hyp):
                for j, col in enumerate(row):
                    bs_hyp_pad[i, j] = col
            #print("padded", bs_hyp_pad)
            bos_array = np.full(shape=(batch_size, 1), fill_value=self.bos_index)
            # prepend bos but cut off one bos
            bs_hyp_pad_bos = np.concatenate((bos_array, bs_hyp_pad), axis=1)[:, :-1]
            #print("with bos", bs_hyp_pad_bos)

            # treat bs output as target for forced decoding to get log likelihood of bs output
            bs_out, _, _, _ = self.decode(encoder_output=encoder_out,
                                       encoder_hidden=encoder_hidden,
                                       trg_input=batch.trg.new(bs_hyp_pad_bos).long(),
                                       src_mask=batch.src_mask,
                                       unrol_steps=bs_hyp_pad_bos.shape[1])
            bs_log_probs = F.log_softmax(bs_out, dim=-1)
            #greedy_log_prob = self.force_decode(encoder_output=encoder_out,
            #                                 encoder_hidden=encoder_hidden,
            #                                 trg_input=batch.trg.new(greedy_hyp).long(),
            #                                 src_mask=batch.src_mask)

            bs_target = batch.trg.new(bs_hyp_pad).long()

            bs_nll = criterion(
                input=bs_log_probs.contiguous().view(-1, bs_log_probs.size(-1)),
                target=bs_target.view(-1))

            if chunk_type == "marking":
                # in case of markings: "chunk-based" feedback: nll of bs weighted by 0/1
                # 1 if correct, 0 if incorrect
                # fill bs_hyp with padding, since different length
                #print("bs", bs_hyp_pad)
                #print("trg", trg_np)
                # padding area is zero
                markings = np.zeros_like(bs_hyp_pad, dtype=float)
                for i, row in enumerate(bs_hyp):
                    for j, val in enumerate(row):
                        try:
                            if trg_np[i,j] == val:
                                markings[i,j] = 1.
                        except IndexError: # BS is longer than trg
                                continue
                chunk_loss = (bs_nll.view(batch_size, -1) * batch.trg.new(
                    markings).float()).sum(1)

            elif chunk_type == "sbleu":
                # decode hypothesis and target
                join_char = " " if level in ["word", "bpe"] else ""
                bs_hyp_decoded = [join_char.join(t) for t in
                                  arrays_to_sentences(arrays=bs_hyp,
                                            vocabulary=self.trg_vocab,
                                            cut_at_eos=True)]
                trg_np_decoded = [join_char.join(t) for t in
                                  arrays_to_sentences(arrays=trg_np,
                                            vocabulary=self.trg_vocab,
                                            cut_at_eos=True)]
                assert len(trg_np_decoded) == len(bs_hyp_decoded)
                # compute sBLEUs
                sbleus = np.array(sbleu(bs_hyp_decoded, trg_np_decoded))
                # use same reward for all the tokens
                chunk_loss = bs_nll.sum(-1)*(1-batch.trg.new(
                    sbleus)).float()

           # print("trg", batch.trg.detach().numpy())
            #print("markings", markings)
            # no need to add trg mask -> padding is masked automatically
            #print("bs log probs", bs_log_probs.shape)

            #print("bs nll", bs_nll)
            #print("markings", markings)
            # bs hyp might not have the same length as tf seq
            # TODO prevent model from preferring shorter ones
            assert chunk_loss.size(0) == batch_size
            #print("chunk loss", chunk_loss)

            # in case of self-supervision: run BS to use as target instead
            # costs: 0
            # sum over time dimension
            self_sup_loss = bs_nll.view(batch_size, -1).sum(-1)  # batch
            assert self_sup_loss.size(0) == batch_size
            #print("self sup loss", self_sup_loss)
            # weigh by confidence = mean(prob(sample))
            #confidence = torch.exp(-bs_nll.view(batch_size, -1).mean(-1))
            entropy = (-torch.exp(bs_log_probs)*bs_log_probs).sum(-1).mean(1)
            #print("entropy", entropy)
            self_sup_loss = self_sup_loss - entropy#*confidence.detach()
            #print("weighted", self_sup_loss)

            regulator_out = self.regulate(batch.src, bs_target)
            reg_log_probs = F.log_softmax(regulator_out, dim=-1)

            #print("reg_log_probs", reg_log_probs.shape)

            # sample an output
            reg_dist = Categorical(logits=regulator_out)
            #reg_pred = torch.argmax(regulator_out, dim=-1)
            reg_pred = reg_dist.sample()
            #print("regulator prediction", reg_pred)

            # heuristic: always choose one type of supervision
            if pred is not False:
                if pred == "random":
                    # random choice
                    fill_value = np.random.randint(0, 4, size=batch_size)
                else:
                    fill_value = pred
                reg_pred = torch.from_numpy(np.full(shape=(batch_size), fill_value=fill_value)).to(regulator_out.device).long()
            #print("regulator prediction", reg_pred)

            #one_hot_reg_pred = torch.eye(
            #    self.regulator.output_size).index_select(dim=0, index=reg_pred)
            #print("one hot", one_hot_reg_pred)

            # now decide which loss counts for which batch
            # every element in the batch could get a different type of feedback
            # so we have to iterate over the batch
            # or compute losses for all options always and then weigh them
            # which one is more expensive?
            # probably option 1
            # -> regulator predicts one-hot weighting of losses for each input
            #print("regulator pred", one_hot_reg_pred.detach().numpy())

            # need a matrix with
            # [none, self, chunk, post, ]-losses
            #none_loss = self_sup_loss.new_zeros(size=(batch_size,))
            #all_losses = torch.stack([none_loss, self_sup_loss, chunk_loss, pe_loss], dim=1)
            #print("all losses", all_losses)
            # masking out those losses that were not chosen for batch
            #batch_loss = (one_hot_reg_pred.detach()*all_losses).sum(1)
            batch_loss = 0
            batch_tokens = 0
            batch_seqs = 0
            # TODO check if losses balanced? norm needed? avg over batch?
            # TODO rather loop over loss and sum
            #print(reg_pred)
            for i, p in enumerate(reg_pred):
                if p == 0:
                    continue
                elif p == 1:
                    batch_loss += self_sup_loss[i]
                    batch_tokens += bs_hyp[i].size
                    batch_seqs += 1
                elif p == 2:
                    batch_loss += chunk_loss[i]
                    batch_seqs += 1
                    if chunk_type == "marking":
                        batch_tokens += markings[i].sum()
                    elif chunk_type == "sbleu":
                        batch_tokens += bs_hyp[i].size
                elif p == 3:
                    batch_loss += pe_loss[i]
                    batch_seqs += 1
                    batch_tokens += (batch.trg[i] != self.pad_index).sum().cpu().numpy()
            if type(batch_loss) == int:
                batch_loss = None
            #print(batch_tokens, batch_seqs, batch.ntokens, batch.nseqs)
        else:
            batch_loss = pe_loss.sum(0)  # with regulator summing is not done within criterion
            reg_log_probs = None
            reg_pred = None
            batch_tokens = batch.ntokens
            batch_seqs = batch.nseqs

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
