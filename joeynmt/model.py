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
from joeynmt.discrete_correctors import RecurrentDiscreteCorrector
from joeynmt.constants import PAD_TOKEN, EOS_TOKEN, BOS_TOKEN
from joeynmt.search import beam_search, greedy
from joeynmt.vocabulary import Vocabulary
from joeynmt.helpers import arrays_to_sentences
from joeynmt.metrics import token_accuracy, bleu, f1_bin, \
    token_edit_reward, token_recall_reward, token_lcs_reward


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
    #corrector = RecurrentCorrector(**cfg["corrector"],
    #                               trg_embed=trg_embed, encoder=encoder,
    #                               decoder_size=decoder.hidden_size)

    corrector = RecurrentDiscreteCorrector(**cfg["corrector"],
                                           encoder=encoder,
                                           vocab_size=len(trg_vocab),
                                           emb_size=trg_embed.embedding_dim,
                                           prev_hidden_size=decoder.hidden_size)

    model = Model(encoder=encoder, decoder=decoder, corrector=corrector,
                  src_embed=src_embed, trg_embed=trg_embed,
                  src_vocab=src_vocab, trg_vocab=trg_vocab)
    simulated_marking_function = cfg.get(
        "simulated_marking_function")
    if simulated_marking_function == "accuracy":
        model.marking_fun = lambda hyp, ref: token_edit_reward(
            gold=ref, pred=hyp, shifted=model.corrector.shift_rewards)
    elif simulated_marking_function == "recall":
        model.marking_fun = lambda hyp, ref: token_recall_reward(gold=ref,
                                                                pred=hyp)
    elif simulated_marking_function == "lcs":
        model.marking_fun = lambda hyp, ref: token_lcs_reward(gold=ref,
                                                             pred=hyp)
    else:
        model.marking_fun = None

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
        :param correct: run corrector part as well
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
            # TODO decoder predictions: what if beam search? (for training always greedy)
            greedy_pred = torch.argmax(outputs, dim=-1) #.cpu().numpy()  # batch x length

            #rev_predicted, rev_pred_mask, pred_length = \
             #   self._revert_prepare_seq(seq=greedy_pred, aux_tensor=outputs)

            # predict corrections
            #corrections, rewards, corr_src_att_probs = self.correct(
            #    y=rev_predicted, y_length=pred_length,
            #    mask=rev_pred_mask, y_states=att_vectors,
            #    encoder_output=encoder_output, src_mask=src_mask)
            comb_states = self.corrector.decoder_bridge(
                prev_states=att_vectors.detach(),
                prev_outputs=self.trg_embed(greedy_pred.detach()))
            # batch x trg_len x emb+hidden

            # run decoder again with corrections*(1-rewards)
            # if reward is 1 -> no correction
            corrector_output = self.correct(encoder_output=encoder_output,
                         encoder_hidden=encoder_hidden,
                         src_mask=src_mask,
                         trg_input=trg_input,
                         unrol_steps=unrol_steps,
                         comb_states=comb_states)
            corr_outputs, corr_hidden, corr_src_att_probs, \
            corr_att_vectors, rewards = corrector_output

            return outputs, greedy_pred, rewards, corr_outputs, corr_hidden, \
                   corr_src_att_probs

        return decoder_output

    def _revert_prepare_seq(self, seq, aux_tensor):
        """
        Revert a sequence of predictions (np.array, no gradient flow!)
        and generate mask and length
        :param seq: batch_size x time
        :param aux_tensor: tensor to create new tensor like
        :return:
        """
        # compute mask with numpy
        eos = np.where(seq == self.trg_vocab.stoi[EOS_TOKEN], 1, 0)
        # mark everything after first eos with 1, even if non-eos
        eos_filled = np.where(np.cumsum(eos, axis=1) >= 1, 1, 0)
        # then create mask with 0s after first eos
        mask = np.where(np.cumsum(eos_filled, axis=1) > 1, 0, 1)

        # reverse the prediction to read it in backwards
        # flip in time (copy is needed for contiguity)
        rev_seq = aux_tensor.new_tensor(
            np.flip(seq, axis=1).copy(), dtype=torch.long)
        rev_seq_mask = aux_tensor.new_tensor(np.flip(mask, axis=1).copy(),
                                             dtype=torch.uint8)
        seq_length = rev_seq_mask.sum(1)
        return rev_seq, rev_seq_mask, seq_length

    def correct(self, encoder_output, encoder_hidden, src_mask, trg_input,
                unrol_steps, comb_states,
                decoder_hidden=None):
        """
        Run the corrector to predict corrections for hidden states
        :param y:
        :param y_length:
        :param mask:
        :param y_states:
        :return:
        """
        #return self.corrector(self.trg_embed(y), y_length, mask,
        #                      y_states, encoder_output.detach(), src_mask)
        return self.corrector(trg_embed=self.trg_embed(trg_input),
                            encoder_output=encoder_output,
                            encoder_hidden=encoder_hidden,
                            src_mask=src_mask,
                            unrol_steps=unrol_steps,
                            hidden=decoder_hidden,
                            comb_states=comb_states,
                            )

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

    def get_loss_for_batch(self, batch, criterion,
                           logging_fun=None, marking_fun=None):
        """
        Compute non-normalized loss for batch for corrector and mt

        :param batch:
        :param criterion:
        :param logging_fun:
        :param marking_fun: marking function for inducing token-feedback
        :return:
        """
        mt_outputs, original_pred, rewards, corr_outputs, corr_hidden, \
            src_corr_att_probs = self.forward(
                src=batch.src, trg_input=batch.trg_input, correct=True,
                src_mask=batch.src_mask, src_lengths=batch.src_lengths)

        # reward model is trained to predict whether mt predictions are correct
        # the targets for this model are computed dynamically

        reward_targets = np.expand_dims(
            marking_fun(original_pred.cpu().numpy(), batch.trg.cpu().numpy()), 2)
        #    token_lcs_reward(
          #  token_edit_reward(
        #        batch.trg.cpu().numpy(), original_pred.astype(int)),
               # shifted=self.corrector.shift_rewards),
        #    2)

        assert reward_targets.shape == rewards.shape  # batch x time x 1

        # loss for MT
        # compute log probs
        mt_log_probs = F.log_softmax(mt_outputs, dim=-1)

        # compute batch loss
        mt_loss = criterion(
            input=mt_log_probs.contiguous().view(-1, mt_log_probs.size(-1)),
            target=batch.trg.contiguous().view(-1))

        # loss is MSE between targets and predictions
        reward_loss = torch.mean(
            (rewards.new(reward_targets)-rewards)**2)

        # loss for correction: log-likelihood of reference under corrected model
        # compute log probs of correction
        corr_log_probs = F.log_softmax(corr_outputs, dim=-1)

        # compute batch loss for corrector
        corrector_loss = criterion(
            input=corr_log_probs.contiguous().view(-1, corr_log_probs.size(-1)),
            target=batch.trg.contiguous().view(-1))

        #print("mt loss", mt_loss)
        #print("corr loss", corrector_loss)
        #print("reward", reward_loss)
        #print("*coeff", reward_loss*self.corrector.reward_coeff)

        # reward loss gets weighed by a coefficient since it's on a diff. scale
        corr_loss = corrector_loss+self.corrector.reward_coeff*reward_loss

        if logging_fun is not None:
            logging_fun("before corr: {}".format(
                " ".join(arrays_to_sentences(
                    original_pred, vocabulary=self.trg_vocab)[0])))
            corr_pred = torch.argmax(corr_outputs, dim=2).cpu().numpy()
            logging_fun("after corr: {}".format(
                " ".join(arrays_to_sentences(
                    corr_pred, vocabulary=self.trg_vocab)[0])))
            logging_fun("ref: {}".format(
                " ".join(arrays_to_sentences(
                    batch.trg, vocabulary=self.trg_vocab)[0])))
            logging_fun("total corrector loss: {}; "
                        "corrector xent: {:.2f} ({:.2f}\%), "
                        "reward MSE: {:.2f} ({:.2f}\%) ".format(
                            corr_loss,
                            corrector_loss, corrector_loss/corr_loss*100,
                            reward_loss, reward_loss/corr_loss*100))
            logging_fun("mt loss {:.2f}, corr loss {:.2f}".format(mt_loss,
                                                                  corr_loss))

        return mt_loss, corr_loss



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

        # first pass decoding
        # greedy decoding
        if beam_size == 0:
            stacked_output, stacked_attention_scores, \
            stacked_att_vectors = greedy(
                encoder_hidden=encoder_hidden, encoder_output=encoder_output,
                src_mask=batch.src_mask, embed=self.trg_embed,
                bos_index=self.bos_index, decoder=self.decoder,
                max_output_length=max_output_length)
            # batch, time, max_src_length
        else:  # beam size
            stacked_output, stacked_attention_scores, stacked_att_vectors = \
                beam_search(size=beam_size, encoder_output=encoder_output,
                            encoder_hidden=encoder_hidden,
                            src_mask=batch.src_mask, embed=self.trg_embed,
                            max_output_length=max_output_length,
                            alpha=beam_alpha, eos_index=self.eos_index,
                            pad_index=self.pad_index, bos_index=self.bos_index,
                            decoder=self.decoder,
                            src_lengths=batch.src_lengths,
                            return_attention=True,
                            return_attention_vectors=True)
        # TODO decoder predictions: for training always greedy

        # pre-compute projected encoder outputs
        # (the "keys" for the attention mechanism)
        # this is only done for efficiency
        #if hasattr(self.corrector.attention, "compute_proj_keys"):
        #    self.corrector.attention.compute_proj_keys(encoder_output)

        # move output into torch again
        torch_stacked_output = encoder_output.new(stacked_output).long()

        # bridge two decoders
        #rev_predicted, rev_pred_mask, pred_length = \
        #    self._revert_prepare_seq(seq=stacked_output,
        #                             aux_tensor=encoder_output)
        #print(rev_predicted.shape)
        #rnn_outputs = self.corrector._read_rnn(input=self.trg_embed(rev_predicted))
        # print("bw rnn output", rnn_outputs.shape)  # batch x time x hidden

        # concat with y_states
        #comb_states = torch.cat([stacked_att_vectors, rnn_outputs],
        #                        dim=2)  # batch x time x decoder.hidden_size+hidden
        # Problem: comb_states might be shorter than max_length
        comb_states = self.corrector.decoder_bridge(
            prev_states=encoder_output.new(stacked_att_vectors).detach(),
            prev_outputs=self.trg_embed(torch_stacked_output.detach()))

        # run corrector
        if beam_size == 0:
            corrected_stacked_output, corrected_stacked_attention_scores, _, \
                rewards = \
                greedy(
                        encoder_hidden=encoder_hidden,
                        encoder_output=encoder_output,
                        src_mask=batch.src_mask, embed=self.trg_embed,
                        bos_index=self.bos_index, decoder=self.corrector,
                        max_output_length=max_output_length,
                        comb_states=comb_states)
            # batch, time, max_src_length
        else:  # beam size
            # TODO track and return rewards
            rewards = None
            corrected_stacked_output, corrected_stacked_attention_scores, _, = \
                beam_search(size=beam_size, encoder_output=encoder_output,
                            encoder_hidden=encoder_hidden,
                            src_mask=batch.src_mask, embed=self.trg_embed,
                            max_output_length=max_output_length,
                            alpha=beam_alpha, eos_index=self.eos_index,
                            pad_index=self.pad_index,
                            bos_index=self.bos_index,
                            decoder=self.corrector,
                            src_lengths=batch.src_lengths,
                            return_attention_vectors=False,
                            return_attention=True,
                            comb_states=comb_states)
            #print("Corrected stacked out", corrected_stacked_output)

        return stacked_output, stacked_attention_scores, \
               corrected_stacked_output, corrected_stacked_attention_scores, \
               rewards

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
