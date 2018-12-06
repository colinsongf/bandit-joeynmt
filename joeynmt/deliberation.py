# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from joeynmt.helpers import tile
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
        #print("dec1_hidden", dec1_hidden[0].shape)  # 1x batch x hidden size
        #print("dec1_outputs", dec1_outputs.shape)  # batch x length x vocab
        #print("dec_att_vectors", dec1_att_vectors.shape)  # batch x length x hidden
        # TODO pass on outputs or hidden? paper says hidden states
        # what if we use attentional vectors instead? (would also contain context)
        d1_states = dec1_att_vectors.detach()  # don't backprop from d2 through d1
        # TODO original paper uses beam seach with k=2 here, let's use greedy search for efficiency
        d1_greedy = torch.argmax(dec1_outputs, dim=-1).detach()  # don't backprop through here
        d1_predictions = d1_greedy # TODO backprop through embeddings though?  # batch x max_length
        #print("d1 pred", d1_predictions.shape)
        # TODO is there a target mask?? pro: ignore after </s>, con: full knowledge
        # zero out everything after last eos
        # eos indicator tensor: 1 if eos
        eos = torch.where(d1_greedy.eq(self.trg_vocab.stoi[EOS_TOKEN]), d1_greedy.new_full([1], 1), d1_greedy.new_full([1], 0))
        # mask all positions after the first eos to 0 (cumsum of eoses is > 1)
        trg_mask = torch.where(torch.cumsum(eos, dim=1).gt(1), d1_greedy.new_full([1], 0), d1_greedy.new_full([1], 1)).unsqueeze(1).byte()
        #print("trg mask", trg_mask.shape)  # batch x 1 x max_length
        # trg_mask = d1_states.new_ones(d1_states.shape[0], 1, d1_states.shape[1]).byte()

        dec2_outputs = self.decode2(encoder_output=encoder_output,
                                    encoder_hidden=encoder_hidden,
                                    src_mask=src_mask, trg_input=trg_input,
                                    unrol_steps=unrol_steps,
                                    trg_mask=trg_mask,
                                    d1_states=d1_states,
                                    d1_predictions=d1_predictions)
        #print("src att1", dec1_att_probs)
        #print("src_att2", dec2_outputs[3])
        #print("dec_att2", dec2_outputs[4])

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
                             d1_predictions=self.trg_embed(d1_predictions),
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
          #  stacked_output1, stacked_attention_scores1 = greedy(
          #      encoder_hidden=encoder_hidden, encoder_output=encoder_output,
          #      src_mask=batch.src_mask, embed=self.trg_embed,
          #      bos_index=self.bos_index, decoder=self.decoder1,
          #      max_output_length=max_output_length)
            # stacked_output.cpu().numpy(), \
           #stacked_attention_scores.cpu().numpy(), \
           #stacked_output2.cpu().numpy(), \
           #stacked_src_attention_scores.cpu().numpy(), \
           #stacked_d1_attention_scores.cpu().numpy()
            stacked_output1, stacked_attention_scores, stacked_output2, stacked_src_attention_scores, stacked_d1_attention_scores = greedy_delib(
                encoder_hidden=encoder_hidden, encoder_output=encoder_output,
                src_mask=batch.src_mask, embed=self.trg_embed,
                bos_index=self.bos_index, decoders=[self.decoder1, self.decoder2],
                max_output_length=max_output_length, eos_index=self.trg_vocab.stoi[EOS_TOKEN])
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
            stacked_output2, stacked_attention_scores2 = \
                beam_search(size=beam_size, encoder_output=encoder_output,
                            encoder_hidden=encoder_hidden,
                            src_mask=batch.src_mask, embed=self.trg_embed,
                            max_output_length=max_output_length,
                            alpha=beam_alpha, eos_index=self.eos_index,
                            pad_index=self.pad_index, bos_index=self.bos_index,
                            decoders=[self.decoder1, self.decoder2])

        return stacked_output1, stacked_attention_scores, stacked_output2, stacked_src_attention_scores, stacked_d1_attention_scores

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


def greedy_delib(src_mask, embed, bos_index, max_output_length, decoders,
           encoder_output, encoder_hidden, eos_index):
    """
    Greedy decoding: in each step, choose the word that gets highest score.
    :param src_mask:
    :param embed:
    :param bos_index:
    :param max_output_length:
    :param decoders:
    :param encoder_output:
    :param encoder_hidden:
    :return:
    """
    decoder1, decoder2 = decoders
    batch_size = src_mask.size(0)
    prev_y = src_mask.new_full(size=[batch_size, 1], fill_value=bos_index,
                               dtype=torch.long)
    prev_y2 = src_mask.new_full(size=[batch_size, 1], fill_value=bos_index,
                               dtype=torch.long)
    output = []
    attention_scores = []
    attention_vectors = []
    hidden = None
    prev_att_vector = None
    for t in range(max_output_length):
        # run 1st decoder
        # decode one single step
        out, hidden, att_probs, prev_att_vector = decoder1(
            encoder_output=encoder_output,
            encoder_hidden=encoder_hidden,
            src_mask=src_mask,
            trg_embed=embed(prev_y),
            hidden=hidden,
            prev_att_vector=prev_att_vector,
            unrol_steps=1)
        # out: batch x time=1 x vocab (logits)

        # greedy decoding: choose arg max over vocabulary in each step
        next_word = torch.argmax(out, dim=-1)  # batch x time=1
        output.append(next_word.squeeze(1))
        prev_y = next_word
        attention_scores.append(att_probs.squeeze(1))
        attention_vectors.append(prev_att_vector.squeeze(1))
        # batch, max_src_lengths
    stacked_output = torch.stack(output, dim=1)  # batch, time
    stacked_attention_scores = torch.stack(attention_scores, dim=1)
    stacked_attention_vectors = torch.stack(attention_vectors, dim=1)

    d1_states = stacked_attention_vectors
    d1_greedy = stacked_output
    d1_predictions = d1_greedy

    # zero out everything after last eos
    # eos indicator tensor: 1 if eos
    eos = torch.where(d1_greedy.eq(eos_index),
                      d1_greedy.new_full([1], 1), d1_greedy.new_full([1], 0))
    # mask all positions after the first eos to 0 (cumsum of eoses is > 1)
    trg_mask = torch.where(torch.cumsum(eos, dim=1).gt(1),
                           d1_greedy.new_full([1], 0),
                           d1_greedy.new_full([1], 1)).unsqueeze(1).byte()
    output2 = []
    src_attention_scores = []
    d1_attention_scores = []
    # comb_attention_vectors = []
    hidden2 = None
    prev_comb_att_vector = None
    for t in range(max_output_length):
        # then run 2nd decoder
        out2, hidden2, src_att_probs, d1_att_probs, comb_att_vectors = \
            decoder2(encoder_output=encoder_output,
                                encoder_hidden=encoder_hidden,
                                src_mask=src_mask, trg_embed=embed(prev_y2),
                                unrol_steps=1,
                                hidden=hidden2,
                                trg_mask=trg_mask,
                                d1_states=d1_states,
                                d1_predictions=embed(d1_predictions),
                                prev_comb_att_vector=prev_comb_att_vector)

        # greedy decoding: choose arg max over vocabulary in each step
        next_word = torch.argmax(out2, dim=-1)  # batch x time=1
        output2.append(next_word.squeeze(1))
        prev_y2 = next_word
        src_attention_scores.append(src_att_probs.squeeze(1))
        d1_attention_scores.append(d1_att_probs.squeeze(1))
      #  comb_attention_vectors.append(prev_comb_att_vector)
        # batch, max_src_lengths
    stacked_output2 = torch.stack(output2, dim=1)  # batch, time
    stacked_src_attention_scores = torch.stack(src_attention_scores, dim=1)
    stacked_d1_attention_scores = torch.stack(d1_attention_scores, dim=1)
   # stacked_comb_attention_vectors = torch.stack(comb_attention_vectors, axis=1)

    return stacked_output.cpu().numpy(), stacked_attention_scores.cpu().numpy(), stacked_output2.cpu().numpy(), stacked_src_attention_scores.cpu().numpy(), \
                stacked_d1_attention_scores.cpu().numpy()

def beam_search_delib(decoders, size, bos_index, eos_index, pad_index, encoder_output,
                encoder_hidden, src_mask, max_output_length, alpha, embed,
                n_best=1):
    """
    Beam search with size k. Follows OpenNMT-py implementation.
    In each decoding step, find the k most likely partial hypotheses.
    `alpha` is the factor for length penalty.
    :param decoder:
    :param size:
    :param bos_index:
    :param eos_index:
    :param pad_index:
    :param encoder_output:
    :param encoder_hidden:
    :param src_mask:
    :param max_output_length:
    :param alpha:
    :param embed:
    :param n_best:
    :return:
    """
    # init
    batch_size = src_mask.size(0)
    hidden = decoder.init_hidden(encoder_hidden)

    # tile hidden decoder states and encoder output beam_size times
    hidden = tile(hidden, size, dim=1)  # layers x batch*k x dec_hidden_size
    att_vectors = None

    encoder_output = tile(encoder_output.contiguous(), size,
                          dim=0)  # batch*k x src_len x enc_hidden_size

    src_mask = tile(src_mask, size, dim=0)  # batch*k x 1 x src_len

    batch_offset = torch.arange(
        batch_size, dtype=torch.long, device=encoder_output.device)
    beam_offset = torch.arange(
        0,
        batch_size * size,
        step=size,
        dtype=torch.long,
        device=encoder_output.device)
    alive_seq = torch.full(
        [batch_size * size, 1],
        bos_index,
        dtype=torch.long,
        device=encoder_output.device)

    # Give full probability to the first beam on the first step.
    topk_log_probs = (torch.tensor([0.0] + [float("-inf")] * (size - 1),
                                   device=encoder_output.device).repeat(
        batch_size))

    # Structure that holds finished hypotheses.
    hypotheses = [[] for _ in range(batch_size)]

    results = {}
    results["predictions"] = [[] for _ in range(batch_size)]
    results["scores"] = [[] for _ in range(batch_size)]
    results["gold_score"] = [0] * batch_size

    for step in range(max_output_length):
        decoder_input = alive_seq[:, -1].view(-1, 1)

        # expand current hypotheses
        # decode one single step
        # out: logits for final softmax
        out, hidden, att_scores, att_vectors = decoder(
            encoder_output=encoder_output,
            encoder_hidden=encoder_hidden,
            src_mask=src_mask,
            trg_embed=embed(decoder_input),
            hidden=hidden,
            prev_att_vector=att_vectors,
            unrol_steps=1)

        log_probs = F.log_softmax(out, dim=-1).squeeze(1)  # batch*k x trg_vocab

        # multiply probs by the beam probability (=add logprobs)
        log_probs += topk_log_probs.view(-1).unsqueeze(1)
        curr_scores = log_probs

        # compute length penalty
        if alpha > -1:
            length_penalty = ((5.0 + (step + 1)) / 6.0) ** alpha
            curr_scores /= length_penalty

        # flatten log_probs into a list of possibilities
        curr_scores = curr_scores.reshape(-1, size * decoder.output_size)

        # pick currently best top k hypotheses (flattened order)
        topk_scores, topk_ids = curr_scores.topk(size, dim=-1)

        if alpha > -1:
            # recover original log probs
            topk_log_probs = topk_scores * length_penalty

        # reconstruct beam origin and true word ids from flattened order
        topk_beam_index = topk_ids.div(decoder.output_size)
        topk_ids = topk_ids.fmod(decoder.output_size)

        # map beam_index to batch_index in the flat representation
        batch_index = (
            topk_beam_index
            + beam_offset[:topk_beam_index.size(0)].unsqueeze(1))
        select_indices = batch_index.view(-1)

        # append latest prediction
        alive_seq = torch.cat(
            [alive_seq.index_select(0, select_indices),
             topk_ids.view(-1, 1)], -1)  # batch_size*k x hyp_len

        is_finished = topk_ids.eq(eos_index)
        if step + 1 == max_output_length:
            is_finished.fill_(1)
        # end condition is whether the top beam is finished
        end_condition = is_finished[:, 0].eq(1)

        # save finished hypotheses
        if is_finished.any():
            predictions = alive_seq.view(-1, size, alive_seq.size(-1))
            for i in range(is_finished.size(0)):
                b = batch_offset[i]
                if end_condition[i]:
                    is_finished[i].fill_(1)
                finished_hyp = is_finished[i].nonzero().view(-1)
                # store finished hypotheses for this batch
                for j in finished_hyp:
                    hypotheses[b].append((
                        topk_scores[i, j],
                        predictions[i, j, 1:])  # ignore start_token
                    )
                # if the batch reached the end, save the n_best hypotheses
                if end_condition[i]:
                    best_hyp = sorted(
                        hypotheses[b], key=lambda x: x[0], reverse=True)
                    for n, (score, pred) in enumerate(best_hyp):
                        if n >= n_best:
                            break
                        results["scores"][b].append(score)
                        results["predictions"][b].append(pred)
            non_finished = end_condition.eq(0).nonzero().view(-1)
            # if all sentences are translated, no need to go further
            if len(non_finished) == 0:
                break
            # remove finished batches for the next step
            topk_log_probs = topk_log_probs.index_select(0, non_finished)
            batch_index = batch_index.index_select(0, non_finished)
            batch_offset = batch_offset.index_select(0, non_finished)
            alive_seq = predictions.index_select(0, non_finished) \
                .view(-1, alive_seq.size(-1))

            # reorder indices, outputs and masks
            select_indices = batch_index.view(-1)
            encoder_output = encoder_output.index_select(0, select_indices)
            src_mask = src_mask.index_select(0, select_indices)

            if isinstance(hidden, tuple):
                # for LSTMs, states are tuples of tensors
                h, c = hidden
                h = h.index_select(1, select_indices)
                c = c.index_select(1, select_indices)
                hidden = (h, c)
            else:
                # for GRUs, states are single tensors
                hidden = hidden.index_select(1, select_indices)

            att_vectors = att_vectors.index_select(0, select_indices)

    def pad_and_stack_hyps(hyps, pad_value):
        filled = np.ones((len(hyps), max([h.shape[0] for h in hyps])),
                         dtype=int) * pad_value
        for j, h in enumerate(hyps):
            for k, i in enumerate(h):
                filled[j, k] = i
        return filled

    # from results to stacked outputs
    assert n_best == 1
    # only works for n_best=1 for now
    final_outputs = pad_and_stack_hyps([r[0].cpu().numpy() for r in
                                        results["predictions"]],
                                       pad_value=pad_index)

    # TODO also return attention scores and probabilities
    return final_outputs, None