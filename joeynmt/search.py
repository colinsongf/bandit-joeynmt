# coding: utf-8
import torch
import torch.nn.functional as F
import numpy as np

from joeynmt.helpers import tile


def greedy(src_mask, embed, bos_index, max_output_length, decoder,
           encoder_output, encoder_hidden, corrections=None):
    """
    Greedy decoding: in each step, choose the word that gets highest score.

    :param src_mask:
    :param embed:
    :param bos_index:
    :param max_output_length:
    :param decoder:
    :param encoder_output:
    :param encoder_hidden:
    :param corrections:
    :return:
    """
    batch_size = src_mask.size(0)
    prev_y = src_mask.new_full(size=[batch_size, 1], fill_value=bos_index,
                               dtype=torch.long)
    output = []
    attention_scores = []
    attention_vectors = []
    hidden = None
    prev_att_vector = None
    for t in range(max_output_length):
        # decode one single step
        out, hidden, att_probs, prev_att_vector = decoder(
            encoder_output=encoder_output,
            encoder_hidden=encoder_hidden,
            src_mask=src_mask,
            trg_embed=embed(prev_y),
            hidden=hidden,
            prev_att_vector=prev_att_vector,
            unrol_steps=1,
            corrections=corrections
        )
        # out: batch x time=1 x vocab (logits)

        # greedy decoding: choose arg max over vocabulary in each step
        next_word = torch.argmax(out, dim=-1)  # batch x time=1
        output.append(next_word.squeeze(1).cpu().numpy())
        prev_y = next_word
        attention_scores.append(att_probs.squeeze(1).cpu().numpy())
        attention_vectors.append(prev_att_vector.squeeze(1))
        # batch, max_src_lengths
    stacked_output = np.stack(output, axis=1)  # batch, time
    stacked_attention_scores = np.stack(attention_scores, axis=1)
    attention_vectors = torch.stack(attention_vectors, dim=1)  # not numpy
    return stacked_output, stacked_attention_scores, attention_vectors


def beam_search(decoder, size, bos_index, eos_index, pad_index, encoder_output,
                encoder_hidden, src_mask, src_lengths,
                max_output_length, alpha, embed,
                n_best=1, return_attention=False,
                return_attention_vectors=False,
                corrections=None):
    """
    Beam search with size k. Follows OpenNMT-py implementation.
    In each decoding step, find the k most likely partial hypotheses.

    :param decoder:
    :param size: size of the beam
    :param bos_index:
    :param eos_index:
    :param pad_index:
    :param encoder_output:
    :param encoder_hidden:
    :param src_mask:
    :param src_lengths:
    :param max_output_length:
    :param alpha: `alpha` factor for length penalty
    :param embed:
    :param n_best: return this many hypotheses, <= beam
    :param return_attention:
    :param return_attention_vectors:
    :param corrections:
    :return:
    """
    # init
    batch_size = src_mask.size(0)
    hidden = decoder.init_hidden(encoder_hidden)

    # tile hidden decoder states and encoder output beam_size times
    hidden = tile(hidden, size, dim=1)  # layers x batch*k x dec_hidden_size
    encoder_output = tile(encoder_output.contiguous(), size,
                          dim=0)  # batch*k x src_len x enc_hidden_size
    src_mask = tile(src_mask, size, dim=0)  # batch*k x 1 x src_len
    src_lengths = tile(src_lengths, size, dim=0)

    if corrections is not None:
        corrections = tile(corrections, size, dim=0)


    batch_offset = torch.arange(
        batch_size, dtype=torch.long, device=encoder_output.device)
    beam_offset = torch.arange(
        0,
        batch_size * size,
        step=size,
        dtype=torch.long,
        device=encoder_output.device)

    # prediction is filled with BOS
    alive_seq = torch.full(
        [batch_size * size, 1],
        bos_index,
        dtype=torch.long,
        device=encoder_output.device)
    # fill attn vectors as well for common indexing
#    print(src_lengths)
#    print(encoder_output.shape)
    alive_attn = torch.full(
        [batch_size * size, 1, encoder_output.shape[1]],
        0,
        dtype=torch.float,
        device=encoder_output.device)
    alive_attn_v = torch.full(
        [batch_size * size, 1, decoder.hidden_size],
        0,
        dtype=torch.float,
        device=encoder_output.device)
    #print("initial alive att", alive_attn.shape)

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
    results["att_probs"] = [[] for _ in range(batch_size)]
    results["att_vectors"] = [[] for _ in range(batch_size)]

    prev_att_vectors = None

    for step in range(max_output_length):
 #       print("STEP", step)
        decoder_input = alive_seq[:, -1].view(-1, 1)
       # print("decoder input", decoder_input.shape)
        #if prev_att_vectors is not None:
        #    print("prev_att_vector", prev_att_vectors.shape)
        #if corrections is not None:
         #   print("corrections", corrections.shape)

        # expand current hypotheses
        # decode one single step
        # out: logits for final softmax
        out, hidden, att_scores, prev_att_vectors = decoder(
            encoder_output=encoder_output,
            encoder_hidden=encoder_hidden,
            src_mask=src_mask,
            trg_embed=embed(decoder_input),
            hidden=hidden,
            prev_att_vector=prev_att_vectors,
            unrol_steps=1,
            corrections=corrections)

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
  #      print(select_indices)  # batch times k selected indices

        # append latest prediction
  #      print("alive srq before select", alive_seq.shape)
  #      print("alive seq after select", alive_seq.index_select(0, select_indices).shape)
        alive_seq = torch.cat(
            [alive_seq.index_select(0, select_indices),
             topk_ids.view(-1, 1)], -1)  # batch_size*k x hyp_len

        if return_attention:
   #         print("att scores", att_scores.shape)  # batch_size*k x src_lengths
            current_attn = att_scores.index_select(0, select_indices)
   #         print("after select", current_attn.shape)
            if alive_attn is None:
                alive_attn = current_attn
#                print("setting alive attn", alive_attn.shape)

            else:
 #               print("before select", alive_attn.shape)
  #              print("sel indices", select_indices.shape)
                alive_attn = alive_attn.index_select(0, select_indices)
   #             print("after select", alive_attn.shape)
    #            print("alive vs curr", alive_attn.shape, current_attn.shape)
                alive_attn = torch.cat([alive_attn, current_attn], 1)
     #           print("after cat", alive_attn.shape)

        if return_attention_vectors:
      #      print("att vectors", prev_att_vectors.shape)  # batch_size*k x decoder.hidden_size
            current_attn_v = prev_att_vectors.index_select(0, select_indices)
#            print("after select att v", current_attn_v.shape)
            if alive_attn_v is None:
 #               print("setgtin alive", current_attn_v.shape)
                alive_attn_v = current_attn_v
            else:
  #              print("alive vs curr v", alive_attn_v.shape, current_attn_v.shape)

                alive_attn_v = alive_attn_v.index_select(0, select_indices)
                alive_attn_v = torch.cat([alive_attn_v, current_attn_v], 1)

        is_finished = topk_ids.eq(eos_index)
   #     print("finished", is_finished)
        if step + 1 == max_output_length:
            is_finished.fill_(1)
        # end condition is whether the top beam is finished
        end_condition = is_finished[:, 0].eq(1)

        # save finished hypotheses
        if is_finished.any():
            predictions = alive_seq.view(-1, size, alive_seq.size(-1))  # batch x beam x trg_len
    #        print("pred", predictions.shape)
     #       print("is finished alive att", alive_attn.shape)  # batch x trg_len x src_len
     #       print(alive_attn.view(batch_size, size, alive_attn.size(-2), alive_attn.size(-1)).shape)
            att_probs = alive_attn.view(-1, size, alive_attn.size(-2), alive_attn.size(-1))
                    #   if alive_attn is not None else None)
                    #alive_attn.view(
                    #    alive_attn.size(0), -1, size, alive_attn.size(-1))
                    #if alive_attn is not None else None)
            att_vectors = alive_attn_v.view(-1, size, alive_attn_v.size(-2), alive_attn_v.size(-1))
                   # if alive_attn_v is not None else None)
                    #alive_attn_v.view(
                    #    alive_attn_v.size(0), -1, size, alive_attn_v.size(-1))
                    #if alive_attn_v is not None else None)
            for i in range(is_finished.size(0)):
      #          print("i", i)
                b = batch_offset[i]
                if end_condition[i]:
                    is_finished[i].fill_(1)
                finished_hyp = is_finished[i].nonzero().view(-1)
                # store finished hypotheses for this batch
                for j in finished_hyp:
       #             print("att probs", att_probs.shape, "i", i, "j", j)
        #            print("att vs", att_vectors.shape, "i", i, "j", j)
         #           print("pred", predictions.shape)
                    hypotheses[b].append((
                        topk_scores[i, j],
                        predictions[i, j, 1:],  # ignore start_token
                        att_probs[i, j, 1:],#src_lengths[i]],
                        att_vectors[i, j, 1:])#src_lengths[i]])
                        #att_probs[:, i, j, :src_lengths[i]],
                        #att_vectors[:, i, j, :src_lengths[i]])
                    )
                # if the batch reached the end, save the n_best hypotheses
                if end_condition[i]:
                    best_hyp = sorted(
                        hypotheses[b], key=lambda x: x[0], reverse=True)
                    for n, (score, pred, att_p, att_v) in enumerate(best_hyp):
                        if n >= n_best:
                            break
                        results["scores"][b].append(score)
                        results["predictions"][b].append(pred)
                        results["att_vectors"][b].append(att_v)
                        results["att_probs"][b].append(att_p)
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

            if alive_attn is not None:
                # might get smaller now, not whole batch is expanded
                alive_attn = att_probs.index_select(0, non_finished) \
                    .view(alive_seq.size(0), -1, alive_attn.size(-1))
            if alive_attn_v is not None:
                alive_attn_v = att_vectors.index_select(0, non_finished) \
                    .view(alive_seq.size(0), -1, alive_attn_v.size(-1))

            # we don't need parts of the prev_att_vectors if some where finished
            if prev_att_vectors is not None:
                # first merge batch and beam dimension, then select, then unmerge again
                prev_att_vectors = prev_att_vectors.view(att_probs.size(0), att_probs.size(1), prev_att_vectors.size(-1)).index_select(0, non_finished).view(-1, 1, prev_att_vectors.size(-1))
            # same for the corrections
            if corrections is not None:
                corrections = corrections.view(att_probs.size(0), att_probs.size(1), corrections.size(-2), corrections.size(-1)).index_select(0, non_finished).view(-1, corrections.size(-2), corrections.size(-1))

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

            #print("att vec", att_vectors.shape)
            #att_vectors = att_vectors.index_select(0, select_indices)

    def pad_and_stack_hyps(hyps, pad_value):
        filled = np.ones((len(hyps), max([h.shape[0] for h in hyps])),
                         dtype=int) * pad_value
      #  print("filled", filled.shape)  # batch x max(trg_length)
    #    print(len(hyps), hyps[0].shape)
        for j, h in enumerate(hyps):
            for k, i in enumerate(h):
                filled[j, k] = i
        return filled

    # from results to stacked outputs
    assert n_best == 1
    # only works for n_best=1 for now
    #print(results["predictions"])
    final_outputs = pad_and_stack_hyps([r[0].cpu().numpy() for r in
                                        results["predictions"]],
                                       pad_value=pad_index)
    if return_attention:

        def pad_and_stack_att_probs(aps, pad_value):
            # resulting tensor should have shape batch x max trg length x max src length
            filled = np.ones((len(aps), max([h.shape[0] for h in aps]), max([h.shape[1] for h in aps])),
                             dtype=float) * pad_value
            for j, h in enumerate(aps):
                for k, i in enumerate(h):
                    for l, m in enumerate(i):
                       filled[j, k, l] = m
            return filled

       # print(len(results))  # 5
       # print(len(results["att_probs"])) # 10 batch
       # print(len(results["att_probs"][0])) # 1 should be trg_length
       # print(results["att_probs"][0][0].shape) # 10 src_leng
       # print(([r[0].cpu().numpy().shape for r in
       #                             results["att_probs"]]))
        #final_att_probs = np.stack([r[0].cpu().numpy() for r in
        #                            results["att_probs"]], axis=1)
        final_att_probs = pad_and_stack_att_probs([r[0].cpu().numpy() for r in results["att_probs"]], pad_value=0)
        #print("stat att prob", final_att_probs.shape)
    else:
        final_att_probs = None
    if return_attention_vectors:
        def pad_and_stack_att_vecs(ap_vs, pad_value):
         #   print("ap vs", ap_vs[0].shape)
            # resulting tensor should have shape batch x max trg length x max src length
            filled = np.ones((len(ap_vs), max([h.shape[0] for h in ap_vs]), decoder.hidden_size),
                             dtype=float) * pad_value
          #  print("Fi", filled.shape)
            for j, h in enumerate(ap_vs):
                for k, i in enumerate(h):
                    for l, m in enumerate(i):
                       filled[j, k, l] = m
            return filled
        final_att_vectors = pad_and_stack_att_vecs([r[0] for r in results["att_vectors"]], pad_value=0)
       # print("statt att vec", final_att_vectors.shape)
        #final_att_vectors = torch.stack([r[0] for r in
        #                             results["att_vectors"]])
    else:
        final_att_vectors = None
    return final_outputs, final_att_probs, final_att_vectors