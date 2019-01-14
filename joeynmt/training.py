# coding: utf-8
import argparse
import logging
import time
import os
import numpy as np
import shutil
import itertools


import torch
import torch.nn as nn

from joeynmt.model import build_model

from joeynmt.batch import Batch
from joeynmt.helpers import log_data_info, load_data, \
    load_config, log_cfg, store_attention_plots, make_data_iter, \
    load_model_from_checkpoint, store_correction_plots
from joeynmt.prediction import validate_on_data
from joeynmt.metrics import token_accuracy, bleu, f1_bin


class TrainManager:
    """ Manages training loop, validations, learning rate scheduling
    and early stopping."""

    def __init__(self, model, config):
        """
        Creates a new TrainManager for a model, specified as in configuration.

        :param model:
        :param config:
        """
        train_config = config["training"]
        self.model = model
        self.overwrite = train_config.get("overwrite", False)
        self.model_dir = self._make_model_dir(train_config["model_dir"])
        self.valid_report_file = "{}/validations.txt".format(self.model_dir)
        self.logger = self._make_logger()
        self.pad_index = self.model.pad_index
        self.bos_index = self.model.bos_index
        criterion = nn.NLLLoss(ignore_index=self.pad_index, reduction='sum')
        self.learning_rate_min = train_config.get("learning_rate_min", 1.0e-8)
        if train_config["loss"].lower() not in ["crossentropy", "xent",
                                                "mle", "cross-entropy"]:
            raise NotImplementedError("Loss is not implemented. Only xent.")
        learning_rate = train_config.get("learning_rate", 3.0e-4)
        weight_decay = train_config.get("weight_decay", 0)
        if model.corrector is not None:
            # 2 sets of parameters -> 2 optimizers
            # see https://stackoverflow.com/questions/51578235/pytorch-how-to-get-the-gradient-of-loss-function-twice
            all_params = model.named_parameters()
            corrector_params = {k:v for (k,v) in all_params if "corrector" in k}
            self.corrector_params = corrector_params
            self.logger.debug(
                "CORRECTOR PARAMS: {}".format(corrector_params.keys()))
            self.logger.debug("CORRECTOR size: {}".format(
                sum([np.prod(p.size()) for p in self.corrector_params.values()])
            ))
            mt_params = {k:v for (k,v) in model.named_parameters()
                         if k not in corrector_params.keys()}
            self.logger.debug("MT PARAMS: {}".format(mt_params.keys()))
            self.logger.debug("MT size: {}".format(
                sum([np.prod(p.size()) for p in mt_params.values()])))

            for k, v in model.named_parameters():
                if v.requires_grad:
                    self.logger.debug("Updating {}".format(k))
                else:
                    self.logger.debug("NOT Updating {}".format(k))

            self.optimizer = {}
            if train_config["optimizer"].lower() == "adam":
                self.optimizer["mt"] = torch.optim.Adam(
                    mt_params.values(), weight_decay=weight_decay,
                    lr=learning_rate["mt"])
                self.optimizer["corrector"] = torch.optim.Adam(
                    corrector_params.values(), weight_decay=weight_decay,
                    lr=learning_rate["corrector"])
            else:
                # default
                self.optimizer["mt"] = torch.optim.SGD(
                    mt_params.values(), weight_decay=weight_decay,
                    lr=learning_rate["mt"])
                self.optimizer["corrector"] = torch.optim.SGD(
                    corrector_params.values(), weight_decay=weight_decay,
                    lr=learning_rate["corrector"])
        else:
            if train_config["optimizer"].lower() == "adam":
                self.optimizer = torch.optim.Adam(
                    model.parameters(), weight_decay=weight_decay, lr=learning_rate)
            else:
                # default
                self.optimizer = torch.optim.SGD(
                    model.parameters(), weight_decay=weight_decay, lr=learning_rate)
        self.schedule_metric = train_config.get("schedule_metric",
                                                "eval_metric")
        self.ckpt_metric = train_config.get("ckpt_metric", "eval_metric")
        self.best_ckpt_iteration = 0
        # if we schedule after BLEU/chrf, we want to maximize it, else minimize
        scheduler_mode = "max" if self.schedule_metric == "eval_metric" \
            else "min"
        # the ckpt metric decides on how to find a good early stopping point:
        # ckpts are written when there's a new high/low score for this metric
        if self.ckpt_metric == "eval_metric":
            self.best_ckpt_score = -np.inf
            self.is_best = lambda x: x > self.best_ckpt_score
        else:
            self.best_ckpt_score = np.inf
            self.is_best = lambda x: x < self.best_ckpt_score
        self.scheduler = None
        if "scheduling" in train_config.keys() and \
                train_config["scheduling"]:
            if train_config["scheduling"].lower() == "plateau":
                if model.corrector is not None:
                    # 2 schedulers
                    self.scheduler = {k: torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer=v,
                        mode=scheduler_mode,
                        verbose=False,
                        threshold_mode='abs',
                        factor=train_config.get("decrease_factor", 0.1),
                        patience=train_config.get("patience", 10))
                                      for k, v in self.optimizer.items()}
                else:
                    # learning rate scheduler
                    self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer=self.optimizer,
                        mode=scheduler_mode,
                        verbose=False,
                        threshold_mode='abs',
                        factor=train_config.get("decrease_factor", 0.1),
                        patience=train_config.get("patience", 10))
            elif train_config["scheduling"].lower() == "decaying":
                if model.corrector is not None:
                    # 2 schedulers
                    self.scheduler = {k: torch.optim.lr_scheduler.StepLR(
                        optimizer=v,
                        step_size=train_config.get("decaying_step_size", 10))
                                      for k, v in self.optimizer.items()}
                else:
                    self.scheduler = torch.optim.lr_scheduler.StepLR(
                        optimizer=self.optimizer,
                        step_size=train_config.get("decaying_step_size", 10))
            elif train_config["scheduling"].lower() == "exponential":
                if model.corrector is not None:
                    # 2 schedulers
                    self.scheduler = {k: torch.optim.lr_scheduler.ExponentialLR(
                        optimizer=v,
                        gamma=train_config.get("decrease_factor", 0.99)
                    ) for k, v in self.optimizer}
                else:
                    self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                        optimizer=self.optimizer,
                        gamma=train_config.get("decrease_factor", 0.99)
                    )
        self.shuffle = train_config.get("shuffle", True)
        self.epochs = train_config["epochs"]
        self.batch_size = train_config["batch_size"]
        self.criterion = criterion
        self.normalization = train_config.get("normalization", "batch")
        self.normalize_corrector = train_config.get("normalize_corrector", False)
        self.steps = 0
        # stop training if this flag is True by reaching learning rate minimum
        self.stop = False
        self.total_tokens = 0
        self.max_output_length = train_config.get("max_output_length", None)
        self.use_cuda = train_config["use_cuda"]
        if self.use_cuda:
            self.model.cuda()
        self.logging_freq = train_config.get("logging_freq", 100)
        self.validation_freq = train_config.get("validation_freq", 1000)
        self.eval_metric = train_config.get("eval_metric", "bleu")
        self.print_valid_sents = train_config["print_valid_sents"]
        self.level = config["data"]["level"]
        self.clip_grad_fun = None
        if "clip_grad_val" in train_config.keys():
            clip_value = train_config["clip_grad_val"]
            self.clip_grad_fun = lambda params:\
                nn.utils.clip_grad_value_(parameters=params,
                                          clip_value=clip_value)
        elif "clip_grad_norm" in train_config.keys():
            max_norm = train_config["clip_grad_norm"]
            self.clip_grad_fun = lambda params:\
                nn.utils.clip_grad_norm_(parameters=params, max_norm=max_norm)

        assert not ("clip_grad_val" in train_config.keys() and
                    "clip_grad_norm" in train_config.keys()), \
            "you can only specify either clip_grad_val or clip_grad_norm"

        if "load_model" in train_config.keys():
            model_load_path = train_config["load_model"]
            self.logger.info("Loading model from {}".format(model_load_path))
            self.load_checkpoint(model_load_path)

        self.trainable_params = [n for (n, p) in self.model.named_parameters()
                            if p.requires_grad]
        self.logger.info("Trainable parameters: {}".format(
            self.trainable_params))

    def save_checkpoint(self):
        """
        Save the model's current parameters and state to a checkpoint.

        :return:
        """
        model_path = "{}/{}.ckpt".format(self.model_dir, self.steps)
        state = {
            "steps": self.steps,
            "total_tokens": self.total_tokens,
            "best_ckpt_score": self.best_ckpt_score,
            "best_ckpt_iteration": self.best_ckpt_iteration,
            "model_state": self.model.state_dict(),
        }
        if type(self.optimizer) is not dict:
            state["optimizer_state"] = self.optimizer.state_dict()
            state["scheduler_state"] = self.scheduler.state_dict() if \
                self.scheduler is not None else None
        else:
            for k,v in self.optimizer.items():
                state["{}_optimizer_state".format(k)] = v.state_dict()
            for k,v in self.scheduler.items():
                state["{}_scheduler_state".format(k)] = v.state_dict()
                #if self.scheduler is not None else None,
        torch.save(state, model_path)

    def load_checkpoint(self, path):
        """
        Load a model from a given checkpoint file.

        :param path:
        :return:
        """
        model_checkpoint = load_model_from_checkpoint(
            path=path, use_cuda=self.use_cuda)

        # restore model and optimizer parameters
        self.model.load_state_dict(model_checkpoint["model_state"])
        self.optimizer.load_state_dict(model_checkpoint["optimizer_state"])

        if model_checkpoint["scheduler_state"] is not None:
            self.scheduler.load_state_dict(model_checkpoint["scheduler_state"])

        # restore counts
        self.steps = model_checkpoint["steps"]
        self.total_tokens = model_checkpoint["total_tokens"]
        self.best_ckpt_score = model_checkpoint["best_ckpt_score"]
        self.best_ckpt_iteration = model_checkpoint["best_ckpt_iteration"]

        # move parameters to cuda
        if self.use_cuda:
            self.model.cuda()

    def _make_model_dir(self, model_dir):
        """
        Create a new directory for the model.

        :param model_dir:
        :return:
        """
        if os.path.isdir(model_dir):
            if not self.overwrite:
                raise FileExistsError(
                    "Model directory exists and overwriting is disabled.")
        else:
            os.makedirs(model_dir)
        return model_dir

    def _make_logger(self):
        """
        Create a logger for logging the training process.
        :return:
        """
        logger = logging.getLogger(__name__)
        logger.setLevel(level=logging.DEBUG)
        fh = logging.FileHandler(
            "{}/train.log".format(self.model_dir))
        fh.setLevel(level=logging.DEBUG)
        logger.addHandler(fh)
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(message)s')
        fh.setFormatter(formatter)
        sh.setFormatter(formatter)
        logging.getLogger("").addHandler(sh)
        logger.info("Hello! This is Joey-NMT.")
        return logger

    def train_and_validate(self, train_data, valid_data):
        """
        Train the model and validate it from time to time on the validation set.

        :param train_data:
        :param valid_data:
        :return:
        """
        train_iter = make_data_iter(train_data, batch_size=self.batch_size,
                                    train=True, shuffle=self.shuffle)
        epoch_no = 0
        for epoch_no in range(self.epochs):
            self.logger.info("EPOCH {}".format(epoch_no + 1))
            self.model.train()

            start = time.time()
            total_valid_duration = 0
            processed_tokens = self.total_tokens

            for batch_no, batch in enumerate(iter(train_iter), 1):
                # reactivate training
                self.model.train()
                batch = Batch(batch, self.pad_index, use_cuda=self.use_cuda)
                batch_loss = self._train_batch(batch)

                # log learning progress
                if self.model.training and self.steps % self.logging_freq == 0:
                    elapsed = time.time() - start - total_valid_duration
                    elapsed_tokens = self.total_tokens - processed_tokens
                    self.logger.info(
                        "Epoch %d Step: %d Loss: %f Tokens per Sec: %f" %
                        (epoch_no + 1, self.steps, batch_loss,
                         elapsed_tokens / elapsed))
                    start = time.time()
                    total_valid_duration = 0

                # validate on whole dev set (greedy decoding)
                if self.steps % self.validation_freq == 0:
                    valid_start_time = time.time()

                    valid_score, valid_sent_score,\
                        corr_valid_score, corr_valid_sent_score, \
                        valid_loss, valid_ppl, valid_sources, \
                        valid_sources_raw, valid_references, \
                        valid_hypotheses, corr_valid_hypotheses, \
                        valid_hypotheses_raw, corr_valid_hypotheses_raw,\
                        valid_attention_scores, corr_valid_attention_scores, \
                        corrections, rewards, reward_targets = validate_on_data(
                            batch_size=self.batch_size, data=valid_data,
                            eval_metric=self.eval_metric,
                            level=self.level, model=self.model,
                            use_cuda=self.use_cuda,
                            max_output_length=self.max_output_length,
                            criterion=self.criterion)

                    # TODO decide whether to write checkpoint: use corr?
                    if self.ckpt_metric == "loss":
                        ckpt_score = valid_loss
                    elif self.ckpt_metric in ["ppl", "perplexity"]:
                        ckpt_score = valid_ppl
                    else:
                        ckpt_score = valid_score

                    new_best = False
                    if self.is_best(ckpt_score):
                        self.best_ckpt_score = ckpt_score
                        self.best_ckpt_iteration = self.steps
                        self.logger.info(
                            'Hooray! New best validation result [{}]!'.format(
                                self.ckpt_metric))
                        new_best = True
                        self.save_checkpoint()

                    # pass validation score or loss or ppl to scheduler
                    if self.schedule_metric == "loss":
                        # schedule based on loss
                        schedule_score = valid_loss
                    elif self.schedule_metric in ["ppl", "perplexity"]:
                        # schedule based on perplexity
                        schedule_score = valid_ppl
                    else:
                        # schedule based on evaluation score
                        schedule_score = valid_score
                    if self.scheduler is not None:
                        if type(self.scheduler) is dict:
                            # corrector is scheduled after eval metric
                            self.scheduler["corrector"].step(corr_valid_score)
                            # make scheduler step for MT model
                            self.scheduler["mt"].step(schedule_score)
                        else:
                            self.scheduler.step(schedule_score)

                    # always print first x sentences
                    for p in range(self.print_valid_sents):
                        self.logger.debug("Example #{}".format(p))
                        self.logger.debug("\tRaw source: {}".format(
                            valid_sources_raw[p]))
                        self.logger.debug("\tSource: {}".format(
                            valid_sources[p]))
                        self.logger.debug("\tReference: {}".format(
                            valid_references[p]))
                        self.logger.debug("\tRaw hypothesis: {}".format(
                            valid_hypotheses_raw[p]))
                        self.logger.debug("\tHypothesis: {}".format(
                            valid_hypotheses[p]))
                        self.logger.debug("\tRaw hypothesis (CORR): {}".format(
                            corr_valid_hypotheses_raw[p]))
                        self.logger.debug("\tHypothesis (CORR): {}".format(
                            corr_valid_hypotheses[p]))
                        self.logger.debug("\tRewards: {}".format(
                            [(t, r[0]) for t, r in
                             zip(valid_hypotheses_raw[p], rewards[p])]))
                        self.logger.debug("\tGold rewards: {}".format(
                            [(t, r[0]) for t, r in
                             zip(valid_hypotheses_raw[p], reward_targets[p])]))
                    valid_duration = time.time() - valid_start_time
                    total_valid_duration += valid_duration

                    # MSE of reward model
                    assert len(reward_targets) == len(rewards)

                    # collect all rewards in one big array
                    # (all valid instances, all time steps)
                    rewards_flat = np.array(list(
                        itertools.chain.from_iterable(rewards))).flatten()
                    reward_targets_flat = np.array(list(
                        itertools.chain.from_iterable(reward_targets))).flatten()
                    bin_reward_targets_flat = np.equal(reward_targets_flat, 1).astype(int)
                    assert rewards_flat.shape == reward_targets_flat.shape

                    reward_mse = np.mean((reward_targets_flat-rewards_flat)**2)

                    reward_corr = np.corrcoef(reward_targets_flat.astype(float),
                                              rewards_flat)[0, 1]
                    # transform rewards into binary labels with 0.5 as threshold
                    # then compute accuracy
                    bin_rewards_flat = np.greater_equal(
                        rewards_flat, 0.5).astype(int)
                    bin_reward_acc = np.equal(bin_rewards_flat,
                                              bin_reward_targets_flat)\
                                         .sum()/reward_targets_flat.size

                    # compute f1 for both classes
                    f1_1, f1_0 = f1_bin(bin_rewards_flat,
                                        bin_reward_targets_flat)
                    f1_prod = f1_1*f1_0

                    self.logger.info(
                        'Validation result at epoch {}, step {}: {}: {:.5f}'
                        ' (sent: {:.5f}), corr {}: {:.5f} (sent: {:.5f}),'
                        ' reward MSE: {:.5f}, correl.: {:.2f}, acc.: {:.2f},'
                        ' f1(1): {:.2f}, f1(0): {:.2f}, f1_prod: {:.2f}'
                        ' loss: {:.5f}, ppl: {:.5f}, duration: {:.4f}s'.format(
                            epoch_no+1, self.steps, self.eval_metric,
                            valid_score, valid_sent_score, self.eval_metric,
                            corr_valid_score,
                            corr_valid_sent_score,
                            reward_mse, reward_corr, bin_reward_acc*100,
                            f1_1*100, f1_0*100, f1_prod*100,
                            valid_loss, valid_ppl, valid_duration))

                    # TODO check if this moment computation is correct
                    # since corrections contains lists
                    # it is not flattened
                    corrections_means = corrections.mean()
                    corrections_std = np.sqrt(
                        np.mean((corrections - corrections_means) ** 2))
                    self.logger.info("Correction moments: "
                                     "mean={:.5f}, std={:.5f}.".format(
                        corrections_means, corrections_std))
                    rewards_means = rewards_flat.mean()
                    rewards_std = np.std(rewards_flat)
                    self.logger.info("Reward moments: "
                                     "mean={:.5f}, std={:.5f}.".format(
                        rewards_means, rewards_std))

                    target_rewards_means = reward_targets_flat.mean()
                    target_rewards_std = np.std(reward_targets_flat)
                    self.logger.info("Target reward moments: "
                                     "mean={:.5f}, std={:.5f}.".format(
                        target_rewards_means, target_rewards_std))

                    # find examples where corr improved (token acc or sbleu)
                    max_examples = self.print_valid_sents
                    print_pos_examples = 0
                    print_neg_examples = 0
                    for hyp, corr, ref in zip(valid_hypotheses,
                                              corr_valid_hypotheses,
                                              valid_references):
                        if print_pos_examples >= max_examples \
                                and print_neg_examples >= max_examples:
                            break

                        corr_acc = token_accuracy([corr], [ref],
                                                  level=self.level)
                        corr_sbleu = bleu([corr], [ref])
                        hyp_acc = token_accuracy([hyp], [ref],
                                                 level=self.level)
                        hyp_sbleu = bleu([hyp], [ref])
                        if corr_sbleu > hyp_sbleu:
                            if print_pos_examples >= max_examples:
                                continue
                            print_pos_examples += 1
                            effect = "improved"
                        elif corr_sbleu < hyp_sbleu:
                            if print_neg_examples >= max_examples:
                                continue
                            print_neg_examples += 1
                            effect = "worsened"
                        else:
                            continue

                        self.logger.debug("Corrector {}: "
                                         "\n\tHYP {} ({:.2f})"
                                         "\n\tCORR {} ({:.2f})"
                                         "\n\tREF {}".format(
                            effect, hyp, hyp_sbleu, corr, corr_sbleu, ref))

                    # append to validation report
                    self._add_report(
                        valid_score=valid_score,
                        valid_sent_score=valid_sent_score,
                        valid_loss=valid_loss,
                        valid_ppl=valid_ppl, eval_metric=self.eval_metric,
                        new_best=new_best,
                        corr_valid_score=corr_valid_score,
                        corr_valid_sent_score=corr_valid_sent_score,
                        reward_f1_1=f1_1*100, reward_f1_0=f1_0*100,
                        reward_f1_prod=f1_prod*100, reward_corr=reward_corr,
                        reward_mse=reward_mse, reward_acc=bin_reward_acc*100)

                    # TODO early stopping with corrector

                    # store validation set outputs
                    current_valid_output_file = "{}/{}.hyps".format(
                        self.model_dir,
                        self.steps)
                    corr_current_valid_output_file = "{}/{}.hyps.corr".format(
                        self.model_dir,
                        self.steps)
                    self.store_outputs(
                        valid_hypotheses, current_valid_output_file)
                    self.store_outputs(
                        corr_valid_hypotheses, corr_current_valid_output_file)

                    # store attention plots for first three sentences of
                    # valid data and one randomly chosen example
                    random_example = np.random.randint(
                        0, len(valid_hypotheses))
                    store_attention_plots(attentions=valid_attention_scores,
                                          targets=valid_hypotheses_raw,
                                          sources=[s for s in valid_data.src],
                                          idx=[0, 1, 2, random_example],
                                          output_prefix="{}/att.{}".format(
                                              self.model_dir,
                                              self.steps))
                    # store attention after correction
                    store_attention_plots(attentions=corr_valid_attention_scores,
                                          targets=corr_valid_hypotheses_raw,
                                          sources=[s for s in valid_data.src],
                                          idx=[0, 1, 2, random_example],
                                          output_prefix="{}/corr.att.{}".format(
                                              self.model_dir,
                                              self.steps))

                    store_correction_plots(
                        corrections=corrections, rewards=rewards,
                        targets=valid_hypotheses_raw,
                        corr_targets=corr_valid_hypotheses_raw,
                        idx=[0, 1, 2, random_example],
                        output_prefix="{}/corr.{}".format(
                            self.model_dir, self.steps))

                if self.stop:
                    break
            if self.stop:
                self.logger.info(
                    'Training ended since minimum lr {} was reached.'.format(
                        self.learning_rate_min))
                break
        else:
            self.logger.info('Training ended after {} epochs.'.format(epoch_no+1))
        self.logger.info('Best validation result at step {}: {} {}.'.format(
            self.best_ckpt_iteration, self.best_ckpt_score, self.ckpt_metric))

    def _train_batch(self, batch):
        """
        Train the model on one batch: Compute the loss, make a gradient step.

        :param batch:
        :return:
        """
        # normalize batch loss
        if self.normalization == "batch":
            normalizer = batch.nseqs
        elif self.normalization == "tokens":
            normalizer = batch.ntokens
        else:
            raise NotImplementedError("Only normalize by 'batch' or 'tokens'")

        # standard xent loss
        batch_loss = self.model.get_xent_loss_for_batch(
            batch=batch, criterion=self.criterion)

        norm_batch_loss = batch_loss.sum() / normalizer

        # only if decoder and embeddings not frozen
        if not all(["corrector" in p for p in self.trainable_params]):
            # compute gradient
            norm_batch_loss.backward()
        # grads for corrector params are zero here
        #print("grad norms after Xent", [(k, torch.norm(v.grad, 2)) for k, v in self.model.named_parameters() if v.grad is not None])

        # compute corrector loss separately
        # with torch.autograd.grad(outputs, inputs, grad_outputs=None, retain_graph=None, create_graph=False, only_inputs=True, allow_unused=False)
        corrector_loss = self.model.get_corr_loss_for_batch(
            batch=batch, criterion=self.criterion,
            logging_fun=
            self.logger.debug if not self.steps % self.logging_freq else None)

        # normalize per batch/token
        norm_corrector_loss = corrector_loss / normalizer

        if self.normalize_corrector:
            # use xent loss of decoder as factor to scale corrector loss
            # TODO could also use subtraction, but no evidence that better yet
            norm_corrector_loss /= batch_loss.detach()

        # TODO add RL loss! gain in BLEU
        # TODO maybe penalize even more

        # only train corrector if trainable params exist
        if any(["corrector" in p for p in self.trainable_params]):
            grads = {}
            for name, param in self.corrector_params.items():
                # compute the gradient of the loss wrt to each of the params
                grad = torch.autograd.grad(corrector_loss,
                                           inputs=param, retain_graph=True)[0]
                grads[name] = grad
                assert grad.shape == param.shape
                param.grad = grad
            #print(torch.autograd.grad(corrector_loss, inputs=inputs))
            #print("grad norms for corr: ", [(k, torch.norm(v, 2)) for k, v in grads.items()])

        if not self.steps % self.logging_freq:
            self.logger.debug("Gradient norms (w/o clipping): {}".format(
                              [(k, torch.norm(v.grad, 2)) for k, v
                               in self.model.named_parameters()
                               if v.grad is not None]).replace("),", "\n\t"))

        if self.clip_grad_fun is not None:
            # clip gradients (in-place), corrector params included
            self.clip_grad_fun(params=self.model.parameters())

            if not self.steps % self.logging_freq:
                self.logger.debug("Gradient norms (after clipping): {}".format(
                                  [(k, torch.norm(v.grad, 2)) for k, v
                                   in self.model.named_parameters()
                                   if v.grad is not None]).replace("),", "\n\t"))

        # make gradient step
        if type(self.optimizer) is dict:
            # two optimizers with 2 different learning rates
            self.optimizer["mt"].step()
            self.optimizer["corrector"].step()
            self.optimizer["mt"].zero_grad()
            self.optimizer["corrector"].zero_grad()
        else:
            self.optimizer.step()
            self.optimizer.zero_grad()

        if not self.steps % self.logging_freq:
            self.logger.debug("Corrector loss: {}".format(
                              (corrector_loss/normalizer).detach().cpu().numpy()))
            self.logger.debug("Xent loss: {}".format(
                              norm_batch_loss.detach().cpu().numpy()))
            self.logger.debug("(corr/xent) loss: {}".format(
                              (norm_corrector_loss).cpu().detach().numpy()/
                              norm_batch_loss.cpu().detach().numpy()))

        # increment step and token counter
        self.steps += 1
        self.total_tokens += batch.ntokens
        return norm_batch_loss

    def _add_report(self, valid_score, valid_sent_score,
                    valid_ppl, valid_loss, eval_metric,
                    new_best=False, corr_valid_score=None,
                    corr_valid_sent_score=None,
                    reward_f1_1=None, reward_f1_0=None,
                    reward_f1_prod=None, reward_corr=None,
                    reward_mse=None, reward_acc=None):
        """
        Add a one-line report to validation logging file.

        :param valid_score:
        :param valid_ppl:
        :param valid_loss:
        :param eval_metric:
        :param new_best:
        :param corr_valid_score:
        :return:
        """
        current_lr = -1
        # ignores other param groups for now
        if type(self.optimizer) is dict:
            current_lr = {k: v.param_groups[0]["lr"] for
                          k,v in self.optimizer.items()}

        else:
            for param_group in self.optimizer.param_groups:
                current_lr = param_group['lr']

        if type(current_lr) is dict:
            # TODO adapt to corr
            # only stop if all learning rates have reached minimum
            self.stop = all(
                [v < self.learning_rate_min for v in current_lr.values()])
        else:
            if current_lr < self.learning_rate_min:
                self.stop = True

        report_str = "Steps: {}\tLoss: {:.5f}\tPPL: {:.5f}\tMT-{}: {:.5f}\t" \
                     "MT-sBLEU: {:.5f})".format(
                            self.steps, valid_loss, valid_ppl, eval_metric,
                            valid_score, valid_sent_score)

        if corr_valid_score is not None and corr_valid_sent_score is not None:
            report_str += "\tCorr-{}: {:.5f}\tCorr-sBLEU: {:.5f}".format(
                eval_metric, corr_valid_score, corr_valid_sent_score)

        if reward_mse is not None:
            report_str += "\tReward_MSE: {:.5f}".format(reward_mse)

        if reward_corr is not None:
            report_str += "\tReward_Corr: {:.2f}".format(reward_corr)

        if reward_acc is not None:
            report_str += "\tReward_Acc: {:.2f}".format(reward_acc)

        if reward_f1_prod is not None \
            and reward_f1_0 is not None \
                and reward_f1_1 is not None:
            report_str += "\tF1_1: {:.2f}" \
                          "\tF1_0: {:.2f} " \
                          "\tF1_prod: {:.2f}".format(
                                reward_f1_1, reward_f1_0, reward_f1_prod)

        # at the end add * and lr
        report_str += "\t LR: {}\t{}\n".format(current_lr,
                                               "*" if new_best else "")

        with open(self.valid_report_file, 'a') as opened_file:
            opened_file.write(report_str)

    def store_outputs(self, hypotheses, output_file):
        """
        Write current validation outputs to file in model_dir.
        :param hypotheses:
        :return:
        """
        with open(output_file, 'w') as opened_file:
            for hyp in hypotheses:
                opened_file.write("{}\n".format(hyp))


def train(cfg_file):
    """
    Main training function. After training, also test on test data if given.

    :param cfg_file:
    :return:
    """
    cfg = load_config(cfg_file)
    # set the random seed
    # torch.backends.cudnn.deterministic = True
    seed = cfg["training"].get("random_seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # load the data
    train_data, dev_data, test_data, src_vocab, trg_vocab = \
        load_data(cfg=cfg)

    # build an encoder-decoder model
    model = build_model(cfg["model"], src_vocab=src_vocab, trg_vocab=trg_vocab)

    # for training management, e.g. early stopping and model selection
    trainer = TrainManager(model=model, config=cfg)

    # store copy of original training config in model dir
    shutil.copy2(cfg_file, trainer.model_dir+"/config.yaml")

    # print config
    log_cfg(cfg, trainer.logger)

    log_data_info(train_data=train_data, valid_data=dev_data,
                  test_data=test_data, src_vocab=src_vocab, trg_vocab=trg_vocab,
                  logging_function=trainer.logger.info)
    model.log_parameters_list(logging_function=trainer.logger.info)

    logging.info(model)

    # store the vocabs
    src_vocab_file = "{}/src_vocab.txt".format(cfg["training"]["model_dir"])
    src_vocab.to_file(src_vocab_file)
    trg_vocab_file = "{}/trg_vocab.txt".format(cfg["training"]["model_dir"])
    trg_vocab.to_file(trg_vocab_file)

    # train the model
    trainer.train_and_validate(train_data=train_data, valid_data=dev_data)

    if test_data is not None:
        # test model
        if "testing" in cfg.keys():
            beam_size = cfg["testing"].get("beam_size", 0)
            beam_alpha = cfg["testing"].get("alpha", -1)
        else:
            beam_size = 0
            beam_alpha = -1

        score, sent_score, corr_score, sent_corr_score, \
        loss, ppl, sources, sources_raw, references, \
        hypotheses, corr_hypotheses, \
        hypotheses_raw, corr_hypotheses_raw, \
        attention_scores, corr_attention_scores, corrections, \
        rewards, reward_targets =\
            validate_on_data(
                data=test_data, batch_size=trainer.batch_size,
                eval_metric=trainer.eval_metric, level=trainer.level,
                max_output_length=trainer.max_output_length,
                model=model, use_cuda=trainer.use_cuda, criterion=None,
                beam_size=beam_size, beam_alpha=beam_alpha)
        
        if "trg" in test_data.fields:
            decoding_description = "Greedy decoding" if beam_size == 0 else \
                "Beam search decoding with beam size = {} and alpha = {}"\
                    .format(beam_size, beam_alpha)
            trainer.logger.info("{:4s}: {} (sent: {}) {} [{}]".format(
                "Test data result", score, sent_score, trainer.eval_metric,
                decoding_description))
            trainer.logger.info("{:4s}: {} (sent: {}) {} [{}]".format(
                "Test data result after correction", corr_score,
                sent_corr_score, trainer.eval_metric, decoding_description))
        else:
            trainer.logger.info(
                "No references given for {}.{} -> no evaluation.".format(
                    cfg["data"]["test"],cfg["data"]["src"]))

        output_path_set = "{}/{}.{}".format(
            trainer.model_dir, "test",cfg["data"]["trg"])
        corr_output_path_set = "{}/{}.{}.corr".format(
            trainer.model_dir, "test", cfg["data"]["trg"])
        with open(output_path_set, mode="w", encoding="utf-8") as f:
            for h in hypotheses:
                f.write(h + "\n")
        with open(corr_output_path_set, mode="w", encoding="utf-8") as f:
            for h in corr_hypotheses:
                f.write(h + "\n")
        trainer.logger.info("Test translations saved to: {}".format(
            output_path_set))
        trainer.logger.info("Test corrected translations saved to: {}".format(
            corr_output_path_set))

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Joey-NMT')
    parser.add_argument("config", default="configs/default.yaml", type=str,
                        help="Training configuration file (yaml).")
    args = parser.parse_args()
    train(cfg_file=args.config)
