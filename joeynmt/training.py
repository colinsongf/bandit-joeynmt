# coding: utf-8
import argparse
import logging
import time
import os
import numpy as np
import shutil
from collections import defaultdict

import torch
import torch.nn as nn

from joeynmt.model import build_model

from joeynmt.batch import Batch
from joeynmt.helpers import log_data_info, load_data, \
    load_config, log_cfg, store_attention_plots, make_data_iter, \
    load_model_from_checkpoint
from joeynmt.prediction import validate_on_data
from joeynmt.deliberation import DeliberationModel
from joeynmt.attention import BahdanauAttention


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
        self.pad_index = self.model.pad_index
        self.bos_index = self.model.bos_index
        self.learning_rate_min = train_config.get("learning_rate_min", 1.0e-8)
        if train_config["loss"].lower() not in ["crossentropy", "xent",
                                                "mle", "cross-entropy"]:
            raise NotImplementedError("Loss is not implemented. Only xent.")
        learning_rate = train_config.get("learning_rate", 3.0e-4)
        weight_decay = train_config.get("weight_decay", 0)
        if isinstance(self.model, DeliberationModel):
            # separate optimizers for two decoders, shared parameters are included in both
            # decoder1 parameters and encoder parameters
            param_dec1 = [p for name, p in model.named_parameters() if "decoder1" in name or "trg_embed" in name]
            # decoder2 parameters (trg embedding is shared) and encoder parameters
            param_dec2 = [p for name, p in model.named_parameters() if "decoder2" in name]
            param_enc = [p for name, p in model.named_parameters() if "encoder" in name or "src_embed" in name]
            param_dec1_names = [name for name, p in model.named_parameters() if
                          "decoder1" in name or "trg_embed" in name]
            param_dec2_names = [name for name, p in model.named_parameters() if
                          "decoder2" in name]
            param_enc_names = [name for name, p in model.named_parameters() if
                                "encoder" in name or "src_embed" in name]
            if type(learning_rate) == list:
                assert len(learning_rate) == 3
                learning_rate1, learning_rate2, learning_rate3 = learning_rate
            else:
                learning_rate1, learning_rate2, learning_rate3 = \
                    learning_rate, learning_rate, learning_rate
            param_dict_enc = {"params": param_enc, "lr": learning_rate1}
            param_dict_d1 = {"params": param_dec1, "lr": learning_rate2}
            param_dict_d2 = {"params": param_dec2, "lr": learning_rate3}
            if train_config["optimizer"].lower() == "adam":
                # create 1 optimizer with 2 param groups!
                # see https://pytorch.org/docs/stable/optim.html
                # weight decay and lr are default if for one group not specified
                self.optimizer = torch.optim.Adam(
                    params=[param_dict_enc, param_dict_d1, param_dict_d2],
                    weight_decay=weight_decay,
                    lr=learning_rate1)
            else:
                # default
                self.optimizer = torch.optim.SGD(
                    params=[param_dict_enc, param_dict_d1, param_dict_d2],
                    weight_decay=weight_decay, lr=learning_rate1)
        else:
            if train_config["optimizer"].lower() == "adam":
                self.optimizer = torch.optim.Adam(
                    model.parameters(), weight_decay=weight_decay,
                    lr=learning_rate)
            else:
                # default
                self.optimizer = torch.optim.SGD(
                    model.parameters(), weight_decay=weight_decay,
                    lr=learning_rate)

        self.schedule_metric = train_config.get("schedule_metric",
                                                "eval_metric")
        if self.schedule_metric == "eval_metric":
            # if we schedule after BLEU/chrf, we want to maximize it
            scheduler_mode = "max"
        else:
            # if we schedule after loss or perplexity, we want to minimize it
            scheduler_mode = "min"
        self.scheduler = None
        if "scheduling" in train_config.keys() and \
                train_config["scheduling"]:
            if train_config["scheduling"].lower() == "plateau":
                # learning rate scheduler
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer=self.optimizer,
                    mode=scheduler_mode,
                    verbose=True,
                    threshold_mode='abs',
                    factor=train_config.get("decrease_factor", 0.1),
                    patience=train_config.get("patience", 10))
            elif train_config["scheduling"].lower() == "decaying":
                self.scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer=self.optimizer,
                    step_size=train_config.get("decaying_step_size", 10))
            elif train_config["scheduling"].lower() == "exponential":
                self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                        optimizer=self.optimizer,
                        gamma=train_config.get("decrease_factor", 0.99))
        self.shuffle = train_config.get("shuffle", True)
        self.epochs = train_config["epochs"]
        self.batch_size = train_config["batch_size"]
        self.batch_multiplier = train_config.get("batch_multiplier", 1)
        self.normalization = train_config.get("normalization", "batch")
        self.steps = 0
        # stop training if this flag is True by reaching learning rate minimum
        self.stop = False
        self.total_tokens = 0
        self.best_ckpt_score = 0
        self.best_ckpt_iteration = 0
        self.max_output_length = train_config.get("max_output_length", None)
        self.overwrite = train_config.get("overwrite", False)
        self.model_dir = self._make_model_dir(train_config["model_dir"])
        self.logger = self._make_logger()
        self.valid_report_file = "{}/validations.txt".format(self.model_dir)
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
            # deliberation: load decoders params for both decoders
            # pre-trained model can have 1 or 2 decoders (either base or delib model)
            model_load_path = train_config["load_model"]
            self.logger.info("Loading model from {}".format(model_load_path))
            self.load_checkpoint(model_load_path)

        trainable_params = [n for (n, p) in self.model.named_parameters()
                            if p.requires_grad]
        self.logger.info("Trainable parameters: {}".format(trainable_params))

    def save_checkpoint(self):
        """
        Save the model's current parameters and state to a checkpoint.
        :return:
        """
        #self.optimizer.state = defaultdict(dict, self.optimizer.state)
        model_path = "{}/{}.ckpt".format(self.model_dir, self.steps)
        state = {
            "steps": self.steps,
            "total_tokens": self.total_tokens,
            "best_ckpt_score": self.best_ckpt_score,
            "best_ckpt_iteration": self.best_ckpt_iteration,
            "model_state": self.model.state_dict(),
        }
        state["optimizer_state"] = self.optimizer.state_dict()
        state["scheduler_state"] = self.scheduler.state_dict() if \
            self.scheduler is not None else None
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
        param_dict = model_checkpoint["model_state"]
        if isinstance(self.model, DeliberationModel):
            self.logger.info("Initializing decoders.")
            # check if ckpt was also deliberation model
            if not any(["decoder1" in k for k in  param_dict.keys()]):
                new_param_dict = {}
                for name, param in param_dict.items():
                    if "decoder" in name:
                        name_suffix = name.split("decoder.")[1]
                        new_name1 = "decoder1." + name_suffix
                        new_name2 = "decoder2." + name_suffix
                        if not "att" in name:
                            new_param_dict[new_name1] = param
                            new_param_dict[new_name2] = param
                        else:
                            # first decoder loads pre-trained attention
                            new_param_dict[new_name1] = param
                            # second decoder: same src attention as first decoder
                            if "attention" in name:
                                new_param_dict[new_name2.replace("attention", "src_attention")] = param
                            elif "att_vector" in name:
                                new_param_dict[new_name2.replace("att_vector", "comb_att_vector")] = param
                    else:
                        new_param_dict[name] = param
                # TODO other initialization?
                # attention vector layer of 2nd decoder and attention between decoders have be initialized randomly (different shape)
                scale = 0.1
                init = lambda p: nn.init.uniform_(p, a=-scale, b=scale)
                #new_param_dict["decoder2.comb_att_vector_layer.weight"] = init(torch.empty_like(self.model.decoder2.comb_att_vector_layer.weight))
                #new_param_dict["decoder2.comb_att_vector_layer.bias"] = init(torch.empty_like(self.model.decoder2.comb_att_vector_layer.bias))

                # new layer to combine contexts
                new_param_dict["decoder2.context_comb_layer.weight"] = init(torch.empty_like(self.model.decoder2.context_comb_layer.weight))
                new_param_dict["decoder2.context_comb_layer.bias"] = init(torch.empty_like(self.model.decoder2.context_comb_layer.bias))

                new_param_dict["decoder2.d1_attention.key_layer.weight"] = init(torch.empty_like(self.model.decoder2.d1_attention.key_layer.weight))
                if isinstance(self.model.decoder2.d1_attention, BahdanauAttention):
                    # LuongAttention doesn't have these
                    new_param_dict["decoder2.d1_attention.query_layer.weight"] = init(torch.empty_like(self.model.decoder2.d1_attention.query_layer.weight))
                    new_param_dict["decoder2.d1_attention.energy_layer.weight"] = init(torch.empty_like(self.model.decoder2.d1_attention.energy_layer.weight))

                if self.model.baseline:
                    if "total_samples" not in param_dict:
                        # buffer for reward baselines
                        new_param_dict["total_samples"] = torch.zeros_like(self.model.total_samples)
                        new_param_dict["total_cost"] = torch.zeros_like(self.model.total_cost)
                    else:
                        new_param_dict["total_samples"] = param_dict["total_samples"]
                        new_param_dict["total_cost"] = param_dict["total_cost"]
            else:
                new_param_dict = param_dict
            self.model.load_state_dict(new_param_dict)

        else:
            self.model.load_state_dict(param_dict)

        if isinstance(self.model, DeliberationModel):
            if "optimizer_state" in model_checkpoint.keys():
                self.logger.info("Loading optimizer from 1 decoder.")
                opt_state_dict = model_checkpoint["optimizer_state"]
                try:
                    self.optimizer.load_state_dict(opt_state_dict)
                except ValueError:
                    self.logger.warning("Failed to load optimizer.")
            if isinstance(self.optimizer, torch.optim.SGD):
                # for SGD
                # https://discuss.pytorch.org/t/saving-and-loading-sgd-optimizer/2536/5
                #self.optimizer.state = defaultdict(dict, self.optimizer.state)
                # SGD requires stored momentum etc with params
                for p in self.optimizer.param_groups:
                    if "momentum" not in p.keys():
                        p["momentum"] = 0
                    if "dampening" not in p.keys():
                        p["dampening"] = 0
                    if "nesterov" not in p.keys():
                        p["nesterov"] = 0
            #elif "optimizer_state" in model_checkpoint.keys():
            #    self.logger.info("Loading optimizer from 1 decoder.")
            #    # TODO not possible in PyTorch, since it stores ids, not names see:
            #    # https://discuss.pytorch.org/t/load-optimizer-for-partial-parameters/2617
            #    # loaded model only had one decoder, optimizer, scheduler
            #    #print(model_checkpoint["optimizer_state"]["param_groups"])
            #    self.optimizer1.load_state_dict(
            #        model_checkpoint["optimizer_state"])
            #    self.optimizer2.load_state_dict(
            #        model_checkpoint["optimizer_state"])
            else:
                self.logger.warning("No optimizer loaded.")

            if "scheduler_state" in model_checkpoint.keys():
                self.scheduler.load_state_dict(
                    model_checkpoint["scheduler_state"])
            else:
                self.logger.warning("No scheduler loaded.")

        else:
            self.optimizer.load_state_dict(model_checkpoint["optimizer_state"])

            if model_checkpoint["scheduler_state"] is not None:
                self.scheduler.load_state_dict(model_checkpoint["scheduler_state"])

        # restore counts
        self.steps = model_checkpoint["steps"]
        self.total_tokens = model_checkpoint["total_tokens"]
        try:
            self.best_ckpt_score = model_checkpoint["best_ckpt_score"]
            self.best_ckpt_iteration = model_checkpoint["best_ckpt_iteration"]
        except KeyError:  # previous versions
            self.best_ckpt_score = model_checkpoint["best_valid_score"]
            self.best_ckpt_iteration = model_checkpoint["best_valid_iteration"]

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

    def validate_and_report(self, valid_data, epoch_no, total_valid_duration):
        valid_start_time = time.time()

        valid_result = validate_on_data(
            batch_size=self.batch_size, data=valid_data,
            eval_metric=self.eval_metric,
            level=self.level, model=self.model,
            use_cuda=self.use_cuda,
            max_output_length=self.max_output_length)
        if isinstance(self.model, DeliberationModel):
            (valid_score,
             aux_valid_score), valid_loss, valid_ppl, valid_sources, \
            valid_sources_raw, valid_references, (
                valid_hypotheses1, valid_hypotheses2), \
            (valid_hypotheses_raw1, valid_hypotheses_raw2), \
            (valid_attention_scores, valid_src_attention_scores,
             valid_d1_attention_scores) = valid_result
            valid_hypotheses = valid_hypotheses2
            valid_hypotheses_raw = valid_hypotheses_raw2
            # TODO pass attention scores on to later code
        else:
            valid_score, valid_loss, valid_ppl, valid_sources, \
            valid_sources_raw, valid_references, valid_hypotheses, \
            valid_hypotheses_raw, valid_attention_scores = valid_result
            aux_valid_score = None
        if valid_score > self.best_ckpt_score:
            self.best_ckpt_score = valid_score
            self.best_ckpt_iteration = self.steps
            self.logger.info('Hooray! New best validation result!')
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
            self.scheduler.step(schedule_score)

        # append to validation report
        self._add_report(
            valid_score=valid_score, valid_loss=valid_loss,
            valid_ppl=valid_ppl, eval_metric=self.eval_metric,
            new_best=self.steps == self.best_ckpt_iteration,
            aux_valid_score=aux_valid_score)

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
            if isinstance(self.model, DeliberationModel):
                self.logger.debug("\tRaw hypothesis Dec1: {}".format(
                    valid_hypotheses_raw1[p]))
                self.logger.debug("\tHypothesis Dec1: {}".format(
                    valid_hypotheses1[p]))

        valid_duration = time.time() - valid_start_time
        total_valid_duration += valid_duration
        self.logger.info(
            'Validation result at epoch {}, step {}: {}: {}, '
            'loss: {}, ppl: {}, duration: {:.4f}s'.format(
                epoch_no + 1, self.steps, self.eval_metric,
                valid_score, valid_loss, valid_ppl, valid_duration))
        if isinstance(self.model, DeliberationModel):
            self.logger.info('D1 Validation result at epoch {}, '
                             'step {}: {}: {}'.format(
                epoch_no + 1, self.steps, self.eval_metric,
                aux_valid_score))
        # store validation set outputs
        self.store_outputs(valid_hypotheses)

        # store attention plots for first three sentences of
        # valid data and one randomly chosen example
        store_attention_plots(attentions=valid_attention_scores,
                              targets=valid_hypotheses_raw,
                              sources=[s for s in valid_data.src],
                              idx=[0, 1, 2,
                                   np.random.randint(0, len(
                                       valid_hypotheses))],
                              output_prefix="{}/att.{}".format(
                                  self.model_dir,
                                  self.steps))
        return total_valid_duration

    def train_and_validate(self, train_data, valid_data):
        """
        Train the model and validate it from time to time on the validation set.
        :param train_data:
        :param valid_data:
        :return:
        """
        train_iter = make_data_iter(train_data, batch_size=self.batch_size,
                                    train=True, shuffle=self.shuffle)
        # initial validation before training
        self.validate_and_report(
            valid_data=valid_data, epoch_no=0,
            total_valid_duration=0)

        for epoch_no in range(self.epochs):
            self.logger.info("EPOCH {}".format(epoch_no + 1))
            self.model.train()

            start = time.time()
            total_valid_duration = 0
            processed_tokens = self.total_tokens
            count = 0

            for batch_no, batch in enumerate(iter(train_iter), 1):
                # reactivate training
                self.model.train()
                batch = Batch(batch, self.pad_index, use_cuda=self.use_cuda)
                # only update every batch_multiplier batches
                # https://medium.com/@davidlmorton/increasing-mini-batch-size-without-increasing-memory-6794e10db672
                update = count == 0
                #print(count, update, self.steps)
                batch_loss = self._train_batch(batch, update=update)
                count = self.batch_multiplier if update else count
                count -= 1

                # log learning progress
                if self.model.training and self.steps % self.logging_freq == 0 and update:
                    elapsed = time.time() - start - total_valid_duration
                    elapsed_tokens = self.total_tokens - processed_tokens
                    self.logger.info(
                        "Epoch %d Step: %d Loss: %f Tokens per Sec: %f" %
                        (epoch_no + 1, self.steps, batch_loss,
                         elapsed_tokens / elapsed))
                    start = time.time()
                    total_valid_duration = 0

                # validate on whole dev set
                if self.steps % self.validation_freq == 0 and update:
                    self.validate_and_report(
                        valid_data=valid_data, epoch_no=epoch_no,
                        total_valid_duration=total_valid_duration)
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
            self.best_ckpt_iteration, self.best_ckpt_score, self.eval_metric))

    def _train_batch(self, batch, update=True):
        """
        Train the model on one batch: Compute the loss, make a gradient step.
        :param batch:
        :return:
        """
        batch_loss = self.model.get_loss_for_batch(batch=batch)
        # TODO does it fit together if batch_loss is sum of batch?
        # normalize batch loss
        if self.normalization == "batch":
            normalizer = batch.nseqs
        elif self.normalization == "tokens":
            normalizer = batch.ntokens
        else:
            raise NotImplementedError("Only normalize by 'batch' or 'tokens'")

        norm_batch_loss = batch_loss.sum() / normalizer
        norm_batch_multiply = norm_batch_loss / self.batch_multiplier

        # compute gradients
        norm_batch_multiply.backward()

        if self.clip_grad_fun is not None:
            # clip gradients (in-place)
            self.clip_grad_fun(
                params=[p for p in self.model.parameters() if p.requires_grad])

        if update:
            # make gradient step
            self.optimizer.step()
            self.optimizer.zero_grad()
            # increment step
            self.steps += 1
        # increment token counter
        self.total_tokens += batch.ntokens
        return norm_batch_loss

    def _add_report(self, valid_score, valid_ppl, valid_loss, eval_metric,
                    new_best=False, aux_valid_score=None):
        """
        Add a one-line report to validation logging file.
        :param valid_score:
        :param valid_ppl:
        :param valid_loss:
        :param eval_metric:
        :param new_best:
        :param aux_valid_score: if 2 decoders, then this is for auxiliary
        :return:
        """
        current_lr = -1
        if isinstance(self.model, DeliberationModel):
            current_lrs = []
            for param_group in self.optimizer.param_groups:
                current_lrs.append(param_group['lr'])
            # stop if all lrs are smaller than minimum
            if type(self.learning_rate_min) is not list:
                self.learning_rate_min = [self.learning_rate_min,
                                          self.learning_rate_min,
                                          self.learning_rate_min]
            self.stop = all([clr < minlr for clr, minlr in
                             zip(current_lrs, self.learning_rate_min)])
            current_lr = "{}".format(current_lrs)

        else:
            # ignores other param groups for now
            for param_group in self.optimizer.param_groups:
                current_lr = param_group['lr']

            if current_lr < self.learning_rate_min:
                self.stop = True

        with open(self.valid_report_file, 'a') as opened_file:
            report_str = "Steps: {}\tLoss: {:.5f}\tPPL: {:.5f}\t{}: {:.5f}\t" \
                "LR: {}\t{}\n".format(
                    self.steps, valid_loss, valid_ppl, eval_metric,
                    valid_score, current_lr, "*" if new_best else "")
            if aux_valid_score is not None:
                report_str = report_str.strip()+"\tAux {}: {:.5f}\n".format(
                    eval_metric, aux_valid_score)
            opened_file.write(report_str)

    def store_outputs(self, hypotheses):
        """
        Write current validation outputs to file in model_dir.
        :param hypotheses:
        :return:
        """
        current_valid_output_file = "{}/{}.hyps".format(self.model_dir,
                                                        self.steps)
        with open(current_valid_output_file, 'w') as opened_file:
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

        if isinstance(model, DeliberationModel):
            # TODO
            pass
        else:
            score, loss, ppl, sources, sources_raw, references, hypotheses, hypotheses_raw, attention_scores  = validate_on_data(
                data=test_data, batch_size=trainer.batch_size,
                eval_metric=trainer.eval_metric, level=trainer.level,
                max_output_length=trainer.max_output_length,
                model=model, use_cuda=trainer.use_cuda,
                beam_size=beam_size, beam_alpha=beam_alpha)

            if "trg" in test_data.fields:
                decoding_description = "Greedy decoding" if beam_size == 0 else "Beam search decoding with beam size = {} and alpha = {}".format(beam_size, beam_alpha)
                trainer.logger.info("{:4s}: {} {} [{}]".format("Test data result", score, trainer.eval_metric, decoding_description))
            else:
                trainer.logger.info("No references given for {}.{} -> no evaluation.".format(cfg["data"]["test"],cfg["data"]["src"]))

            output_path_set = "{}/{}.{}".format(trainer.model_dir,"test",cfg["data"]["trg"])
            with open(output_path_set, mode="w", encoding="utf-8") as f:
                for h in hypotheses:
                    f.write(h + "\n")
            trainer.logger.info("Test translations saved to: {}".format(output_path_set))

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Joey-NMT')
    parser.add_argument("config", default="configs/default.yaml", type=str,
                        help="Training configuration file (yaml).")
    args = parser.parse_args()
    train(cfg_file=args.config)
