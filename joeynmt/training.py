# coding: utf-8
import argparse
import logging
import time
import os
import numpy as np
import shutil
from collections import OrderedDict
import random

import torch
import torch.nn as nn

from joeynmt.model import build_model

from joeynmt.batch import Batch
from joeynmt.helpers import log_data_info, load_data, \
    load_config, log_cfg, store_attention_plots, make_data_iter, \
    load_model_from_checkpoint
from joeynmt.prediction import validate_on_data


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
        test_config = config["testing"]
        self.beam_size = test_config.get("beam_size", 5)
        self.beam_alpha = test_config.get("beam_alpha", 1.0)
        self.model = model
        self.overwrite = train_config.get("overwrite", False)
        self.model_dir = self._make_model_dir(train_config["model_dir"])
        self.logger = self._make_logger()
        self.pad_index = self.model.pad_index
        self.bos_index = self.model.bos_index
        self.loss_weights = train_config.get("loss_weights", {"mt": 1.0, "regulator": 1.0})
        criterion = nn.NLLLoss(ignore_index=self.pad_index, reduction='none')
        self.learning_rate_min = train_config.get("learning_rate_min", 1.0e-8)
        if train_config["loss"].lower() not in ["crossentropy", "xent",
                                                "mle", "cross-entropy"]:
            raise NotImplementedError("Loss is not implemented. Only xent.")
        self.learning_rate = train_config.get("learning_rate", 3.0e-4)
        weight_decay = train_config.get("weight_decay", 0)

        if model.regulator is not None:
            self.logger.info("Regulator output labels: {}".format(self.model.regulator.index2label))
            self.init_from_mt = train_config.get("init_from_mt", False)
            # 2 sets of parameters -> 2 optimizers
            # see https://stackoverflow.com/questions/51578235/pytorch-how-to-get-the-gradient-of-loss-function-twice
            all_params = list(model.named_parameters())
            # sorting is required to keep track of parameter groups for optimizers
            sorted_reg = sorted(
                {k: v for (k, v) in all_params if
                 "reg" in k}.items())  # if "corrector" in key
            self.regulator_params = OrderedDict(sorted_reg)
            self.logger.info(
                "REGULATOR PARAMS: {}".format(self.regulator_params.keys()))
            self.logger.info("REGULATOR size: {}".format(
                sum([np.prod(p.size()) for p in self.regulator_params.values()])
            ))
            sorted_mt = sorted(
                {k: v for (k, v) in all_params if "reg" not in k}.items())
            self.mt_params = OrderedDict(sorted_mt)
            self.logger.info("MT PARAMS: {}".format(self.mt_params.keys()))
            self.logger.info("MT size: {}".format(
                sum([np.prod(p.size()) for p in self.mt_params.values()])))

            for k, v in model.named_parameters():
                if v.requires_grad:
                    self.logger.debug("Updating {}".format(k))
                else:
                    self.logger.debug("NOT Updating {}".format(k))

            self.optimizer = {}
            if train_config["optimizer"].lower() == "adam":
                self.optimizer["mt"] = torch.optim.Adam(
                    self.mt_params.values(), weight_decay=weight_decay,
                    lr=self.learning_rate["mt"])
                self.optimizer["regulator"] = torch.optim.Adam(
                    self.regulator_params.values(), weight_decay=weight_decay,
                    lr=self.learning_rate["regulator"])
            else:
                # default
                self.optimizer["mt"] = torch.optim.SGD(
                    self.mt_params.values(), weight_decay=weight_decay,
                    lr=self.learning_rate["mt"])
                self.optimizer["regulator"] = torch.optim.SGD(
                    self.regulator_params.values(), weight_decay=weight_decay,
                    lr=self.learning_rate["regulator"])
        else:
            all_params = list(model.named_parameters())
            sorted_mt = sorted({k: v for (k, v) in all_params}.items())
            self.mt_params = OrderedDict(sorted_mt)
            self.logger.info("MT PARAMS: {}".format(self.mt_params.keys()))
            self.logger.info("MT size: {}".format(
                sum([np.prod(p.size()) for p in self.mt_params.values()])))

            # order is needed for reloading a different model
            if train_config["optimizer"].lower() == "adam":
                self.optimizer = torch.optim.Adam(
                    self.mt_params.values(), weight_decay=weight_decay,
                    lr=self.learning_rate)
            else:
                # default
                self.optimizer = torch.optim.SGD(
                    self.mt_params.values(), weight_decay=weight_decay,
                    lr=self.learning_rate)
        self.eval_metric = train_config.get("eval_metric", "bleu")
        self.schedule_metric = train_config.get("schedule_metric",
                                                "eval_metric")
        self.trainable_params = [n for (n, p) in self.model.named_parameters()
                                 if p.requires_grad]
        self.logger.info("Trainable parameters: {}".format(
            self.trainable_params))
        self.ckpt_metric = train_config.get("ckpt_metric", "eval_metric")
        self.best_ckpt_iteration = 0
        # if we schedule after BLEU/chrf, we want to maximize it, else minimize
        if self.schedule_metric == "eval_metric" and self.eval_metric in ["bleu", "chrf"]:
            scheduler_mode = "max"
        else:
            scheduler_mode = "min"
        # the ckpt metric decides on how to find a good early stopping point:
        # ckpts are written when there's a new high/low score for this metric
        if self.ckpt_metric == "eval_metric" and self.eval_metric in ["bleu", "chrf"]:
            self.best_ckpt_score = -np.inf
            self.is_best = lambda x: x > self.best_ckpt_score
        else:
            self.best_ckpt_score = np.inf
            self.is_best = lambda x: x < self.best_ckpt_score
        self.scheduler = None
        self.patience = train_config.get("patience", 10)
        if "scheduling" in train_config.keys() and \
                train_config["scheduling"]:
            if train_config["scheduling"].lower() == "plateau":
                if model.regulator is not None:
                    # 2 schedulers
                    self.scheduler = {
                    k: torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer=v,
                        mode=scheduler_mode,
                        verbose=False,
                        threshold_mode='abs',
                        factor=train_config.get("decrease_factor", 0.1),
                        patience=self.patience)
                    for k, v in sorted(self.optimizer.items())}
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
                if model.regulator is not None:
                    # 2 schedulers
                    self.scheduler = {k: torch.optim.lr_scheduler.StepLR(
                        optimizer=v,
                        step_size=train_config.get("decaying_step_size",
                                                   10))
                                      for k, v in
                                      sorted(self.optimizer.items())}
                else:
                    self.scheduler = torch.optim.lr_scheduler.StepLR(
                        optimizer=self.optimizer,
                        step_size=train_config.get("decaying_step_size",
                                                   10))
            elif train_config["scheduling"].lower() == "exponential":
                if model.regulator is not None:
                    # 2 schedulers
                    self.scheduler = {
                    k: torch.optim.lr_scheduler.ExponentialLR(
                        optimizer=v,
                        gamma=train_config.get("decrease_factor", 0.99)
                    ) for k, v in sorted(self.optimizer.items())}
                else:
                    self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                        optimizer=self.optimizer,
                        gamma=train_config.get("decrease_factor", 0.99)
                    )
        self.shuffle = train_config.get("shuffle", True)
        self.epochs = train_config["epochs"]
        self.batch_size = train_config["batch_size"]
        self.valid_batch_size = train_config.get("valid_batch_size",
                                                 self.batch_size)
        self.batch_multiplier = train_config.get("batch_multiplier", 1)
        self.criterion = criterion
        self.normalization = train_config.get("normalization", "batch")
        self.steps = 0
        # stop training if this flag is True by reaching learning rate minimum
        self.stop = False
        self.total_tokens = 0
        self.max_output_length = train_config.get("max_output_length", None)
        self.valid_report_file = "{}/validations.txt".format(self.model_dir)
        self.use_cuda = train_config["use_cuda"]
        if self.use_cuda:
            self.model.cuda()
        self.logging_freq = train_config.get("logging_freq", 100)
        self.validation_freq = train_config.get("validation_freq", 1000)
        self.print_valid_sents = train_config["print_valid_sents"]
        self.level = config["data"]["level"]
        self.lowercase = config["data"].get("lowercase", False)
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

        trainable_params = [n for (n, p) in self.model.named_parameters()
                            if p.requires_grad]
        self.logger.info("Trainable parameters: {}".format(trainable_params))
        assert len(trainable_params) > 0

        # statistics for regulation
        # TODO load them from previous model
        self.regulator_outputs = []
        self.total_cost = 0 #train_config.get("budget", 0)
        self.baseline = train_config.get("baseline", False)
        self.entropy_regularizer = train_config.get("entropy_regularizer", 0)
        self.cost_weight = train_config.get("cost_weight", 1.0)
        #assert 1 > self.cost_weight > 0
        #self.logger.info("Initial budget: {}".format(self.budget))
        self.rewards = []
        #self.budgeted_cost = train_config.get("budgeted_cost", False)
        self.only_sup = train_config.get("only_sup", False)
        self.chunk_type = train_config.get("chunk_type", "marking")
        self.logger.info("Learning with feedback of type {}".format(
            self.chunk_type))
        self.self_entropy = train_config.get("self_entropy", False)
        if self.self_entropy:
            self.logger.info("Maximizing entropy for self-training.")
        self.weak_baseline = train_config.get("weak_baseline", False)
        if self.weak_baseline:
            self.logger.info("Using baseline for weak feedback.")
        self.weak_search = train_config.get("weak_search", "sample")
        self.logger.info("Using {} for weak feedback.".format(self.weak_search))
        self.weak_temperature = train_config.get("weak_temperature", 1.0)
        self.logger.info("Using temp={} for weak feedback.".format(self.weak_temperature))
        self.weak_case_sensitive = train_config.get("weak_case_sensitive", True)
        self.logger.info("Using case sensitive weak feedback? {}".format(self.weak_case_sensitive))
        self.pe_ratio = train_config.get("pe_ratio", 1.0)
        self.logger.info("Using PEs with ratio {}".format(self.pe_ratio))
        self.logger.info("Decoding with beam size={} and alpha={} for feedback.".format(self.beam_size, self.beam_alpha))
        self.logger.info("Regulator baseline {}".format(self.baseline))

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
            "model_state": self.model.state_dict()
        }
        if type(self.optimizer) is not dict:
            state["optimizer_state"] = self.optimizer.state_dict()
            state["scheduler_state"] = self.scheduler.state_dict() if \
                self.scheduler is not None else None
        else:
            for k, v in sorted(self.optimizer.items()):
                state["{}_optimizer_state".format(k)] = v.state_dict()
            for k, v in sorted(self.scheduler.items()):
                state["{}_scheduler_state".format(k)] = v.state_dict()
                # if self.scheduler is not None else None,

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
        #self.model.load_state_dict(model_checkpoint["model_state"])

        # TODO initialize embeddings and src encoder with MT params if possible

        # restore model
        param_dict = model_checkpoint["model_state"]

        if self.model.regulator is not None:
            new_param_dict = {}
            curr_state_dict = self.model.state_dict()
            for name, param in param_dict.items():
                # if regulator params are different shape or don't exist: initialize newly
                if "reg" in name:
                    # regulator param
                    # params that are in loaded model but not in current model
                    if name not in curr_state_dict:
                        self.logger.info("Skipping loading of {}".format(name))
                        continue
                    # params that have different shapes
                    elif curr_state_dict[name].shape != param.shape:
                        self.logger.info(
                            "Shape mismatch for param {}: old {}, new {}. "
                            "Initializing randomly.".format(
                                name, param.shape, curr_state_dict[name].shape))
                        if "output_layer" in name:
                            obj = self.model.regulator.output_layer
                           # print(getattr(obj, name.split(".")[-1]))
                            new_param_dict[name] = getattr(obj, name.split(".")[-1]) #init(torch.empty_like(getattr(obj, name.split(".")[-1])))
                        elif "src_rnn" in name:
                            obj = self.model.regulator.src_rnn
                            new_param_dict[name] = getattr(obj,
                                                           name.split(".")[-1])
                        elif "middle_layer" in name:
                            obj = self.model.regulator.middle_layer
                            new_param_dict[name] = getattr(obj,
                                                           name.split(".")[-1])
                        elif "reg_src_embed.lut" in name:
                            obj = self.model.reg_src_embed.lut
                            new_param_dict[name] = getattr(obj,
                                                           name.split(".")[-1])

                    else:
                        new_param_dict[name] = param
                else:
                    # standard loading
                    new_param_dict[name] = param

            # now add params that are not there yet
            for name, param in curr_state_dict.items():
                if name not in new_param_dict.keys():
                    self.logger.info("{} doesn't exist in loaded checkpoint. Initializing randomly.".format(name))
                    if "output_layer" in name:
                        obj = self.model.regulator.output_layer
                        # print(getattr(obj, name.split(".")[-1]))
                        new_param_dict[name] = getattr(obj, name.split(".")[
                            -1])  # init(torch.empty_like(getattr(obj, name.split(".")[-1])))
                    elif "src_rnn" in name:
                        obj = self.model.regulator.src_rnn
                        new_param_dict[name] = getattr(obj,
                                                       name.split(".")[-1])
                        # TODO also init src from MT
                    elif "middle_layer" in name:
                        obj = self.model.regulator.middle_layer
                        new_param_dict[name] = getattr(obj,
                                                       name.split(".")[-1])
                    elif "reg_src_embed.lut" in name:
                        if param_dict["src_embed.lut.weight"].shape == self.model.reg_src_embed.lut.weight.shape and self.init_from_mt:
                            self.logger.info("Initializing regulator embeddings from MT embeddings.")
                            new_param_dict[name] = param_dict["src_embed.lut.weight"]

                        else:
                            obj = self.model.reg_src_embed.lut
                            new_param_dict[name] = getattr(obj,
                                                           name.split(".")[-1])


            self.model.load_state_dict(new_param_dict)
        else:
            self.model.load_state_dict(model_checkpoint["model_state"])

        # restore optimizer parameters
        if type(self.optimizer) == dict:
            # loaded model did have a regulator
            if "mt_optimizer_state" in model_checkpoint:
                self.optimizer["mt"].load_state_dict(
                    model_checkpoint["mt_optimizer_state"])
                self.logger.info("MT optimizer: {}".format(self.optimizer["mt"]))
            # loaded model didn't have a regulator
            elif "optimizer_state" in model_checkpoint:
                self.optimizer["mt"].load_state_dict(
                    model_checkpoint["optimizer_state"])
                self.logger.info("MT optimizer: {}".format(self.optimizer["mt"]))
            # else: newly created
            else:
                self.logger.info("Newly creating optimizer for MT.")
                pass
            # now for regulator
            if "regulator_optimizer_state" in model_checkpoint:
                self.optimizer["regulator"].load_state_dict(
                    model_checkpoint["regulator_optimizer_state"])
            else:  # doesn't have any (newly created)
                self.logger.info("Newly creating optimizer for regulator.")
                pass

            # overwrite learning rates
            for o_name, o in self.optimizer.items():
                for i, param_group in enumerate(o.param_groups):
                    old_lr = float(param_group['lr'])
                    new_lr = self.learning_rate[o_name]
                    if old_lr != new_lr:
                        param_group['lr'] = new_lr
                        self.logger.info(
                            'Optimizer for {}: Overwriting learning rate'
                            ' of group {} from {:.4e} to {:.4e}.'.format(
                                o_name, i, old_lr, new_lr))
                    # needed for SGD
                    weight_decay = param_group.get('weight_decay', 0)
                    momentum = param_group.get('momentum', 0)
                    dampening = param_group.get('dampening', 0)
                    nesterov = param_group.get('nesterov', 0)
                    param_group['weight_decay'] = weight_decay
                    param_group['momentum'] = momentum
                    param_group['dampening'] = dampening
                    param_group['nesterov'] = nesterov
        else:
            self.optimizer.load_state_dict(model_checkpoint["optimizer_state"])

        if type(self.scheduler) == dict:
            if "mt_scheduler_state" in model_checkpoint:
                self.scheduler["mt"].load_state_dict(
                    model_checkpoint["mt_scheduler_state"])
            elif "scheduler_state" in model_checkpoint:
                self.scheduler["mt"].load_state_dict(
                    model_checkpoint["scheduler_state"])
            else:
                self.logger.info("Newly creating scheduler for MT.")
                pass
            if "regulator_scheduler_state" in model_checkpoint:
                self.scheduler["regulator"].load_state_dict(
                    model_checkpoint["regulator_scheduler_state"])
            else:
                self.logger.info("Newly creating scheduler for regulator.")
                pass

            for s_name, s in self.scheduler.items():
                new_patience = self.patience
                old_patience = s.patience
                if old_patience != new_patience:
                    s.patience = new_patience
                    self.logger.info(
                        'Scheduler for {}: Overwriting patience'
                        ' from {:.4e} to {:.4e}.'.format(
                            s_name, old_patience, new_patience))
        else:
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
        for epoch_no in range(self.epochs):
            self.logger.info("EPOCH {}".format(epoch_no + 1))
            self.model.train()

            start = time.time()
            total_valid_duration = 0
            processed_tokens = self.total_tokens
            count = self.batch_multiplier-1
            # collect for regulator while update=False
            all_reg_log_probs = []
            all_reg_preds = []
            batch_costs = []

            for batch_no, batch in enumerate(iter(train_iter), 1):
                # reactivate training
                self.model.train()
                batch = Batch(batch, self.pad_index, use_cuda=self.use_cuda)

                # only update every batch_multiplier batches
                # see https://medium.com/@davidlmorton/increasing-mini-batch-size-without-increasing-memory-6794e10db672
                update = count == 0

                # print(count, update, self.steps)
                batch_loss, reg_log_probs, reg_pred, costs = \
                    self._train_batch_mt(batch, update=update, pred=self.only_sup)
                all_reg_log_probs.append(reg_log_probs)
                all_reg_preds.append(reg_pred)
                batch_costs.append(costs)

                if reg_pred is not None:
                    self.regulator_outputs.extend(reg_pred.detach().cpu().numpy())
                else:
                    entropy = None

                reg_batch_loss = None
                if self.model.regulator is not None:
                    # now without fixed budget
                    # # TODO this is not exact
                    # if self.budget < 0:
                    #     self.stop = True
                    #     self.logger.info("Training ended since budget is consumed.")

                    reg_batch_loss = 0
                    # regulator can only be trained when validation is performed, which only happens after a full batch_multiplier*batch_size batch
                    if self.loss_weights["regulator"] > 0 and update:
                        # TODO what's the validation criterion? instead of BLEU could be regret
                        with torch.no_grad():
                            valid_score_immediate, valid_loss_immediate, \
                            valid_ppl_immediate, _, \
                            _, _, _, \
                            _, _ = \
                                validate_on_data(
                                    batch_size=self.valid_batch_size, data=valid_data,
                                    eval_metric=self.eval_metric,
                                    level=self.level, model=self.model,
                                    use_cuda=self.use_cuda,
                                    max_output_length=self.max_output_length,
                                    # but without running train input again
                                    criterion=None,
                                    case_sensitive=not self.lowercase)
                        #print("reward", valid_score_immediate)
                        reward = valid_score_immediate/100

                        total_valid_duration = self.process_validation(
                            epoch_no=epoch_no, valid_hypotheses=None,
                            valid_hypotheses_raw=None,
                            valid_sources_raw=None,
                            valid_sources=None,
                            valid_references=None,
                            valid_attention_scores=None,
                            valid_loss=valid_loss_immediate, valid_score=valid_score_immediate,
                            valid_ppl=valid_ppl_immediate, store_attention=False,
                            store_outputs=False, valid_start_time=0)

                        self.model.train()
                        # use validation result to update regulator

                        # TODO include cost in reward baseline!
                        #if self.baseline is not False:
                        #    # either mean
                        #    if self.baseline == "mean":
                        #        baseline_reward = np.mean(self.rewards) if len(self.rewards) > 0 else 0
                        #    # or previous
                        #    elif self.baseline == "previous":
                        #        num_previous = len(self.rewards)-1
                        #        baseline_reward = self.rewards[-2] if num_previous > 0 else 0
                        #    # first
                        #    elif self.baseline == "first":
                        #        baseline_reward = self.rewards[0] if len(self.rewards) > 0 else 0
                        #    # mean of previous x
                        #    elif self.baseline == "window":
                        #        window_size = 5
                        #        num_previous = len(self.rewards)-1
                        #        baseline_reward = np.mean(self.rewards[-window_size+1:-1] if num_previous > window_size else 0)

                            #print(self.rewards)
                        #    #print("baseline", baseline_reward)
                        #    reward -= baseline_reward
                            #print("reward with baseline", reward)

                        assert len(all_reg_log_probs) == self.batch_multiplier
                        assert len(all_reg_preds) == self.batch_multiplier
                        assert len(batch_costs) == self.batch_multiplier
                        all_costs = torch.cat(batch_costs, 0)
                        reg_batch_loss, entropy = \
                            self._train_batch_regulator(
                                regulator_log_probs=torch.cat(all_reg_log_probs, 0),
                                regulator_pred=torch.cat(all_reg_preds, 0),
                                reward=reward, update=update, costs=all_costs)
                        batch_costs = []
                        self.total_cost += all_costs.sum()
                        # TODO this is not exact
                        #if self.budget < 0:
                        #    self.stop = True
                        #    self.logger.info("Training ended since budget is consumed.")

                    #print("reg batch loss", reg_batch_loss)

                count = self.batch_multiplier if update else count
                count -= 1

                # reset collection of reg preds when update has happened
                if update:
                    all_reg_log_probs = []
                    all_reg_preds = []


                # log learning progress
                if self.model.training and self.steps % self.logging_freq == 0 \
                        and update:
                    elapsed = time.time() - start - total_valid_duration
                    elapsed_tokens = self.total_tokens - processed_tokens
                    self.logger.info(
                        "Epoch {} Step: {} MT Loss: {} Reg Loss: {} Reg Entropy: {} "
                        "Tokens per Sec: {}. Total cost: {}".format(
                            epoch_no + 1, self.steps, batch_loss, reg_batch_loss,
                            entropy,
                            elapsed_tokens / elapsed, self.total_cost))
                    start = time.time()
                    total_valid_duration = 0

                # validate on the entire dev set
                if self.steps % self.validation_freq == 0 and update:
                    valid_start_time = time.time()

                    valid_score, valid_loss, valid_ppl, valid_sources, \
                        valid_sources_raw, valid_references, valid_hypotheses, \
                        valid_hypotheses_raw, valid_attention_scores = \
                        validate_on_data(
                            batch_size=self.valid_batch_size, data=valid_data,
                            eval_metric=self.eval_metric,
                            level=self.level, model=self.model,
                            use_cuda=self.use_cuda,
                            max_output_length=self.max_output_length,
                            criterion=self.criterion,
                            case_sensitive=not self.lowercase)

                    total_valid_duration = self.process_validation(
                        epoch_no=epoch_no, valid_hypotheses=valid_hypotheses,
                        valid_hypotheses_raw=valid_hypotheses_raw,
                        valid_sources_raw=valid_sources_raw,
                        valid_sources=valid_sources,
                        valid_references=valid_references,
                        valid_attention_scores=valid_attention_scores,
                        valid_loss=valid_loss, valid_score=valid_score,
                        valid_ppl=valid_ppl, store_attention=True,
                        store_outputs=True, valid_start_time=valid_start_time)

                if self.stop:
                    break

            if self.stop:
                #self.logger.info(
                #    'Training ended since minimum lr {} was reached.'.format(
                #        self.learning_rate_min))
                self.logger.info("Training ended.")
                break
        else:
            self.logger.info('Training ended after {} epochs.'.format(
                epoch_no+1))
        self.logger.info('Best validation result at step {}: {} {}.'.format(
            self.best_ckpt_iteration, self.best_ckpt_score, self.ckpt_metric))

    def process_validation(self, valid_loss, valid_ppl, valid_score,
                           valid_sources_raw, valid_sources, valid_references,
                           valid_hypotheses_raw, valid_hypotheses,
                           valid_attention_scores,
                           valid_start_time, epoch_no,
                           store_outputs=True, store_attention=False):
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
        elif self.schedule_metric == "eval_metric":
            # schedule based on evaluation score
            schedule_score = valid_score

        if self.scheduler is not None:
            if type(self.scheduler) is dict:
                # regulator is scheduled after eval metric
                # TODO schedule lr after smth else?
                if self.loss_weights["regulator"] > 0:
                    self.scheduler["regulator"].step(
                        valid_score)
                if self.loss_weights["mt"] > 0:
                    # make scheduler step for MT model
                    self.scheduler["mt"].step(schedule_score)
            else:
                self.scheduler.step(schedule_score)
        # append to validation report
        self._add_report(
            valid_score=valid_score, valid_loss=valid_loss,
            valid_ppl=valid_ppl, eval_metric=self.eval_metric,
            new_best=new_best)

        if valid_sources is not None and valid_references is not None \
                and valid_hypotheses is not None:
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
        valid_duration = time.time() - valid_start_time
        self.logger.info(
            'Validation result at epoch {}, step {}: {}: {}, '
            'loss: {}, ppl: {}, duration: {:.4f}s'.format(
                epoch_no + 1, self.steps, self.eval_metric,
                valid_score, valid_loss, valid_ppl, valid_duration))

        # store validation set outputs
        if store_outputs and valid_hypotheses is not None:
            self.store_outputs(valid_hypotheses)

        if store_attention and valid_attention_scores is not None:
            # store attention plots for first three sentences of
            # valid data and one randomly chosen example
            store_attention_plots(attentions=valid_attention_scores,
                                  targets=valid_hypotheses_raw,
                                  sources=[s for s in valid_sources_raw],
                                  idx=[0, 1, 2,
                                       np.random.randint(0, len(
                                           valid_hypotheses))],
                                  output_prefix="{}/att.{}".format(
                                      self.model_dir,
                                      self.steps))
        return valid_duration

    def _train_batch_mt(self, batch, update=True, pred=False):
        """
        Train the model on one batch: Compute the loss, make a gradient step.

        :param batch:
        :param update: if False, only store gradient. if True also make update
        :return:
        """
        # only run regulator if its loss is > 0

        batch_loss, regulator_log_probs, regulator_pred, batch_tokens, batch_seqs, \
            batch_costs = self.model.get_loss_for_batch(
                        batch=batch, criterion=self.criterion,
                        regulate=self.loss_weights["regulator"] > 0 and
                                 self.model.regulator is not None,
                        pred=pred,
                        max_output_length=self.max_output_length,
                        chunk_type=self.chunk_type, level=self.level,
                        entropy=self.self_entropy,
                        weak_search=self.weak_search,
                        weak_baseline=self.weak_baseline,
                        weak_temperature=self.weak_temperature,
                        logger=self.logger,
                        case_sensitive=self.weak_case_sensitive,
                        pe_ratio=self.pe_ratio,
                        beam_size=self.beam_size,
                        beam_alpha=self.beam_alpha)

        if batch_loss is None:
            # if no supervision is chosen for whole batch -> no cost
            return None, regulator_log_probs, regulator_pred, regulator_pred.new(regulator_pred.size()).zero_().float()

        # normalize batch loss
        # counts might be adapted since parts of batch are ignored through reg. choice
        if self.normalization == "batch":
            normalizer = batch_seqs
        elif self.normalization == "tokens":
            normalizer = batch_tokens
        else:
            raise NotImplementedError("Only normalize by 'batch' or 'tokens'")

        norm_batch_loss = torch.mul(batch_loss.sum(), 1/normalizer)
        # division needed since loss.backward sums the gradients until updated
        norm_batch_multiply = norm_batch_loss / self.batch_multiplier

        norm_batch_multiply = self.loss_weights["mt"] * norm_batch_multiply

        if not all(["regulator" in p for p in self.trainable_params]) and \
                        self.loss_weights["mt"] > 0.:
            # compute gradient
            norm_batch_multiply.backward(retain_graph=True)

        # gradients for regulator parameters should be zero
#        assert not any(["reg" in k for k,v in self.model.named_parameters()
#                        if v.grad is not None and torch.norm(v.grad) > 0])

        # only update MT params

        if self.clip_grad_fun is not None:
            # clip gradients (in-place)
            self.clip_grad_fun(params=self.model.parameters())

        if update:
            # make gradient step
            if type(self.optimizer) is dict:
                # two optimizers with 2 different learning rates
                if self.loss_weights["mt"] > 0:
                    self.optimizer["mt"].step()
                    self.optimizer["mt"].zero_grad()
               # if self.loss_weights["regulator"] > 0:
               #     self.optimizer["regulator"].step()
               #     for name, param in self.corrector_params.items():
               #         if param.requires_grad:
               #             # set the gradients back to zero
               #             param.grad = param.grad.detach()
               #             param.grad.zero_()
               #             # self.optimizer["corrector"].zero_grad()
            else:
                self.optimizer.step()
                self.optimizer.zero_grad()

            # increment step counter
            self.steps += 1

        # increment token counter
        self.total_tokens += batch.ntokens

        return norm_batch_loss, regulator_log_probs, regulator_pred, batch_costs

    def _train_batch_regulator(self, regulator_log_probs, regulator_pred, reward, costs, update=True):
        # regulator_log_prob*(reward-cost)
        # compute cost
       # costs = self.model.regulator.get_costs(regulator_pred.detach().cpu().numpy(), hyps=, refs=)
        # select correct part of log probs with nll
        #print("reg log prob", regulator_log_probs.shape)
        #print("regulator pred", regulator_pred.shape)
        # TODO include cost in basleine?
        # .view(regulator_pred.size(0), -1)
        nll = self.criterion(input=regulator_log_probs, target=regulator_pred.detach())
        #print("costs", costs)  # batch_size
        #print(self.budget)
        #budget_used = 1-(self.budget/max(self.initial_budget, 1))
        #print("budget used", budget_used)
        # cost is scaled by budget percentage that is already used
        # the smaller the remaining budget, the more grows the cost
        #if self.budgeted_cost:
        #    budgeted_costs = costs*(1+budget_used)
        #    #print("cost with budget", budgeted_costs)
        #else:
        #    budgeted_costs = costs
        self.logger.debug("nll: {}".format(nll))
        #print("logprob", regulator_log_probs)
        # introduce parameter for interpolation
        #trade_off = (1-self.cost_weight)*(1-reward) + self.cost_weight*budgeted_costs
        # improvement in reward per increase in cost -> (r-r_prev) / cost
        self.logger.info("COST: {}, REWARD: {}".format(costs, reward))
        self.logger.info("PREDS: {}".format(regulator_pred))
        #trade_off = reward - self.cost_weight*costs
        #trade_off = reward / (costs+1)
        #self.logger.debug("trade_off: {}".format(trade_off))
        #if self.baseline and len(self.rewards) > 0:
        #    self.logger.debug("MEAN BASELINE: {}".format(np.mean(self.rewards)))
        #    trade_off = trade_off - trade_off.new([np.mean(self.rewards)])
        #    self.logger.debug("trade_off-BL: {}".format(trade_off))

        # keeping track of rewards (not costs)
        self.rewards.append(reward)

        self.logger.info("BASELINE {} ({})".format(self.baseline, len(self.rewards)))
        if self.baseline and len(self.rewards) > 0:
            # either mean
            baseline_reward = 0
            if self.baseline == "mean":
                 baseline_reward = np.mean(self.rewards) if len(self.rewards) > 0 else 0
            # or previous
            elif self.baseline == "previous":
                num_previous = len(self.rewards)-1
                baseline_reward = self.rewards[-2] if num_previous > 0 else 0
            # first
            elif self.baseline == "first":
                baseline_reward = self.rewards[0] if len(self.rewards) > 0 else 0
            # mean of previous x
            elif self.baseline == "window":
                window_size = 5
                num_previous = len(self.rewards)-1
                baseline_reward = np.mean(self.rewards[-window_size+1:-1] if num_previous > window_size else 0)
            else:
                raise ValueError("Invalid baseline specified.")

            self.logger.info("{} BASELINE: {}".format(self.baseline, np.mean(self.rewards)))
            reward -= baseline_reward
        trade_off = costs.new([reward*100]) / (costs+1)
        if self.entropy_regularizer > 0:
            entropy_penalty = self.entropy_regularizer*-nll - self.entropy_regularizer
            self.logger.info("entropy penalty {}".format(entropy_penalty))
            trade_off -= entropy_penalty

        self.logger.debug("trade_off: {}".format(trade_off))
        # baseline does not include cost
        #self.rewards.append(reward)
        #self.logger.info(nll)
        #self.logger.info(trade_off)
        #self.logger.info(trade_off.cuda())
        #self.logger.info(trade_off.cuda().detach())
#        #self.logger.info(costs.new(trade_off).to(regulator_log_probs.device).cuda().detach())
        #self.logger.info(costs.new(trade_off))
        #self.logger.info(costs.new(trade_off).to(regulator_log_probs.device))
        reg_loss = torch.mul(nll, costs.new(trade_off)) #costs.new(trade_off).to(regulator_log_probs.device).cuda().detach()) #regulator_log_probs.new(trade_off)) .to(regulator_log_probs.device).detach()) #regulator_log_probs.new([reward])
        #print("loss", reg_loss) # batch_size

        #entropy = 0
        #if self.entropy_regularizer > 0:
        # entropy of all outputs
        entropy = -torch.mul(torch.exp(regulator_log_probs),
                                 regulator_log_probs).sum(1)  # batch_size
        #   # print("entropy", entropy)
        #    reg_loss += self.entropy_regularizer*entropy
        #    entropy = entropy.sum()

        # average over batch
        norm_batch_loss = reg_loss.mean(0)
        #print("norm_batch loss", norm_batch_loss)

        # division needed since loss.backward sums the gradients until updated
        norm_batch_multiply = norm_batch_loss / self.batch_multiplier

        norm_batch_multiply = self.loss_weights["regulator"] * norm_batch_multiply

        if any(["regulator" in p for p in self.trainable_params]) and \
                        self.loss_weights["regulator"] > 0.:
            # compute gradient
            norm_batch_multiply.backward() # retain_graph=True

        # gradients for regulator parameters should be zero
        #assert not any(["reg" in k for k, v in self.model.named_parameters()
        #                if v.grad is not None and torch.norm(v.grad) > 0])
        #print("reg grads",
        #      [(k, torch.norm(v.grad) if v.grad is not None else None) for k,v in self.model.named_parameters()])

        # only update regulator params

        if self.clip_grad_fun is not None:
            # clip gradients (in-place)
            self.clip_grad_fun(params=self.model.parameters())

        if update:
            # make gradient step
            if type(self.optimizer) is dict:
                # two optimizers with 2 different learning rates
                if self.loss_weights["regulator"] > 0:
                    self.optimizer["regulator"].step()
                    self.optimizer["regulator"].zero_grad()
                    # if self.loss_weights["regulator"] > 0:
                    #     self.optimizer["regulator"].step()
                    #     for name, param in self.corrector_params.items():
                    #         if param.requires_grad:
                    #             # set the gradients back to zero
                    #             param.grad = param.grad.detach()
                    #             param.grad.zero_()
                    #             # self.optimizer["corrector"].zero_grad()
            else:
                self.optimizer.step()
                self.optimizer.zero_grad()

        return norm_batch_loss, entropy

    def _add_report(self, valid_score, valid_ppl, valid_loss, eval_metric,
                    new_best=False):
        """
        Add a one-line report to validation logging file.

        :param valid_score:
        :param valid_ppl:
        :param valid_loss:
        :param eval_metric:
        :param new_best:
        :return:
        """
        current_lr = -1
        if type(self.optimizer) is dict:
            current_lr = {k: v.param_groups[0]["lr"] for
                          k, v in sorted(self.optimizer.items())}

        else:        # ignores other param groups for now
            for param_group in self.optimizer.param_groups:
                current_lr = param_group['lr']

        if type(current_lr) is dict:
            # only stop if all learning rates have reached minimum
            self.stop = all(
                [v < self.learning_rate_min for v in current_lr.values()])
        else:
            if current_lr < self.learning_rate_min:
                self.stop = True

        # TODO use sent-score
        report_str = "Steps: {}\tLoss: {:.5f}\tPPL: {:.5f}\tMT-{}: {:.5f}\t" \
                     "MT-s{}: {:.5f}".format(
            self.steps, valid_loss, valid_ppl, eval_metric,
            valid_score, eval_metric, valid_score)

        if self.loss_weights["regulator"] > 0:
            # add statistics
            report_str += "\t Avg_Reward: {}".format(np.mean(self.rewards))
            report_str += "\t Total_Cost: {}".format(self.total_cost)
            current_reg_out = self.regulator_outputs[-self.batch_size:]
            total = self.batch_size
            report_str += "\t %no_sup: {:.2f}".format(current_reg_out.count(self.model.regulator.label2index.get("none", -1))/total*100)
            report_str += "\t %self_sup: {:.2f}".format(current_reg_out.count(self.model.regulator.label2index.get("self", -1))/total*100)
            report_str += "\t %weak_sup: {:.2f}".format(current_reg_out.count(self.model.regulator.label2index.get("weak", -1))/total*100)
            report_str += "\t %full_sup: {:.2f}".format(current_reg_out.count(self.model.regulator.label2index.get("full", -1))/total*100)
        # at the end add * and lr
        lr_str = ""
        if type(current_lr) is dict:
            for k, v in current_lr.items():
                lr_str += "\tLR-{}: {}".format(k, v)
        else:
            lr_str += "\tLR: {}".format(current_lr)
        report_str += "{}\t{}\n".format(lr_str,"*" if new_best else "")


        with open(self.valid_report_file, 'a') as opened_file:
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
    random.seed(seed)

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

        score, loss, ppl, sources, sources_raw, references, hypotheses, \
        hypotheses_raw, attention_scores  = validate_on_data(
            data=test_data, batch_size=trainer.valid_batch_size,
            eval_metric=trainer.eval_metric, level=trainer.level,
            max_output_length=trainer.max_output_length,
            model=model, use_cuda=trainer.use_cuda, criterion=None,
            beam_size=beam_size, beam_alpha=beam_alpha)
        
        if "trg" in test_data.fields:
            decoding_description = "Greedy decoding" if beam_size == 0 else \
                "Beam search decoding with beam size = {} and alpha = {}"\
                    .format(beam_size, beam_alpha)
            trainer.logger.info("{:4s}: {} {} [{}]".format(
                "Test data result", score, trainer.eval_metric,
                decoding_description))
        else:
            trainer.logger.info(
                "No references given for {}.{} -> no evaluation.".format(
                    cfg["data"]["test"],cfg["data"]["src"]))

        output_path_set = "{}/{}.{}".format(
            trainer.model_dir,"test",cfg["data"]["trg"])
        with open(output_path_set, mode="w", encoding="utf-8") as f:
            for h in hypotheses:
                f.write(h + "\n")
        trainer.logger.info("Test translations saved to: {}".format(
            output_path_set))

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Joey-NMT')
    parser.add_argument("config", default="configs/default.yaml", type=str,
                        help="Training configuration file (yaml).")
    args = parser.parse_args()
    train(cfg_file=args.config)
