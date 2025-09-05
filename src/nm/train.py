import logging
import random
import argparse
import os
import sys
import yaml
import shutil

import numpy as np
import torch
from torch import nn
from sklearn.metrics import precision_recall_fscore_support

from nm import seqdata
from nm import posdata
from nm import action_nn
from nm import mem_nn
from nm import logicdata
from nm import state
from nm import utils
from nm import wmt2014data


# NOTE: this evaluation will differ slightly from predict.py because there are a different number of sequence
# separating characters due to batch size differences.
def evaluate(config: dict, dataset: seqdata.SequenceBatch, action_model: nn.Module,
             use_memory: bool, mem_model: nn.Module) -> tuple[float, float, float, float, float, float]:
    """
    Evaluate a model on a given data set.

    :param config:  configuration info for evaluation
    :param dataset:  the data set
    :param action_model:  the action model
    :param use_memory:  true, if the memory should be used during evaluation
    :param mem_model:  the memory model
    :return:  micro and weighted precision, recall and F1
    """
    # assumes the model is on the right device
    dataset.restart(config["eval_batch_size"], allow_batch_size_of_one=True)
    history = state.StateHistory(config["history"])
    history.add(
        state.BatchState(config["eval_batch_size"], config["input_dim"], torch.float32,
                         config["memory_dim"], torch.float32, config["target_dim"], torch.float32,
                         None, device=config["device"])
    )

    action_model.eval()
    mem_model.eval()

    all_predictions = []
    all_targets = []
    with torch.no_grad():
        while True:
            batch = dataset.next()
            if batch is None:
                break
            utils.update_batch_history(config, batch, history)
            inputs = history.current().inputs()
            memories = history.current().memories()
            actions = action_model(inputs, memories)
            history.current().update_actions(actions)

            all_targets.extend(map(int, history.current().targets().argmax(1)))
            all_predictions.extend(map(int, history.current().actions().argmax(1)))

            if use_memory:
                history.add(
                    state.BatchState(config["eval_batch_size"], config["input_dim"], torch.float32,
                                     config["memory_dim"], torch.float32, config["target_dim"], torch.float32,
                                     None, device=config["device"])
                )
                history.current().subset = history.past(1).subset
                next_memories = mem_model(
                    history.past(1).inputs(), history.past(1).actions(), history.past(1).memories()
                )
                history.current().update_memories(next_memories)

    # TODO: make same changes to predict
    mp, mr, mf1, _ = precision_recall_fscore_support(all_targets, all_predictions, average='micro', zero_division=0)
    wp, wr, wf1, _ = precision_recall_fscore_support(all_targets, all_predictions, average='weighted', zero_division=0)
    return mp, mr, mf1, wp, wr, wf1


class Trainer:
    """
    Trains an action and memory network.
    """

    def __init__(self, train: seqdata.SequenceBatch, train_eval: seqdata.SequenceBatch,
                 validation: seqdata.SequenceBatch, action_model: nn.Module, mem_model: nn.Module):
        # attributes are not dependent on the training configuration
        self.train_data: seqdata.SequenceBatch = train
        self.train_eval_data: seqdata.SequenceBatch = train_eval
        self.validation_data: seqdata.SequenceBatch = validation
        self.action_model: nn.Module = action_model
        self.mem_model: nn.Module = mem_model

    def train(self, config: dict) -> None:
        """
        Train the class' action and memory models.

        :param config:  training configuration
        """
        action_model = self.action_model.to(config["device"])
        mem_model = self.mem_model.to(config["device"])
        action_loss_fn, mem_loss_fn = self.setup_loss(config)
        action_optimizer, action_lr_scheduler = self.setup_action_optimizer(config, action_model)
        mem_optimizer, mem_lr_scheduler = self.setup_memory_optimizer(config, mem_model)
        num_action_params = self.log_number_of_model_parameters("action", action_model)
        num_mem_params = self.log_number_of_model_parameters("memory", mem_model)
        logging.info("Total model parameters %i", num_action_params + num_mem_params)
        # prepare the experience replay history
        history = state.StateHistory(config["history"])
        history.reset()
        history.add(
            state.BatchState(config["batch_size"], config["input_dim"], torch.float32,
                             config["memory_dim"], torch.float32, config["target_dim"], torch.float32,
                             None, device=config["device"])
        )
        # calculate the mean F1 over a window to help notice long-term changes in performance
        smooth_f1_window_size = 7
        train_f1 = 0
        smooth_train_f1 = []
        validation_f1 = 0
        smooth_validation_f1 = []
        num_memory_trainings = 0

        for epoch in range(config["epochs"]):
            logging.info("Starting epoch %i", epoch + 1)

            # reset everything for the epoch
            self.train_data.restart(batch_size=config["batch_size"], allow_batch_size_of_one=False, is_training=True)
            if "wmt_curriculum" in config and config["wmt_curriculum"] is not None:
                # mirror curriculum learning in training data set
                self.train_eval_data.restart(config["eval_batch_size"], allow_batch_size_of_one=True, is_training=True)
            action_model.train()
            num_batches = 0
            epoch_action_loss = 0
            epoch_memory_loss = 0
            num_epoch_memory_trainings = 0

            while True:
                batch = self.train_data.next()
                if batch is None:
                    break
                utils.update_batch_history(config, batch, history)
                # apply the action model and retain output for training memory, and train action model
                actions = action_model(history.current().inputs(), history.current().memories())
                history.current().update_actions(actions)
                action_losses = action_loss_fn(actions, history.current().targets())
                action_losses = action_losses.reshape((action_losses.size(0), 1))
                history.current().update_action_losses(action_losses)
                if history.current().subset is not None:
                    action_losses = action_losses[history.current().subset, :]
                action_loss = action_losses.sum() / action_losses.size(0)
                epoch_action_loss += action_loss.item()
                action_optimizer.zero_grad()
                action_loss.backward()
                if "action_clipping" in config:
                    torch.nn.utils.clip_grad_norm_(action_model.parameters(), config["action_clipping"])
                action_optimizer.step()

                if num_batches > 0 and history.size > 1 and num_batches % (config["settle"] + history.size) == 0:
                    # train the memory model on the learning signal generated from the current predicted actions
                    mem_model.train()
                    for _ in range(config["mem_train_iterations"]):
                        epoch_memory_loss += self.train_memory(
                            config, history, mem_model, mem_loss_fn, mem_optimizer
                        )
                        num_memory_trainings += 1
                        num_epoch_memory_trainings += 1

                # predict next memory state.  first add the next state and then update the memories
                history.add(
                    state.BatchState(config["batch_size"], config["input_dim"], torch.float32,
                                     config["memory_dim"], torch.float32, config["target_dim"], torch.float32,
                                     None, device=config["device"])
                )
                if history.size > 1:
                    history.current().subset = history.past(1).subset
                    mem_model.eval()
                    next_memories = mem_model(
                        history.past(1).inputs(), history.past(1).actions(),history.past(1).memories()
                    )
                    history.current().update_memories(next_memories)

                num_batches += 1

            if action_lr_scheduler is not None:
                old_rate = action_lr_scheduler.get_last_lr()[0]
                action_lr_scheduler.step()
                new_rate = action_lr_scheduler.get_last_lr()[0]
                if old_rate != new_rate:
                    logging.info("Action learning rate set to: %f", new_rate)
            if mem_lr_scheduler is not None:
                old_rate = mem_lr_scheduler.get_last_lr()[0]
                mem_lr_scheduler.step()
                new_rate = mem_lr_scheduler.get_last_lr()[0]
                if old_rate != new_rate:
                    logging.info("Memory learning rate set to: %f", new_rate)

            if "eval_interval" in config and epoch % config["eval_interval"] != 0:
                continue

            logging.info("evaluating...")
            tmp, tmr, tmf1, twp, twr, twf1 = evaluate(
                config, self.train_eval_data, action_model, history.size > 1, mem_model
            )
            smooth_train_f1.append(twf1)
            vmp, vmr, vmf1, vwp, vwr, vwf1 = evaluate(
                config, self.validation_data, action_model, history.size > 1, mem_model
            )
            smooth_validation_f1.append(vwf1)
            if len(smooth_train_f1) > smooth_f1_window_size:
                smooth_train_f1.pop(0)
                smooth_validation_f1.pop(0)
            logging.info("number of memory trainings: %i", num_memory_trainings)
            logging.info("action loss: %f", epoch_action_loss / num_batches)
            logging.info(
                "memory loss: %f",
                epoch_memory_loss / num_epoch_memory_trainings if num_epoch_memory_trainings > 0 else -1
            )
            logging.info(
                "eval on train: micro p %f  r %f  f1 %f  weighted p %f  r %f  f1 %f  smooth-weighted-f1 %f",
                tmp, tmr, tmf1, twp, twr, twf1, np.mean(smooth_train_f1)
            )
            logging.info(
                "eval on validation: micro p %f  r %f  f1 %f  weighted p %f  r %f  f1 %f  smooth-weighted-f1 %f",
                vmp, vmr, vmf1, vwp, vwr, vwf1, np.mean(smooth_validation_f1)
            )
            save_type = ""
            if vwf1 > validation_f1:
                validation_f1 = vwf1
                save_type = "v"
            if twf1 > train_f1:
                train_f1 = twf1
                save_type = "t"
            if not save_type and "save_point" in config and epoch % config["save_point"] == 0:
                save_type = "sp"
            if save_type:
                logging.info("Saving models (%s)...", save_type)
                action_path = os.path.join(config["exp_dir"], "action_{}{}.pytorch".format(save_type, epoch + 1))
                torch.save(action_model, action_path)
                mem_path = os.path.join(config["exp_dir"], "mem_{}{}.pytorch".format(save_type, epoch + 1))
                torch.save(mem_model, mem_path)

    def train_memory(self, config: dict, history: state.StateHistory,
                     mem_model: nn.Module, mem_loss_fn, mem_optimizer) -> float:
        """
        Train the memory model.

        :param config:  training configuration
        :param history:  historical data to train from
        :param mem_model:  memory model to train
        :param mem_loss_fn:  memory loss function
        :param mem_optimizer:  memory optimizer
        :return: the average memory loss in this training step
        """
        assert history.ancient() is not None
        assert config["mem_num_samples"] < config["history"]

        inputs = []
        actions = []
        memories = []
        next_inputs = []
        next_actions = []
        next_memories = []
        action_losses = []

        pairs = []
        history_indices = list(range(1, history.size))
        for i in random.sample(history_indices, config["mem_num_samples"]):
            pairs.append([i, i - 1])

        for past, f_past in pairs:
            subset = history.past(f_past).subset
            inputs.append(history.past(past).inputs(override=subset).squeeze())
            if "mem_train_with_targets" in config and config["mem_train_with_targets"]:
                actions.append(history.past(past).targets(override=subset).squeeze())
            else:
                actions.append(history.past(past).actions(override=subset).squeeze())
            memories.append(history.past(past).memories(override=subset).squeeze())
            next_inputs.append(history.past(f_past).inputs())
            if "mem_train_with_targets" in config and config["mem_train_with_targets"]:
                next_actions.append(history.past(f_past).targets())
            else:
                next_actions.append(history.past(f_past).actions())
            next_memories.append(history.past(f_past).memories())
            action_losses.append(history.past(f_past).action_losses())

        inputs = torch.cat(inputs)
        actions = torch.cat(actions)
        memories = torch.cat(memories)
        next_inputs = torch.cat(next_inputs)
        next_actions = torch.cat(next_actions)
        next_memories = torch.cat(next_memories)
        action_losses = torch.cat(action_losses)

        policy_memories = mem_model(next_inputs, next_actions, next_memories)
        if "mem_l1" in config:
            # apply a notion of sparsity to the memory reward
            action_losses += config["mem_l1"] * torch.mean(torch.abs(policy_memories.clone().detach()))
        if "mem_dampen_losses" in config and config["mem_dampen_losses"] > 0:
            # keep numbers from getting ridiculous in a smoother manner than clamping
            action_losses = config["mem_dampen_losses"] * torch.log(1 + action_losses)
        if "mem_clamp" in config and config["mem_clamp"] > 0:
            # keep numbers from getting ridiculous
            action_losses = torch.clamp(action_losses, min=0, max=config["mem_clamp"])
        expected_memories = config["mem_gamma"] * policy_memories - action_losses

        actual_memories = mem_model(inputs, actions, memories)
        if torch.any(torch.isinf(actual_memories)) or torch.any(torch.isnan(actual_memories)):
            logging.warning("Memory model produced Infs or NaN when calculating actual memories.")
            Trainer.log_nan("inputs", inputs)
            Trainer.log_nan("actions", actions)
            Trainer.log_nan("mem_in", memories)
            Trainer.log_nan("mem_out", actual_memories)
            logging.info("Dimensions %s %s %s", inputs.shape, actions.shape, memories.shape)
            raise ValueError("Memory model produced Infs or NaN")

        mem_loss = mem_loss_fn(actual_memories, expected_memories)
        mem_optimizer.zero_grad()
        mem_loss.backward()
        if "mem_clipping" in config:
            torch.nn.utils.clip_grad_norm_(mem_model.parameters(), config["mem_clipping"])
        mem_optimizer.step()
        return mem_loss.item()

    @staticmethod
    def log_nan(tensor_name: str, t: torch.tensor) -> None:
        max_ = torch.max(t)
        min_ = torch.min(t)
        near_zero = None
        near_zero_abs = None
        has_zeroes = False
        for val in torch.flatten(t).tolist():
            if val == 0:
                has_zeroes = True
            elif near_zero_abs is None or abs(val) < near_zero_abs:
                near_zero_abs = abs(val)
                near_zero = val
        logging.warning(
            "Tensor '{}' values: max {} min {} has_zeroes {} near zero {}".format(
                tensor_name, max_, min_, has_zeroes, near_zero
            )
        )

    @staticmethod
    def log_number_of_model_parameters(name: str, model: nn.Module) -> int:
        total = 0
        for p in model.parameters():
            total += torch.numel(p)
        logging.info("Model '{}' has {} parameters".format(name, total))
        return total

    @staticmethod
    def setup_loss(config: dict) -> tuple:
        """Convert the training configuration to actual loss functions."""
        if config["loss_function"] == "cross_entropy":
            action_loss_fn = nn.CrossEntropyLoss(reduction="none")
        else:
            raise ValueError("Unknown action loss function {}".format(config["loss_function"]))
        return action_loss_fn, torch.nn.SmoothL1Loss()

    @staticmethod
    def setup_action_optimizer(config: dict, action_model: nn.Module):
        """Convert the training configuration to an optimizer, for the action network."""
        if config["action_optimizer"] == "adam":
            action_optimizer = torch.optim.Adam(
                action_model.parameters(),
                lr=config["action_learning_rate"]
            )
        elif config["action_optimizer"] == "sgd":
            action_optimizer = torch.optim.SGD(
                action_model.parameters(),
                lr=config["action_learning_rate"]
            )
        else:
            raise ValueError("Unknown action optimizer {}".format(config["optimizer"]))
        action_lr_scheduler = None
        if "action_lr_rate_decay" in config:
            milestones = config["action_lr_rate_decay"][0]
            if isinstance(milestones, int):
                milestones = [milestones]
            assert milestones[0] >= 1
            action_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                action_optimizer, milestones=milestones, gamma=config["action_lr_rate_decay"][1]
            )
            logging.info("Action learning rate will decay %s", config["action_lr_rate_decay"])
        return action_optimizer, action_lr_scheduler

    @staticmethod
    def setup_memory_optimizer(config: dict, mem_model: nn.Module) -> tuple:
        """Convert the training configuration to an optimizer, for the memory network."""
        if config["mem_optimizer"] == "adam":
            mem_optimizer = torch.optim.Adam(
                mem_model.parameters(),
                lr=config["mem_learning_rate"]
            )
        elif config["mem_optimizer"] == "sgd":
            mem_optimizer = torch.optim.SGD(
                mem_model.parameters(),
                lr=config["mem_learning_rate"]
            )
        elif config["mem_optimizer"] == "rmsprop":
            mem_optimizer = torch.optim.RMSprop(
                mem_model.parameters(),
                lr=config["mem_learning_rate"]
            )
        else:
            raise ValueError("Unknown mem optimizer {}".format(config["optimizer"]))
        mem_lr_scheduler = None
        if "mem_lr_rate_decay" in config:
            milestones = config["mem_lr_rate_decay"][0]
            if isinstance(milestones, int):
                milestones = [milestones]
            assert milestones[0] >= 1
            mem_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                mem_optimizer, milestones=milestones, gamma=config["mem_lr_rate_decay"][1]
            )
            logging.info("Memory learning rate will decay %s", config["mem_lr_rate_decay"])
        return mem_optimizer, mem_lr_scheduler


def run_experiment(config: dict) -> None:
    """Construct an experiment given the training configuration and start the experiment."""
    if "dampen_losses" in config:
        raise ValueError("config key 'dampen_losses' is deprecated.  use 'mem_dampen_losses' instead.")
    if "gamma" in config:
        raise ValueError("config key 'gamma' is deprecated.  use 'mem_gamma' instead.")

    # to make the experiments more reproducible
    random.seed(13)
    np.random.seed(13)
    torch.manual_seed(13)

    if config["dataset"] == "udpos":
        logging.info("Data set is udpos")
        train = posdata.UDPOSSeqBatch(
            "train",
            config["batch_size"],
            truncate=config["train_truncate_size"] if "train_truncate_size" in config else -1,
            allow_batch_size_of_one=False
        )
        logging.info("Data input and target dimensions: %s", train.sizes())
        train_eval = posdata.UDPOSSeqBatch(
            "train",
            config["eval_batch_size"],
            truncate=config["train_eval_truncate_size"],
            allow_batch_size_of_one=True
        )
        validation = posdata.UDPOSSeqBatch(
            "validation",
            config["eval_batch_size"],
            allow_batch_size_of_one=True
        )
    elif config["dataset"] == "logic":
        logging.info("Data set is logic")
        train = logicdata.LogicSeqBatch(
            "train",
            config["batch_size"],
            allow_batch_size_of_one=False
        )
        logging.info("Data input and target dimensions: %s", train.sizes())
        train_eval = logicdata.LogicSeqBatch(
            "train",
            config["eval_batch_size"],
            truncate=config["train_eval_truncate_size"],
            allow_batch_size_of_one=True
        )
        validation = logicdata.LogicSeqBatch(
            "validation",
            config["eval_batch_size"],
            allow_batch_size_of_one=True
        )
    elif config["dataset"] == "wmt2014-en" or config["dataset"] == "wmt2014-en-fr":
        logging.info("Data set is %s", config["dataset"])
        languages = "en" if config["dataset"] == "wmt2014-en" else "en_fr"
        add_chars_to_curriculum = False
        if ("wmt_add_chars_to_curriculum" in config and "wmt_curriculum" in config
                and config["wmt_curriculum"] is not None):
            add_chars_to_curriculum = config["wmt_add_chars_to_curriculum"]
            logging.info("Adding characters to curriculum")
        train = wmt2014data.WMT2014NewsBatch(
            languages,
            "train",
            config["batch_size"],
            truncate=config["wmt_train_truncate_size"] if "wmt_train_truncate_size" in config else -1,
            allow_batch_size_of_one=False,
            curriculum_learning=config["wmt_curriculum"] if "wmt_curriculum" in config else None,
            add_chars_to_first_curriculum=add_chars_to_curriculum
        )
        logging.info("Data input and target dimensions: %s", train.sizes())
        train_eval_curriculum_learning = None
        if "wmt_curriculum" in config and ("wmt_force_non_curriculum_train_eval" not in config or
                                           not config["wmt_force_non_curriculum_train_eval"]):
            train_eval_curriculum_learning = config["wmt_curriculum"]
        logging.info("Train eval 'curriculum_learning' attribute is set to {}.".format(train_eval_curriculum_learning))
        train_eval = wmt2014data.WMT2014NewsBatch(
            languages,
            "train",
            config["eval_batch_size"],
            truncate=config["train_eval_truncate_size"],
            allow_batch_size_of_one=True,
            curriculum_learning=train_eval_curriculum_learning,
            add_chars_to_first_curriculum=add_chars_to_curriculum
        )
        validation = wmt2014data.WMT2014NewsBatch(
            languages,
            "validation",
            config["eval_batch_size"],
            truncate=config["wmt_validation_truncate_size"] if "wmt_validation_truncate_size" in config else -1,
            allow_batch_size_of_one=True
        )
    else:
        raise ValueError("Unknown data set {}".format(config["dataset"]))

    if config["action_model"] == "flatdann":
        action_model = action_nn.FlatDANN(
            config["input_dim"],
            config["memory_dim"],
            config["target_dim"],
            use_swish=("use_swish" in config and config["use_swish"])
        )
    elif config["action_model"] == "flat2dann":
        action_model = action_nn.Flat2DANN(
            config["input_dim"],
            config["memory_dim"],
            config["target_dim"]
        )
    elif config["action_model"] == "embdann":
        action_model = action_nn.EmbDANN(
            config["input_dim"],
            config["memory_dim"],
            config["target_dim"],
            config["action_embedding_dim"]
        )
    elif config["action_model"] == "parallelActionNet":
        action_model = action_nn.ManyActionNN(
            config["input_dim"],
            config["action_embedding_dim"],
            config["memory_dim"],
            config["action_chunk_dim"],
            config["target_dim"],
            activation_fn_is_tanh=("action_fn_is_tanh" in config and config["action_fn_is_tanh"]),
            emb_activation_fn_is_tanh=("action_emb_fn_is_tanh" in config and config["action_emb_fn_is_tanh"])
        )
        logging.info("Action layer chunk dimension %i", action_model.chunk_dim)
        logging.info("Action layer properties %s", action_model.layer_properties)
    else:
        raise ValueError("Unknown action model {}".format(config["model"]))

    if config["mem_model"] == "flatmnn":
        mem_model = mem_nn.MemoryNN(
            config["input_dim"],
            config["memory_dim"],
            target_dim=config["target_dim"]
        )
    elif config["mem_model"] == "flatmnnNA":
        mem_model = mem_nn.MemoryNN(
            config["input_dim"],
            config["memory_dim"]
        )
    elif config["mem_model"] == "embmnn":
        mem_model = mem_nn.MemoryNN(
            config["input_dim"],
            config["memory_dim"],
            input_embedding_dim=config["mem_embedding_dim"],
            target_dim=config["target_dim"]
        )
    elif config["mem_model"] == "embmnnNA":
        mem_model = mem_nn.MemoryNN(
            config["input_dim"],
            config["memory_dim"],
            input_embedding_dim=config["mem_embedding_dim"]
        )
    elif config["mem_model"] == "embWLmnnNA":
        mem_model = mem_nn.MemoryNN(
            config["input_dim"],
            config["memory_dim"],
            input_embedding_dim=config["mem_embedding_dim"],
            intermediate_layers=config["mem_num_intermediate_layers"],
            intermediate_dim=config["mem_intermediate_dim"]
        )
    elif config["mem_model"] == "parallelMemNetNA":
        mem_model = mem_nn.ManyMemoryNN(
            config["mem_num_mem_net"],
            config["input_dim"],
            config["memory_dim"],
            input_embedding_dim=config["mem_embedding_dim"],
            intermediate_layers=config["mem_num_intermediate_layers"],
            intermediate_dim=config["mem_intermediate_dim"],
            funnel_to_zero_chunk=("mem_funnel_to_zero_chunk" in config and config["mem_funnel_to_zero_chunk"]),
            activation_fn_is_tanh=("mem_fn_is_tanh" in config and config["mem_fn_is_tanh"]),
            emb_activation_fn_is_tanh=("mem_emb_fn_is_tanh" in config and config["mem_emb_fn_is_tanh"])
        )
        logging.info("Memory layer chunk dimension %i", mem_model.memory_chunk_dim)
    else:
        raise ValueError("Unknown action model {}".format(config["model"]))

    trainer = Trainer(train, train_eval, validation, action_model, mem_model)
    trainer.train(config)


def setup_logging(config: dict):
    """Setup logging for the model training"""
    logger = logging.getLogger()
    if "debug_level" in config:
        if config["debug_level"] == "debug":
            logger.setLevel(logging.DEBUG)
        elif config["debug_level"] == "info":
            logger.setLevel(logging.INFO)
        elif config["debug_level"] == "warning":
            logger.setLevel(logging.WARNING)
        else:
            raise ValueError("Unknown debug level {}".format(config["debug_level"]))
    else:
        logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(levelname)s:  %(message)s", datefmt='%y-%m-%d %H:%M:%S')
    file_handler = logging.FileHandler(os.path.join(config["exp_dir"], 'experiment.log'))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    levels = {10: "debug", 20: "info", 30: "warn"}
    logging.info("Set log level to %s", levels[logger.level])
    logging.info("Config saved in config.yaml")
    logging.info(os.linesep + "%s" + os.linesep, config)
    logging.info("Experiment starting...")


def main():
    """Convert console arguments into a running experiment"""
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="experiment config file")
    parser.add_argument("expdir", help="experiment directory")
    args = parser.parse_args()

    assert os.path.exists(args.config), "Config {} does not exist".format(args.config)
    assert not os.path.exists(args.expdir), "Experiment dir already exists".format(args.expdir)

    with open(args.config, 'r') as config_stream:
        config_ = yaml.load(config_stream, yaml.CLoader)
        assert "exp_dir" not in config_, "Config includes reserved key 'exp_dir'"
        assert "device" not in config_, "Config includes reserved key 'device'"

    exp_dir = os.path.abspath(args.expdir)
    config_["exp_dir"] = exp_dir
    os.makedirs(exp_dir)

    config_["device"] = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    shutil.copy(args.config, os.path.join(config_["exp_dir"], 'config.yaml'))

    setup_logging(config_)
    run_experiment(config_)


if __name__ == "__main__":
    main()
