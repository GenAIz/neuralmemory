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
             use_memory: bool, mem_model: nn.Module) -> tuple[float, float, float]:
    """
    Evaluate a model on a given data set.

    :param config:  configuration info for evaluation
    :param dataset:  the data set
    :param action_model:  the action model
    :param use_memory:  true, if the memory should be used during evaluation
    :param mem_model:  the memory model
    :return:  precision, recall and F1
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
            history.current().update_actions(actions.clone().detach())

            all_targets.extend(map(int, history.current().targets().argmax(1)))
            all_predictions.extend(map(int, actions.argmax(1)))

            if use_memory:
                history.add(
                    state.BatchState(config["eval_batch_size"], config["input_dim"], torch.float32,
                                     config["memory_dim"], torch.float32, config["target_dim"], torch.float32,
                                     None, device=config["device"])
                )
                history.current().subset = history.past(1).subset
                next_memories = mem_model(inputs, actions.clone().detach(), memories)
                history.current().update_memories(next_memories)

    p, r, f1, _ = precision_recall_fscore_support(all_targets, all_predictions, average='weighted', zero_division=0)
    return p, r, f1


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
        action_optimizer = self.setup_action_optimizer(config, action_model)
        mem_optimizer, mem_lr_scheduler = self.setup_memory_optimizer(config, mem_model)
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
                inputs = history.current().inputs().clone()
                memories = history.current().memories().clone()
                # apply the action model and retain output for training memory, and train action model
                actions = action_model(inputs, memories)
                history.current().update_actions(actions.clone().detach())
                targets = history.current().targets()
                action_losses = action_loss_fn(actions, targets)
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
                    next_memories = mem_model(inputs, actions, memories)
                    history.current().update_memories(next_memories.clone().detach())

                num_batches += 1

            if mem_lr_scheduler is not None:
                mem_lr_scheduler.step()
                logging.info("Learning rate set to: %f", mem_lr_scheduler.get_last_lr()[0])

            logging.info("evaluating...")
            tp, tr, tf1 = evaluate(config, self.train_eval_data, action_model, history.size > 1, mem_model)
            smooth_train_f1.append(tf1)
            vp, vr, vf1 = evaluate(config, self.validation_data, action_model, history.size > 1, mem_model)
            smooth_validation_f1.append(vf1)
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
                "eval on train: p %f  r %f  f1 %f  smooth-f1 %f",
                tp, tr, tf1, np.mean(smooth_train_f1)
            )
            logging.info(
                "eval on validation: p %f  r %f  f1 %f  smooth-f1 %f",
                vp, vr, vf1, np.mean(smooth_validation_f1)
            )
            save_type = ""
            if vf1 > validation_f1:
                validation_f1 = vf1
                save_type = "v"
            if tf1 > train_f1:
                train_f1 = tf1
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
            inputs.append(history.past(past).inputs().clone().detach()[subset, :].squeeze())
            actions.append(history.past(past).actions().clone().detach()[subset, :].squeeze())
            memories.append(history.past(past).memories().clone().detach()[subset, :].squeeze())
            next_inputs.append(history.past(f_past).inputs().clone().detach())
            next_actions.append(history.past(f_past).actions().clone().detach())
            next_memories.append(history.past(f_past).memories().clone().detach())
            action_losses.append(history.past(f_past).action_losses().clone().detach())

        inputs = torch.cat(inputs)
        actions = torch.cat(actions)
        memories = torch.cat(memories)
        next_inputs = torch.cat(next_inputs)
        next_actions = torch.cat(next_actions)
        next_memories = torch.cat(next_memories)
        action_losses = torch.cat(action_losses)
        if "mem_clamp" in config and config["mem_clamp"] > 0:
            # keep numbers from getting ridiculous
            action_losses = torch.clamp(action_losses, min=0, max=config["mem_clamp"])

        policy_memories = mem_model(next_inputs, next_actions, next_memories)
        expected_memories = config["gamma"] * policy_memories - action_losses
        actual_memories = mem_model(inputs, actions, memories)
        if torch.any(torch.isnan(actual_memories)):
            # some annoying numerical instability :(
            logging.warning("Memory model produced (actual) memories with a NaN.")
            is_input = False
            if torch.any(torch.isnan(inputs)):
                is_input = True
                logging.warning("NaN discovered in mem model inputs.")
            if torch.any(torch.isnan(actions)):
                is_input = True
                logging.warning("NaN discovered in mem model actions.")
            if torch.any(torch.isnan(memories)):
                is_input = True
                logging.warning("NaN discovered in mem model memories.")
            if not is_input:
                logging.warning("No model inputs have NaN.")
            sys.exit(1)
        mem_loss = mem_loss_fn(actual_memories, expected_memories)
        mem_optimizer.zero_grad()
        mem_loss.backward()
        if "mem_clipping" in config:
            torch.nn.utils.clip_grad_norm_(mem_model.parameters(), config["mem_clipping"])
        mem_optimizer.step()
        return mem_loss.item()

    def setup_loss(self, config: dict) -> tuple:
        """Convert the training configuration to actual loss functions."""
        if config["loss_function"] == "cross_entropy":
            action_loss_fn = nn.CrossEntropyLoss(reduction="none")
        else:
            raise ValueError("Unknown action loss function {}".format(config["loss_function"]))
        return action_loss_fn, torch.nn.SmoothL1Loss()

    def setup_action_optimizer(self, config: dict, action_model: nn.Module):
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
        return action_optimizer

    def setup_memory_optimizer(self, config: dict, mem_model: nn.Module) -> tuple:
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
            assert config["mem_lr_rate_decay"][0] >= 1
            mem_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                mem_optimizer, milestones=[config["mem_lr_rate_decay"][0]], gamma=config["mem_lr_rate_decay"][1]
            )
            logging.info("Learning rate will decay")
        return mem_optimizer, mem_lr_scheduler


def run_experiment(config: dict) -> None:
    """Construct an experiment given the training configuration and start the experiment."""
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
            config["target_dim"]
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
    else:
        raise ValueError("Unknown action model {}".format(config["model"]))

    if config["mem_model"] == "flatmnn":
        mem_model = mem_nn.MemoryNN(
            config["input_dim"],
            config["target_dim"],
            config["memory_dim"]
        )
    elif config["mem_model"] == "flat2mnn":
        mem_model = mem_nn.Memory2NN(
            config["input_dim"],
            config["target_dim"],
            config["memory_dim"]
        )
    elif config["mem_model"] == "flat3mnn":
        mem_model = mem_nn.Memory3NN(
            config["input_dim"],
            config["target_dim"],
            config["memory_dim"]
        )
    elif config["mem_model"] == "embmnn":
        mem_model = mem_nn.MemEmbMNN(
            config["input_dim"],
            config["target_dim"],
            config["memory_dim"],
            config["mem_embedding_dim"]
        )
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
