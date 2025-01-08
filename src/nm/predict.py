"""
Alpha version of a script that applies a trained model to data
and records some useful debugging information.
"""


import os
import csv
import argparse
import yaml

import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from nm import state
from nm import utils
from nm import seqdata
from nm import logicdata
from nm import posdata
from nm import wmt2014data


class Alignment:
    """
    Captures an alignment between inputs, predictions and targets.
    """

    def __init__(self):
        self.inputs = []
        self.targets = []
        self.actions = []

    def clear(self) -> None:
        self.inputs.clear()
        self.targets.clear()
        self.actions.clear()

    def __len__(self) -> int:
        return max(len(self.inputs), len(self.targets), len(self.actions))

    def __str__(self) -> str:
        return "I: {}{}T: {}{}A: {}{}".format(
            " ".join(self.inputs), os.linesep,
            " ".join(self.targets), os.linesep,
            " ".join(self.actions), os.linesep
        )


def predict(config: dict, dataset: seqdata.SequenceBatch,
            action_model: torch.nn.Module, mem_model: torch.nn.Module,
            prefix: str, verbose: bool = False) -> None:
    """
    Apply a model to a data set.

    :param config:  training config
    :param dataset:  data set to use
    :param action_model:  action model to apply
    :param mem_model:   memory model to apply
    :param prefix:  file path prefix for debug/info files
    :param verbose:  if true, dump to console
    """
    dataset.restart()
    history = state.StateHistory(config["history"])
    history.add(
        state.BatchState(config["eval_batch_size"], config["input_dim"], torch.float32,
                         config["memory_dim"], torch.float32, config["target_dim"], torch.float32,
                         None, device=config["device"])
    )
    y_true = []
    y_predicted = []
    alignment = Alignment()

    alignment_output_stream = None
    if verbose and prefix:
        alignment_path = prefix + "_align.txt"
        alignment_output_stream = open(alignment_path, "w")
    action_model.eval()
    mem_model.eval()
    with torch.no_grad():
        while True:
            batch = dataset.next(include_source=True)
            if batch is None:
                break
            group_ids, inputs_with_sources, targets_with_sources = batch
            if verbose:
                alignment.inputs.append(inputs_with_sources[0][0])
                alignment.targets.append(targets_with_sources[0][0])
            y_true.append(targets_with_sources[0][0])
            batch_without_sources = group_ids, inputs_with_sources[1], targets_with_sources[1]
            utils.update_batch_history(config, batch_without_sources, history)
            inputs = history.current().inputs()
            memories = history.current().memories()
            actions = action_model(inputs, memories)
            actions_as_numpy = torch.argmax(actions).cpu().detach().numpy()
            try:
                out_symbol = dataset.output_to_symbol(int(actions_as_numpy))
                y_predicted.append(out_symbol)
                if verbose:
                    alignment.actions.append(out_symbol)
            except NotImplementedError:
                y_predicted.append(actions_as_numpy)
                if verbose:
                    alignment.actions.append(actions_as_numpy)
            history.current().update_actions(actions.clone().detach())
            history.add(
                state.BatchState(config["eval_batch_size"], config["input_dim"], torch.float32,
                                 config["memory_dim"], torch.float32, config["target_dim"], torch.float32,
                                 None, device=config["device"])
            )

            history.current().subset = history.past(1).subset
            next_memories = mem_model(inputs, actions.clone().detach(), memories)
            history.current().update_memories(next_memories)
            if alignment_output_stream is not None and len(alignment) >= 50:
                alignment_output_stream.write(str(alignment))
                alignment_output_stream.write(os.linesep)
                alignment.clear()
    if alignment_output_stream:
        alignment_output_stream.write(str(alignment))
        alignment_output_stream.close()

    metrics_path = prefix + "_metrics.txt"
    metrics_output_stream = open(metrics_path, "w")

    labels = set()
    labels.update(y_true)
    labels.update(y_predicted)
    labels = list(labels)
    labels.sort()

    matrix = confusion_matrix(y_true, y_predicted, labels=labels)
    table = [["#"]]
    table[0].extend(labels)
    for i in range(matrix.shape[0]):
        row = [labels[i]]
        for j in range(matrix.shape[1]):
            row.append(int(matrix[i, j]))
        table.append(row)
    csv.writer(metrics_output_stream).writerows(table)

    p, r, f1, _ = precision_recall_fscore_support(y_true, y_predicted, average='weighted', zero_division=0)
    prf1_string = "Evaluation: precision {} recall {} f1 {}".format(p, r, f1)
    print(prf1_string)
    metrics_output_stream.write(os.linesep)
    metrics_output_stream.write(prf1_string)
    metrics_output_stream.write(os.linesep)
    metrics_output_stream.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action='store_true', help="output sequence alignment")
    parser.add_argument("config", help="model config")
    parser.add_argument("dataset", help="data set name")
    parser.add_argument("subset", help="data subset such as train")
    parser.add_argument("actionmodel", help="action model")
    parser.add_argument("memmodel", help="mem model")
    parser.add_argument("prefix", help="path prefix for output")
    args = parser.parse_args()

    with open(args.config, 'r') as config_stream:
        config_ = yaml.load(config_stream, yaml.CLoader)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    config_["device"] = device
    config_["eval_batch_size"] = 1

    if args.dataset == "udpos":
        dataset = posdata.UDPOSSeqBatch(
            args.subset, config_["eval_batch_size"], allow_batch_size_of_one=True
        )
    elif args.dataset == "logic":
        dataset = logicdata.LogicSeqBatch(
            args.subset, config_["eval_batch_size"], allow_batch_size_of_one=True
        )
    elif config_["dataset"] == "wmt2014-en" or config_["dataset"] == "wmt2014-en-fr":
        languages = "en" if config_["dataset"] == "wmt2014-en" else "en_fr"
        curriculum_learning = None
        add_chars_to_first_curriculum = False
        if "wmt_curriculum" in config_ and config_["wmt_curriculum"] is not None:
            training_size = 100000000    # big enough to include the entire data set
            if "wmt_train_truncate_size" in config_:
                training_size = config_["wmt_train_truncate_size"]
            curriculum_learning = [1, training_size]
            if "wmt_add_chars_to_curriculum" in config_:
                add_chars_to_first_curriculum = config_["wmt_add_chars_to_curriculum"]
            print("Dataset affected by curriculum learning. (Adding chars: {})".format(add_chars_to_first_curriculum))
        truncate = -1
        if args.subset == "train" and "wmt_train_truncate_size" in config_:
            truncate = config_["wmt_train_truncate_size"]
        elif args.subset == "validation" and "wmt_validation_truncate_size" in config_:
            truncate = config_["wmt_validation_truncate_size"]
        dataset = wmt2014data.WMT2014NewsBatch(
            languages, args.subset, config_["eval_batch_size"],
            truncate=truncate, allow_batch_size_of_one=True,
            curriculum_learning=curriculum_learning, add_chars_to_first_curriculum=add_chars_to_first_curriculum,
            shuffle_on_restart=False
        )
    else:
        raise Exception("bad data name")

    print("Loading models...")
    action_model = torch.load(args.actionmodel)
    mem_model = torch.load(args.memmodel)

    print("Predicting...")
    predict(config_, dataset, action_model, mem_model, args.prefix, verbose=args.verbose)


if __name__ == "__main__":
    main()
