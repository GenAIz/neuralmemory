import random
import os
import json

import numpy as np

from nm import seqdata

# data directory where the generated data should be cached
DATA_DIR = os.environ["NM_DATA_DIR"]

# symbols or vocabulary for the logic problem
SYMBOLS = {
    "a": np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
    "n": np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
    "d": np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
    "o": np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
    "r": np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32),
    "x": np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32),
    "0": np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=np.float32),
    "1": np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], dtype=np.float32),
    "=": np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], dtype=np.float32),
    ".": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], dtype=np.float32),
    "|": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
}
RSYMBOL = ["a", "n", "d", "o", "r", "x", "0", "1", "=", ".", "|"]


# and max number of operands
MIN_LENGTH = 2
MAX_LENGTH = 20


def generate_and() -> tuple[list[str], list[str]]:
    """
    Generates an 'and' sequence.
    :return: Input and targets for 'and'
    """
    sequence = ["a", "n", "d"]
    length = random.randint(MIN_LENGTH, MAX_LENGTH)
    is_false = random.randint(0, 1) == 0
    for i in range(length):
        sequence.append("1")
    if is_false:
        position_of_zero = random.randint(3, 3 + length - 1)
        sequence[position_of_zero] = "0"
    sequence.append("=")
    targets = ["."]
    targets.extend(sequence)
    targets[-1] = "0" if is_false else "1"
    sequence.append("|")
    return sequence, targets


def generate_or() -> tuple[list[str], list[str]]:
    """
    Generates an 'or' sequence.
    :return: Input and targets for 'or'
    """
    sequence = ["o", "r"]
    length = random.randint(MIN_LENGTH, MAX_LENGTH)
    is_true = random.randint(0, 1) == 1
    for i in range(length):
        sequence.append("0")
    if is_true:
        position_of_one = random.randint(2, 2 + length - 1)
        sequence[position_of_one] = "1"
    sequence.append("=")
    targets = ["."]
    targets.extend(sequence)
    targets[-1] = "1" if is_true else "0"
    sequence.append("|")
    return sequence, targets


def generate_xor() -> tuple[list[str], list[str]]:
    """
    Generates an 'xor' sequence.
    :return: Input and targets for 'xor'
    """
    sequence = ["x", "o", "r"]
    first = random.randint(0, 1)
    second = random.randint(0, 1)
    sequence.append(str(first))
    sequence.append(str(second))
    sequence.append("=")
    xor = first ^ second
    targets = ["."]
    targets.extend(sequence)
    targets[-1] = str(xor)
    sequence.append("|")
    return sequence, targets


def generate_data(quantity: int, validation_ratio: float = 0.1,
                  regen: bool = False) -> tuple[list[tuple[list[str], list[str]]], list[tuple[list[str], list[str]]]]:
    """
    Generates logic sequence data or loads it from cache if cached.

    :param quantity:  number of sequences to generate
    :param validation_ratio:  amount to be used for validation data set (the remainder is training)
    :param regen:  if true, force regeneration of the data.
    :return: tuple of train and validation sequence data, where each contains aligned inputs and targets
    """
    cache = os.path.join(DATA_DIR, "logic.dataset.json")
    if regen and os.path.exists(cache):
        with open(cache) as in_stream:
            datasets = json.load(in_stream)
            train = datasets["train"]
            validation = datasets["validation"]
    else:
        dataset = []
        for i in range(quantity):
            which = random.randint(0, 2)
            if which == 0:
                dataset.append(generate_and())
            elif which == 1:
                dataset.append(generate_or())
            else:
                dataset.append(generate_xor())
        stop = int(quantity * (1.0 - validation_ratio))
        assert 0 < stop < quantity - 1
        train = dataset[0:stop]
        validation = dataset[stop:]
        with open(cache, "w") as out_stream:
            json.dump({"train": train, "validation": validation}, out_stream)
    return train, validation


class LogicSeq(seqdata.Sequence):
    """A sequence implementation for the logic sequence."""

    def __init__(self, sid: int, inputs: list, targets: list):
        assert len(inputs) > 0
        assert len(inputs) == len(targets), "Inputs and targets differ in length"
        super().__init__()
        self.sid = sid
        self.inputs = list(inputs)
        self.targets = list(targets)

    def output_to_symbol(self, index: int):
        return RSYMBOL[index]

    def exhausted(self):
        return self.index >= len(self.inputs)

    def sizes(self) -> tuple[int, int]:
        return len(SYMBOLS), len(SYMBOLS)

    def types(self) -> tuple[np.dtype, np.dtype]:
        return np.float32, np.float32

    def transform_inputs(self, index: int, include_source: bool = False):
        assert -1 < index < len(self.inputs), "Input index out of range, {}".format(index)
        if include_source:
            return self.inputs[index], SYMBOLS[self.inputs[index]]
        return SYMBOLS[self.inputs[index]]

    def transform_targets(self, index: int, include_source: bool = False):
        assert 0 <= index < len(self.targets), "Target index out of range, {}".format(index)
        if include_source:
            return self.targets[index], SYMBOLS[self.targets[index]]
        return SYMBOLS[self.targets[index]]


class LogicSeqBatch(seqdata.SequenceBatch):
    """A sequence batch implementation for the logic sequence."""

    def __init__(self, dataset_name: str, batch_size: int, truncate: int = -1, allow_batch_size_of_one: bool = True):
        super().__init__(batch_size, allow_batch_size_of_one=allow_batch_size_of_one)
        assert dataset_name in ["train", "validation"]
        self.dataset_name = dataset_name
        train, validation = generate_data(10000, validation_ratio=0.05)
        if dataset_name == "train":
            data = train
        else:
            data = validation
        self.dataset: list[LogicSeq] = []
        for i in range(len(data)):
            self.dataset.append(LogicSeq(i, data[i][0], data[i][1]))
        if truncate > 0:
            self.dataset = self.dataset[0:truncate]
        self.current_sequence = -1
        self.sizes_ = self.dataset[0].sizes()
        self.dtypes = self.dataset[0].types()

    def restart(self, batch_size: int = -1, allow_batch_size_of_one: bool | None = None,
                is_training: bool = True) -> None:
        for seq in self.dataset:
            seq.restart()
        random.shuffle(self.dataset)
        self.current_sequence = -1
        super().restart(
            batch_size=batch_size, allow_batch_size_of_one=allow_batch_size_of_one, is_training=is_training
        )

    def next_random_sequence(self) -> LogicSeq | None:
        self.current_sequence += 1
        return self.dataset[self.current_sequence] if self.current_sequence < len(self.dataset) else None
