import random
from collections import OrderedDict

from torchtext.datasets import UDPOS
from torchtext.vocab import GloVe
import numpy as np

from nm import seqdata


UDPOS_DATASETS = UDPOS()


def create_tag_dictionary():
    """Create the dictionary of POS tags for use such as 1-hot encoding."""
    udpos_tags = OrderedDict()
    for dataset in UDPOS_DATASETS:
        for words, _, tags in dataset:
            for tag in tags:
                if tag not in udpos_tags:
                    udpos_tags[tag] = len(udpos_tags)
    assert "." in udpos_tags   # required for ending sequences
    reverse_udpos_tags = dict()
    for tag, index in udpos_tags.items():
        reverse_udpos_tags[index] = tag
    return udpos_tags, reverse_udpos_tags


UDPOS_TAGS, REVERSE_UDPOS_TAGS = create_tag_dictionary()


def udpos_one_hot_vectors() -> np.ndarray:
    """Create 1-hot encoded vectors of the POS tags."""
    size = len(UDPOS_TAGS)
    vectors = np.zeros((size, size), dtype=np.float32)
    # assume that the tag order is consistent
    for i in range(vectors.shape[0]):
        vectors[i, i] = 1
    return vectors


GLOVE_840B_300 = GloVe()


class UDPOSSeq(seqdata.Sequence):
    """A sequence implementation for the POS sequence."""

    def __init__(self, sid: int, inputs: list, targets: list, target_vectors: np.ndarray):
        assert len(inputs) > 0
        assert len(inputs) == len(targets), "Inputs and targets differ in length"
        super().__init__()
        self.sid = sid
        self.inputs = list(inputs)
        self.targets = list(targets)
        # guarantee that a seq has an end symbol
        if self.inputs[-1] not in ["!", "?", "."]:
            self.inputs.append(".")
            self.targets.append(".")
        self.target_vectors = target_vectors

    def exhausted(self):
        return self.index >= len(self.inputs)

    def sizes(self) -> tuple[int, int]:
        glove_size = int(GLOVE_840B_300.dim)
        target_size = int(self.target_vectors.shape[1])
        return glove_size, target_size

    def types(self) -> tuple[np.dtype, np.dtype]:
        return np.float32, self.target_vectors.dtype

    def output_to_symbol(self, index: int):
        return REVERSE_UDPOS_TAGS[index]

    def transform_inputs(self, index: int, include_source: bool = False):
        assert -1 < index < len(self.inputs), "Input index out of range, {}".format(index)
        w = self.inputs[index]
        if include_source:
            return w, GLOVE_840B_300[w]
        return GLOVE_840B_300[w]

    def transform_targets(self, index: int, include_source: bool = False):
        assert 0 <= index < len(self.targets), "Target index out of range, {}".format(index)
        t = self.targets[index]
        if include_source:
            return t, self.target_vectors[UDPOS_TAGS[t], :].flatten()
        return self.target_vectors[UDPOS_TAGS[t], :].flatten()


class UDPOSSeqBatch(seqdata.SequenceBatch):
    """A sequence batch implementation for the POS sequence."""

    def __init__(self, dataset_name: str, batch_size: int, truncate: int = -1, allow_batch_size_of_one: bool = True):
        super().__init__(batch_size, allow_batch_size_of_one=allow_batch_size_of_one)
        assert dataset_name in ["train", "validation", "test"]
        self.dataset_name = dataset_name

        self.target_vectors = udpos_one_hot_vectors()

        if dataset_name == "train":
            udpos_dataset = UDPOS_DATASETS[0]
        elif dataset_name == "validation":
            udpos_dataset = UDPOS_DATASETS[1]
        else:
            udpos_dataset = UDPOS_DATASETS[2]
        self.dataset = []
        for words, _, tags in udpos_dataset:
            self.dataset.append(UDPOSSeq(len(self.dataset), words, tags, self.target_vectors))
        random.shuffle(self.dataset)
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

    def next_random_sequence(self) -> UDPOSSeq | None:
        self.current_sequence += 1
        return self.dataset[self.current_sequence] if self.current_sequence < len(self.dataset) else None
