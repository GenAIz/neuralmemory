import random
import os
from collections import OrderedDict
import pickle

import numpy as np

from nm import seqdata


DATA_DIR = os.environ["NM_DATA_DIR"]
ENG_DATA = os.path.join(DATA_DIR, "wmt2014-news/news-commentary-v9.fr-en.en")
FR_DATA = os.path.join(DATA_DIR, "wmt2014-news/news-commentary-v9.fr-en.fr")

EN_EN_CACHE_FILE_PAIRS = os.path.join(DATA_DIR, "wmt2014-news.en.pairs.pickle")
EN_EN_CACHE_FILE_CHAR_TO_VEC = os.path.join(DATA_DIR, "wmt2014-news.en.char_to_vec.pickle")
EN_EN_CACHE_FILE_INDEX_TO_CHAR = os.path.join(DATA_DIR, "wmt2014-news.en.index_to_char.pickle")

EN_FR_CACHE_FILE_PAIRS = os.path.join(DATA_DIR, "wmt2014-news.fr-en.pairs.pickle")
EN_FR_CACHE_FILE_CHAR_TO_VEC = os.path.join(DATA_DIR, "wmt2014-news.fr-en.char_to_vec.pickle")
EN_FR_CACHE_FILE_INDEX_TO_CHAR = os.path.join(DATA_DIR, "wmt2014-news.fr-en.index_to_char.pickle")

START = "\u0298"
END = "\u0444"
EMPTY = "\u02fd"
SEPARATOR = "\u043d"


def collect_chars(lines: list[str], chars: dict[str, int]) -> None:
    """
    Collect a vocabulary of characters, creating a mapping of character to count

    :param lines:  extract characters from the given lines
    :param chars:  map characters to counts
    """
    for i in range(len(lines)):
        for j in range(len(lines[i])):
            c = lines[i][j]
            if c not in chars:
                chars[c] = 0
            chars[c] += 1


def clean_lines(chars: dict[str, int], lines: list[str], threshold: int = 7) -> list[str]:
    """
    Remove certain 'garbage' from lines.

    :param chars:  character counts
    :param lines:  lines to clean
    :param threshold:  characters with counts equal to or below threshold are removed
    :return: clean lines
    """
    for i in range(len(lines)):
        lines[i] = lines[i].replace("&#160;", "")
        lines[i] = lines[i].replace("&nbsp;", "")
        lines[i] = lines[i].replace("&amp;", "&")
        lines[i] = lines[i].replace("&quot;", "\"")
        lines[i] = lines[i].replace("\ufffd", " ")
        lines[i] = lines[i].replace(START, " ")
        lines[i] = lines[i].replace(END, " ")
        lines[i] = lines[i].replace(EMPTY, " ")
        lines[i] = lines[i].replace(SEPARATOR, " ")
        for char, count in chars.items():
            if count <= threshold or 19977 <= ord(char) <= 40065:
                lines[i] = lines[i].replace(char, " ")
        lines[i] = lines[i].strip()
        if lines[i]:
            tokens = lines[i].split()
            lines[i] = " ".join(tokens)
    return lines


def prepare_sentences(lang1: list[str], lang2: list[str]) -> list[tuple[str, str]]:
    """
    Create input/target pairs from sentence/sequence aligned languages.

    :param lang1:  list of string that aligns with lang2
    :param lang2:  list of string that aligns with lang1
    :return:  list of input/target pairs
    """
    assert len(lang1) == len(lang2)
    pairs: set[tuple[str, str]] = set()
    for i in range(len(lang1)):
        s1 = lang1[i]
        s2 = lang2[i]
        if not s1 or not s2:
            # skip empty sentences
            continue
        inputs = "{}{}{}{}".format(START, s1, SEPARATOR, EMPTY * (len(s2) + 1))
        s1_fill = incremental_fill(len(s1))
        targets = "{}{}{}{}{}".format(START, s1_fill, SEPARATOR, s2, END)
        pairs.add((inputs, targets))
        if s1 is not s2:
            inputs = "{}{}{}{}".format(START, s2, SEPARATOR, EMPTY * (len(s1) + 1))
            s2_fill = incremental_fill(len(s2))
            targets = "{}{}{}{}{}".format(START, s2_fill, SEPARATOR, s1, END)
            pairs.add((inputs, targets))
    pairs: list[tuple[str, str]] = list(pairs)
    return pairs


def incremental_fill(quantity: int) -> str:
    """
    Create string with a repeating pattern of 0123456789 of length quantity

    :param quantity: size of string
    :return:  a string with the repeating digits
    """
    fill: list[str] = []
    for i in range(quantity):
        fill.append(str(i % 10))
    return "".join(fill)


def print_chars(chars: dict[str, int]) -> None:
    keys = list(chars.keys())
    keys.sort(key=lambda x: ord(x))
    for ck in keys:
        print(ck, chars[ck], hex(ord(ck)), ord(ck))


def is_cached(pairs_path: str, char_to_vec_path: str, index_to_char_path: str) -> bool:
    """Return true if all files are cached on disk"""
    return os.path.isfile(pairs_path) and os.path.isfile(char_to_vec_path) and os.path.isfile(index_to_char_path)


def cache_data(pairs_path: str, pairs: list[tuple[str, str]],
               char_to_vec_path: str, char_to_1hot: dict[str, np.ndarray],
               index_to_char_path: str, index_to_char: dict[int, str]) -> None:
    """Save all data to disk"""
    with open(pairs_path, 'wb') as out_stream:
        pickle.dump(pairs, out_stream, protocol=pickle.HIGHEST_PROTOCOL)
    with open(char_to_vec_path, 'wb') as out_stream:
        pickle.dump(char_to_1hot, out_stream, protocol=pickle.HIGHEST_PROTOCOL)
    with open(index_to_char_path, 'wb') as out_stream:
        pickle.dump(index_to_char, out_stream, protocol=pickle.HIGHEST_PROTOCOL)


def fetch_cached_data(pairs_path: str, char_to_vec_path: str,
                      index_to_char_path: str) -> tuple[list[tuple[str, str]], dict[str, np.ndarray], dict[int, str]]:
    """Load all data from disk returning pairs, char_to_1hot, index_to_char data structures"""
    with open(pairs_path, 'rb') as in_stream:
        pairs: list[tuple[str, str]] = pickle.load(in_stream)
    with open(char_to_vec_path, 'rb') as in_stream:
        char_to_1hot: dict[str, np.ndarray] = pickle.load(in_stream)
    with open(index_to_char_path, 'rb') as in_stream:
        index_to_char: dict[int, str] = pickle.load(in_stream)
    return pairs, char_to_1hot, index_to_char


def character_pairs(characters) -> list[tuple[str, str]]:
    """Create artificial input-target sentences of 1 character"""
    ins = []
    outs = []
    characters = list(characters)
    characters.sort()
    for char in characters:
        if char in {START, END, EMPTY, SEPARATOR}:
            continue
        ins.append(char)
        outs.append(char)
    return prepare_sentences(ins, outs)


def construct_en_en_pairs() -> tuple[list[tuple[str, str]], dict[str, np.ndarray], dict[int, str]]:
    """
    Create English sentence memorization pairs/data.

    :return: pairs, char to 1hot mapping, index to char mapping
    """
    if is_cached(EN_EN_CACHE_FILE_PAIRS, EN_EN_CACHE_FILE_CHAR_TO_VEC, EN_EN_CACHE_FILE_INDEX_TO_CHAR):
        pairs, char_to_1hot, index_to_char = fetch_cached_data(
            EN_EN_CACHE_FILE_PAIRS, EN_EN_CACHE_FILE_CHAR_TO_VEC, EN_EN_CACHE_FILE_INDEX_TO_CHAR
        )
    else:
        with open(ENG_DATA, newline="\n") as in_stream:
            eng = in_stream.readlines()
        chars: dict[str, int] = dict()
        collect_chars(eng, chars)
        eng = clean_lines(chars, eng)
        chars.clear()
        collect_chars(eng, chars)
        assert START not in chars
        assert END not in chars
        assert EMPTY not in chars
        assert SEPARATOR not in chars
        chars: list = list(chars.keys())
        chars.append(START)
        chars.append(END)
        chars.append(EMPTY)
        chars.append(SEPARATOR)
        chars.sort()
        # map char to 1-hot vector
        # map index to char
        char_to_1hot = OrderedDict()
        index_to_char = OrderedDict()
        for i in range(len(chars)):
            index_to_char[i] = chars[i]
            char_to_1hot[chars[i]] = np.zeros(len(chars), dtype=np.float32)
            char_to_1hot[chars[i]][i] = 1.
        pairs = prepare_sentences(eng, eng)
        random.Random(874).shuffle(pairs)
        cache_data(
            EN_EN_CACHE_FILE_PAIRS, pairs,
            EN_EN_CACHE_FILE_CHAR_TO_VEC, char_to_1hot,
            EN_EN_CACHE_FILE_INDEX_TO_CHAR, index_to_char
        )
    return pairs, char_to_1hot, index_to_char


EN_EN_PAIRS, EN_EN_CHAR_TO_VECTOR, EN_EN_INDEX_TO_CHAR = construct_en_en_pairs()
EN_EN_CHAR_PAIRS = character_pairs(EN_EN_CHAR_TO_VECTOR.keys())


def construct_en_fr_pairs() -> tuple[list[tuple[str, str]], dict[str, np.ndarray], dict[int, str]]:
    """
    Create English-French sentence translation pairs/data.

    :return: pairs, char to 1hot mapping, index to char mapping
    """
    if is_cached(EN_FR_CACHE_FILE_PAIRS, EN_FR_CACHE_FILE_CHAR_TO_VEC, EN_FR_CACHE_FILE_INDEX_TO_CHAR):
        pairs, char_to_1hot, index_to_char = fetch_cached_data(
            EN_FR_CACHE_FILE_PAIRS, EN_FR_CACHE_FILE_CHAR_TO_VEC, EN_FR_CACHE_FILE_INDEX_TO_CHAR
        )
    else:
        with open(ENG_DATA, newline="\n") as in_stream:
            eng = in_stream.readlines()
        with open(FR_DATA, newline="\n") as in_stream:
            fr = in_stream.readlines()
        assert len(eng) == len(fr)
        chars: dict[str, int] = dict()
        collect_chars(eng, chars)
        collect_chars(fr, chars)
        eng = clean_lines(chars, eng)
        fr = clean_lines(chars, fr)
        chars.clear()
        collect_chars(eng, chars)
        collect_chars(fr, chars)
        assert START not in chars
        assert END not in chars
        assert EMPTY not in chars
        assert SEPARATOR not in chars
        chars: list = list(chars.keys())
        chars.append(START)
        chars.append(END)
        chars.append(EMPTY)
        chars.append(SEPARATOR)
        chars.sort()
        # map char to 1-hot vector
        # map index to char
        char_to_1hot = OrderedDict()
        index_to_char = OrderedDict()
        for i in range(len(chars)):
            index_to_char[i] = chars[i]
            char_to_1hot[chars[i]] = np.zeros(len(chars), dtype=np.float32)
            char_to_1hot[chars[i]][i] = 1.
        pairs = prepare_sentences(eng, fr)
        random.Random(874).shuffle(pairs)
        cache_data(
            EN_FR_CACHE_FILE_PAIRS, pairs,
            EN_FR_CACHE_FILE_CHAR_TO_VEC, char_to_1hot,
            EN_FR_CACHE_FILE_INDEX_TO_CHAR, index_to_char
        )
    return pairs, char_to_1hot, index_to_char


EN_FR_PAIRS, EN_FR_CHAR_TO_VECTOR, EN_FR_INDEX_TO_CHAR = construct_en_fr_pairs()
EN_FR_CHAR_PAIRS = character_pairs(EN_FR_CHAR_TO_VECTOR.keys())


class WMT2014NewsSeq(seqdata.Sequence):
    """A sequence implementation for a sentence (memorization or translation)."""

    def __init__(self, sid: str | int, inputs: str, targets: str,
                 char_to_vector: dict[str, np.ndarray], index_to_char: dict[int, str]):
        assert len(inputs) > 0
        assert len(inputs) == len(targets), "Inputs and targets differ in length"
        super().__init__()
        self.sid: str | int = sid
        self.inputs: str = inputs
        self.targets: str = targets
        self.char_to_vector: dict[str, np.ndarray] = char_to_vector
        self.index_to_char: dict[int, str] = index_to_char

    def exhausted(self):
        return self.index >= len(self.inputs)

    def sizes(self) -> tuple[int, int]:
        size = int(self.char_to_vector[self.index_to_char[0]].shape[0])
        return size, size

    def types(self) -> tuple[np.dtype, np.dtype]:
        dtype = self.char_to_vector[self.index_to_char[0]].dtype
        return dtype, dtype

    def output_to_symbol(self, index: int):
        return self.index_to_char[index]

    def transform_inputs(self, index: int, include_source: bool = False):
        assert -1 < index < len(self.inputs), "Input index out of range, {}".format(index)
        w = self.inputs[index]
        if include_source:
            return w, self.char_to_vector[w]
        return self.char_to_vector[w]

    def transform_targets(self, index: int, include_source: bool = False):
        assert 0 <= index < len(self.targets), "Target index out of range, {}".format(index)
        t = self.targets[index]
        if include_source:
            return t, self.char_to_vector[t]
        return self.char_to_vector[t]


class WMT2014NewsBatch(seqdata.SequenceBatch):
    """A sequence batch implementation for a sentence (memorization or translation)."""

    def __init__(self, languages: str, dataset_name: str, batch_size: int,
                 truncate: int = -1, allow_batch_size_of_one: bool = True,
                 curriculum_learning: None | tuple[int, int] = None, add_chars_to_first_curriculum: bool = False,
                 shuffle_on_restart: bool = True):
        super().__init__(batch_size, allow_batch_size_of_one=allow_batch_size_of_one)
        self.languages = languages
        PAIRS, CHAR_TO_VECTOR, INDEX_TO_CHAR, CHAR_PAIRS = self.dataset_constants()
        assert dataset_name in ["train", "validation"]
        self.dataset_name = dataset_name
        self.shuffle_on_restart: bool = shuffle_on_restart
        self.curriculums = None
        self.milestone = 0
        self.training_restarts = 0
        split = int(0.95 * len(PAIRS))
        if dataset_name == "train":
            data = PAIRS[:split]
            if curriculum_learning is None:
                if truncate > 0:
                    data = data[0:truncate]
            else:
                self.milestone, chunk_size = curriculum_learning
                assert self.milestone > 0 and chunk_size > 0
                data = list(data)
                data.sort(key=lambda p: len(p[0]))
                if truncate > 0:
                    data = data[0:truncate]
                self.curriculums = []
                for i in range(len(data)):
                    # note that this method of chunking the data results in a first curriculum that has
                    # 1 more item than the others
                    if i % chunk_size == 0:
                        self.curriculums.append([])
                    self.curriculums[-1].append(data[i])
                data = list(self.curriculums[0])
                if add_chars_to_first_curriculum:
                    data.extend(CHAR_PAIRS)
        else:
            data = PAIRS[split:]
            if truncate > 0:
                data = data[0:truncate]
        self.dataset: list[WMT2014NewsSeq] = []
        for i in range(len(data)):
            self.dataset.append(WMT2014NewsSeq(i, data[i][0], data[i][1], CHAR_TO_VECTOR, INDEX_TO_CHAR))
        self.current_sequence = -1
        self.sizes_ = self.dataset[0].sizes()
        self.dtypes = self.dataset[0].types()

    def dataset_constants(self):
        if self.languages == "en":
            PAIRS = EN_EN_PAIRS
            CHAR_TO_VECTOR = EN_EN_CHAR_TO_VECTOR
            INDEX_TO_CHAR = EN_EN_INDEX_TO_CHAR
            CHAR_PAIRS = EN_EN_CHAR_PAIRS
        elif self.languages == "en_fr":
            PAIRS = EN_FR_PAIRS
            CHAR_TO_VECTOR = EN_FR_CHAR_TO_VECTOR
            INDEX_TO_CHAR = EN_FR_INDEX_TO_CHAR
            CHAR_PAIRS = EN_FR_CHAR_PAIRS
        else:
            raise ValueError("Unknown language pairs data set")
        return PAIRS, CHAR_TO_VECTOR, INDEX_TO_CHAR, CHAR_PAIRS

    def restart(self, batch_size: int = -1, allow_batch_size_of_one: bool | None = None,
                is_training: bool = True) -> None:
        for seq in self.dataset:
            seq.restart()
        if self.curriculums and self.training_restarts != 0 and self.training_restarts % self.milestone == 0:
            _, CHAR_TO_VECTOR, INDEX_TO_CHAR, _ = self.dataset_constants()
            index = int(self.training_restarts / self.milestone)
            if index < len(self.curriculums):
                for i in range(len(self.curriculums[index])):
                    self.dataset.append(WMT2014NewsSeq(
                        i, self.curriculums[index][i][0], self.curriculums[index][i][1], CHAR_TO_VECTOR, INDEX_TO_CHAR
                    ))
        if is_training:
            self.training_restarts += 1
        if self.shuffle_on_restart:
            random.shuffle(self.dataset)
        self.current_sequence = -1
        super().restart(
            batch_size=batch_size, allow_batch_size_of_one=allow_batch_size_of_one, is_training=is_training
        )

    def next_random_sequence(self) -> WMT2014NewsSeq | None:
        self.current_sequence += 1
        return self.dataset[self.current_sequence] if self.current_sequence < len(self.dataset) else None
