import unittest

import numpy as np

from nm import seqdata


class ExampleSequence(seqdata.Sequence):

    def __init__(self, start: int, end: int):
        super().__init__()
        assert start < end, "start must be less than end, but is not"
        self.start = start
        self.end = end

    def transform_inputs(self, index: int, include_source: bool = False):
        input_ = np.zeros(self.sizes()[0], dtype=self.types()[0])
        input_[0] = self.start + index
        if include_source:
            return self.start + index, input_
        return input_

    def transform_targets(self, index: int, include_source: bool = False):
        target = np.zeros(self.sizes()[1], dtype=self.types()[1])
        target[1] = self.start + index
        if include_source:
            return self.start + index, target
        return target

    def exhausted(self) -> bool:
        return self.start + self.index > self.end

    def sizes(self) -> tuple[int, ...]:
        return 3, 2

    def types(self) -> tuple[np.dtype, np.dtype]:
        return np.int32, np.int32


class TestSequence(unittest.TestCase):

    def test_sequence(self):
        es = ExampleSequence(3, 7)
        inputs = es.transform_inputs(0)
        self.assertTrue(np.all(inputs == np.array([3, 0, 0])))
        targets = es.transform_targets(0)
        self.assertTrue(np.all(targets == np.array([0, 3])))
        inputs = []
        es.restart()
        while not es.exhausted():
            inputs.append(es.next())
        self.assertEqual(len(inputs), 5)
        inputs.clear()
        es.restart()
        while not es.exhausted():
            inputs.append(es.next())
        self.assertEqual(len(inputs), 5)


class ExampleSB(seqdata.SequenceBatch):

    def __init__(self, allow_batch_size_of_one: bool = True):
        super().__init__(2, allow_batch_size_of_one=allow_batch_size_of_one)
        self.dataset = []
        self.dataset.append(ExampleSequence(1, 2))
        self.dataset.append(ExampleSequence(3, 7))
        self.dataset.append(ExampleSequence(0, 1))
        self.current_sequence = -1
        self.sizes_ = self.dataset[0].sizes()
        self.dtypes = self.dataset[0].types()

    def restart(self) -> None:
        for seq in self.dataset:
            seq.restart()
        self.current_sequence = -1
        super().restart()

    def next_random_sequence(self) -> ExampleSequence | None:
        self.current_sequence += 1
        return self.dataset[self.current_sequence] if self.current_sequence < len(self.dataset) else None


class TestBatch(unittest.TestCase):

    def test_batch(self):
        sb = ExampleSB()
        self.assertEqual(sb.sizes(), (3, 2))
        self.assertEqual(sb.types(), (np.int32, np.int32))
        sb.restart()
        self.assertEqual(sb.sizes(), (3, 2))
        self.assertEqual(sb.types(), (np.int32, np.int32))
        self.assertFalse(sb.exhausted())

        group_ids, inputs, targets = sb.next()
        self.assertIsNone(group_ids)
        ins = np.array([[1, 0, 0], [3, 0, 0]])
        outs = np.array([[0, 1], [0, 3]])
        self.assertTrue(np.all(inputs == ins))
        self.assertTrue(np.all(targets == outs))

        group_ids, inputs, targets = sb.next()
        self.assertIsNone(group_ids)
        ins = np.array([[2, 0, 0], [4, 0, 0]])
        outs = np.array([[0, 2], [0, 4]])
        self.assertTrue(np.all(inputs == ins))
        self.assertTrue(np.all(targets == outs))

        group_ids, inputs, targets = sb.next()
        self.assertIsNone(group_ids)
        ins = np.array([[0, 0, 0], [5, 0, 0]])
        outs = np.array([[0, 0], [0, 5]])
        self.assertTrue(np.all(inputs == ins))
        self.assertTrue(np.all(targets == outs))

        group_ids, inputs, targets = sb.next()
        self.assertIsNone(group_ids)
        ins = np.array([[1, 0, 0], [6, 0, 0]])
        outs = np.array([[0, 1], [0, 6]])
        self.assertTrue(np.all(inputs == ins))
        self.assertTrue(np.all(targets == outs))

        group_ids, inputs, targets = sb.next()
        self.assertEqual(group_ids, [1])
        ins = np.array([[7, 0, 0]])
        outs = np.array([[0, 7]])
        self.assertTrue(np.all(inputs == ins))
        self.assertTrue(np.all(targets == outs))

        self.assertIsNone(sb.next())
        self.assertTrue(sb.exhausted())

    def test_batch_no_single(self):
        sb = ExampleSB(allow_batch_size_of_one=False)
        sb.restart()
        batches = []
        while not sb.exhausted():
            batches.append(sb.next())
        self.assertEqual(len(batches), 5)   # including None


if __name__ == '__main__':
    unittest.main()
