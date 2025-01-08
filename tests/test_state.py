import unittest

import numpy as np
import torch

from nm import state


class TestEtc(unittest.TestCase):

    def test_zeros(self):
        v = state._zeros(3, np.int32)
        self.assertEqual(v.shape[0], 3)
        self.assertEqual(v.dtype, np.int32)
        v = state._zeros(3, torch.int8)
        self.assertEqual(v.shape[0], 3)
        self.assertEqual(v.dtype, torch.int8)

    def test_zeros_like(self):
        v = state._zeros_like(state._zeros(3, np.int32))
        self.assertEqual(v.shape[0], 3)
        self.assertEqual(v.dtype, np.int32)
        v = state._zeros_like(state._zeros(3, torch.int8))
        self.assertEqual(v.shape[0], 3)
        self.assertEqual(v.dtype, torch.int8)

    def test_dim1(self):
        v = [1,2,3]
        self.assertEqual(state._dim0(v), 3)
        v = np.zeros(4)
        self.assertEqual(state._dim0(v), 4)
        self.assertEqual(state._dim0(np.zeros(1)[0]), 0)
        v = torch.from_numpy(np.zeros(5))
        self.assertEqual(state._dim0(v), 5)
        self.assertEqual(state._dim0(torch.zeros(1)[0]), 0)


class TestState(unittest.TestCase):

    def test_state(self):
        s = state.State(2, np.int8, 3, np.int16, 4, np.int32)
        self.assertEqual(s.input.shape[0], 2)
        self.assertEqual(s.input.dtype, np.int8)
        self.assertEqual(s.memory.shape[0], 3)
        self.assertEqual(s.memory.dtype, np.int16)
        self.assertEqual(s.target.shape[0], 4)
        self.assertEqual(s.target.dtype, np.int32)
        self.assertEqual(s.action_loss, 0)

        s.input[:] = 1
        s.memory[:] = 1
        s.target[:] = 1
        s.action_loss = 1

        s.reset()
        self.assertEqual(np.sum(s.input), 0)
        self.assertEqual(np.sum(s.memory), 0)
        self.assertEqual(np.sum(s.target), 0)
        self.assertEqual(s.action_loss, 0)


class TestBatchState(unittest.TestCase):

    def test_update(self):
        batch = state.BatchState(
            7, 2, np.int8, 3, np.int16, 4, np.int32, None
        )
        batch.update_inputs(np.ones((7, 2), np.int8))
        self.assertNotEqual(np.sum(batch.states[0].input), 0)
        self.assertNotEqual(np.sum(batch.states[1].input), 0)
        self.assertNotEqual(np.sum(batch.states[2].input), 0)
        self.assertNotEqual(np.sum(batch.states[3].input), 0)
        self.assertNotEqual(np.sum(batch.states[4].input), 0)
        self.assertNotEqual(np.sum(batch.states[5].input), 0)
        self.assertNotEqual(np.sum(batch.states[6].input), 0)

    def test_update_subset(self):
        batch = state.BatchState(
            7, 2, np.int8, 3, np.int16, 4, np.int32, [1, 2, 6]
        )
        batch.update_inputs(np.ones((3, 2), np.int8))
        self.assertEqual(np.sum(batch.states[0].input), 0)
        self.assertNotEqual(np.sum(batch.states[1].input), 0)
        self.assertNotEqual(np.sum(batch.states[2].input), 0)
        self.assertEqual(np.sum(batch.states[3].input), 0)
        self.assertEqual(np.sum(batch.states[4].input), 0)
        self.assertEqual(np.sum(batch.states[5].input), 0)
        self.assertNotEqual(np.sum(batch.states[6].input), 0)

    def test_reconstruct(self):
        batch = state.BatchState(
            7, 2, np.int8, 3, np.int16, 4, np.int32, None
        )
        times = batch.inputs()
        self.assertEqual(times.shape, (7, 2))
        self.assertEqual(times.dtype, np.int8)
        x = np.random.randint(1, 4, size=(7, 2), dtype=np.int8)
        batch.update_inputs(x)
        times = batch.inputs()
        self.assertTrue(np.all(times == x))

    def test_reconstruct_subset(self):
        batch = state.BatchState(
            7, 2, np.int8, 3, np.int16, 4, np.int32, [1, 2, 6]
        )
        times = batch.inputs()
        self.assertEqual(times.shape, (3, 2))
        self.assertEqual(times.dtype, np.int8)
        x = np.random.randint(1, 4, size=(3, 2), dtype=np.int8)
        batch.update_inputs(x)
        times = batch.inputs()
        self.assertTrue(np.all(times == x))

    def test_action_losses(self):
        batch = state.BatchState(7, 2, np.int8, 3, np.int16, 4, np.int32, None)
        losses = np.ones((7, 1), dtype=np.int32)
        batch.update_action_losses(losses)
        self.assertEqual(batch.action_losses().shape, (7, 1))
        self.assertEqual(batch.action_losses().sum(), 7)
        losses = torch.zeros((7, 1), dtype=torch.int32)
        batch.update_action_losses(losses)
        self.assertEqual(batch.action_losses().size(), (7, 1))
        self.assertEqual(batch.action_losses().sum(), 0)


class TestHistory(unittest.TestCase):

    def test_history(self):
        h = state.StateHistory(3, False)
        self.assertIsNone(h.current())
        self.assertIsNone(h.past(0))
        self.assertIsNone(h.past(1))
        self.assertIsNone(h.past(2))
        s = state.State(2, np.int8, 3, np.int16, 4, np.int32)
        h.add(s)
        self.assertIsNotNone(h.current())
        self.assertIsNotNone(h.past(0))
        self.assertIsNone(h.past(1))
        self.assertIsNone(h.past(2))
        self.assertEqual(len(h.history), 3)
        self.assertTrue(h.has(0))
        self.assertTrue(h.has(1))
        self.assertTrue(h.has(2))
        self.assertFalse(h.has(3))
        self.assertFalse(h.has(4))


if __name__ == '__main__':
    unittest.main()
