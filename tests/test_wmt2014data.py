import unittest

from nm.wmt2014data import WMT2014NewsBatch, EN_EN_CHAR_PAIRS, EN_FR_CHAR_PAIRS


class TestWMT2014Data(unittest.TestCase):

    def test_wmtdata_en(self):
        w = WMT2014NewsBatch("en", "train", 50, allow_batch_size_of_one=False, curriculum_learning=(1, 500))
        size = 0
        for j in range(2):
            for i in range(50):
                next_size = len(w.dataset[j * 50 + i].targets)
                self.assertLessEqual(size, next_size)
        self.assertEqual(len(EN_EN_CHAR_PAIRS), 167)

    def test_wmtdata_enfr(self):
        size = 0
        w = WMT2014NewsBatch("en_fr", "train", 50, allow_batch_size_of_one=False, curriculum_learning=(1, 500))
        for j in range(2):
            for i in range(50):
                next_size = len(w.dataset[j * 50 + i].targets)
                self.assertLessEqual(size, next_size)
        self.assertEqual(len(EN_FR_CHAR_PAIRS), 204)


if __name__ == '__main__':
    unittest.main()
