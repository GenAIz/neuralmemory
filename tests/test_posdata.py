import unittest

from nm import posdata


class TestPOSData(unittest.TestCase):

    def test_posdata(self):
        vecs = posdata.udpos_one_hot_vectors()
        words = None
        tags = None
        for words, _, tags in posdata.UDPOS_DATASETS[0]:
            break
        seq = posdata.UDPOSSeq(0, words, tags, vecs)
        self.assertEqual(len(seq.inputs), 29)
        self.assertEqual(len(seq.targets), 29)

        temp = []
        while not seq.exhausted():
            temp.append(seq.next())
        self.assertEqual(len(temp), 29)

        seq = posdata.UDPOSSeq(0, words[:-1], tags[:-1], vecs)
        self.assertEqual(len(seq.inputs), 29)
        self.assertEqual(seq.inputs[-1], ".")
        self.assertEqual(len(seq.targets), 29)
        self.assertEqual(seq.targets[-1], ".")


if __name__ == '__main__':
    unittest.main()
