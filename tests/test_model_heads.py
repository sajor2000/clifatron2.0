import unittest

import torch
from torch import nn

from src.model.heads import NextEventHead


class NextEventHeadTest(unittest.TestCase):
    def test_untied_by_default_and_tied_for_ablation(self):
        embedding = nn.Embedding(11, 4)
        untied = NextEventHead(4, 11, input_embedding=embedding)
        tied = NextEventHead(4, 11, tie_weights=True, input_embedding=embedding)

        self.assertNotEqual(untied.projection.weight.data_ptr(), embedding.weight.data_ptr())
        self.assertEqual(tied.projection.weight.data_ptr(), embedding.weight.data_ptr())
        self.assertEqual(untied(torch.zeros(2, 3, 4)).shape, (2, 3, 11))


if __name__ == "__main__":
    unittest.main()
