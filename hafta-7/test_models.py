"""Veri hizası, geleceğe sızıntı, ölçek ve uzun üretim kontrolleri."""
import unittest
import torch
from data import CharacterTokenizer, get_batch
from models import BigramLanguageModel, Head, SingleHeadLanguageModel


class ModelChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1337)
        torch.set_num_threads(1)

    def test_tokenizer_and_batch_shift(self):
        tok = CharacterTokenizer('Aa\n !')
        self.assertEqual(tok.decode(tok.encode('Aa\n !')), 'Aa\n !')
        splits = {'train': torch.arange(20), 'val': torch.arange(100, 120)}
        for split in splits:
            x, y = get_batch(splits, split, 8, 4)
            self.assertEqual(tuple(x.shape), (4, 8))
            self.assertTrue(torch.equal(y, x + 1))
            self.assertTrue(torch.equal(y[:, :-1], x[:, 1:]))
            self.assertTrue(bool((x >= (100 if split == 'val' else 0)).all()))
        # En kısa geçerli veri: yalnızca bir başlangıç konumu.
        x, y = get_batch({'train': torch.arange(9)}, 'train', 8, 1)
        self.assertTrue(torch.equal(x[0], torch.arange(8)))
        self.assertEqual(int(y[0, -1]), 8)

    def test_mask_rows_and_no_future_leakage(self):
        head = Head(8, 4, 8)
        x = torch.randn(2, 8, 8)
        _, wei = head(x, return_weights=True)
        self.assertTrue(torch.equal(wei.triu(1), torch.zeros_like(wei)))
        self.assertTrue(torch.allclose(wei.sum(-1), torch.ones(2, 8)))
        changed = x.clone()
        changed[:, 4:] += torch.randn_like(changed[:, 4:]) * 10
        self.assertTrue(torch.allclose(head(x)[:, :4], head(changed)[:, :4]))

    def test_model_shapes_gradients_and_long_generation(self):
        for model in (BigramLanguageModel(7, 8), SingleHeadLanguageModel(7, 8, 12, 4)):
            x = torch.randint(7, (2, 8))
            logits, loss = model(x, x)
            self.assertEqual(tuple(logits.shape), (2, 8, 7))
            loss.backward()
            self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()))
            sample = model.generate(x, 20, torch.Generator().manual_seed(42))
            self.assertEqual(tuple(sample.shape), (2, 28))
            self.assertTrue(model.training)
        with self.assertRaises(ValueError):
            SingleHeadLanguageModel(7, 8)(torch.ones(1, 9, dtype=torch.long))

    def test_scale_uses_head_size_not_embedding_size(self):
        head = Head(12, 4, 5)
        x = torch.randn(2, 5, 12)
        scores = head.query(x) @ head.key(x).transpose(-2, -1)
        expected_weights = (scores / 2).masked_fill(~head.tril, float('-inf')).softmax(-1)
        _, weights = head(x, return_weights=True)
        self.assertTrue(torch.allclose(weights, expected_weights))


if __name__ == '__main__':
    unittest.main()
