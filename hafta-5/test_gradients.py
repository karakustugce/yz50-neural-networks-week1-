"""Ek bagimlilik olmadan calisan hafta 5 kontrolleri."""

import unittest

from ortak import (
    comparison,
    detailed_forward,
    fused_parameter_grads,
    get_splits,
    init_parameters,
    make_batch,
    manual_backward,
    retain_and_backward,
)


class GradientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Xtr, Ytr = get_splits()["train"]
        cls.Xb, cls.Yb = make_batch(Xtr, Ytr)
        cls.parameters = init_parameters()
        cls.loss, cls.values = detailed_forward(cls.parameters, cls.Xb, cls.Yb)
        retain_and_backward(cls.loss, cls.values, cls.parameters)

    def test_every_intermediate_gradient(self):
        manual = manual_backward(self.parameters, self.values, self.Xb, self.Yb)
        for name, gradient in manual.items():
            target = self.parameters[name] if name in self.parameters else self.values[name]
            self.assertTrue(comparison(name, gradient, target)["approximate"], name)

    def test_fused_cross_entropy_and_batchnorm(self):
        fused = fused_parameter_grads(self.parameters, self.values, self.Xb, self.Yb)
        self.assertTrue(
            comparison("logits", fused["logits"], self.values["logits"])["approximate"]
        )
        self.assertTrue(
            comparison("hprebn", fused["hprebn"], self.values["hprebn"])["approximate"]
        )


if __name__ == "__main__":
    unittest.main()
