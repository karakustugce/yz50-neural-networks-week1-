"""Shape and axis checks that target the difficult part of the assignment."""
import torch
from layers import BatchNorm1d, Flatten, make_model


def test_pair_shapes():
    x = torch.zeros(2, 8, 24)
    assert Flatten(2)(x).shape == (2, 4, 48)
    assert Flatten(2)(torch.zeros(2, 2, 128)).shape == (2, 256)
    model = make_model(27, 8, 24, 128, hierarchical=True)
    model.train()
    output, trace = model(torch.zeros(2, 8, dtype=torch.long), trace=True)
    assert output.shape == (2, 27)
    assert [shape for name, shape in trace if name == "Flatten"] == [
        [2, 4, 48], [2, 2, 256], [2, 256]]


def test_batchnorm_axes():
    # Position 0 has mean 0 and position 1 has mean 10. Both positions
    # must share feature statistics across samples AND sequence positions.
    x = torch.tensor([[[0.], [10.]], [[0.], [10.]]])
    fixed = BatchNorm1d(1)
    legacy = BatchNorm1d(1, legacy_3d=True)
    correct = fixed(x)
    wrong = legacy(x)
    assert torch.allclose(fixed.running_mean.flatten(), torch.tensor([5.]))
    assert torch.allclose(legacy.running_mean.flatten(), torch.tensor([0., 10.]))
    assert correct[0, 0, 0] < -0.99 and correct[0, 1, 0] > 0.99
    assert torch.allclose(wrong, torch.zeros_like(wrong))


if __name__ == "__main__":
    torch.set_num_threads(1)
    test_pair_shapes()
    test_batchnorm_axes()
    print("shape and BatchNorm axis checks passed")
