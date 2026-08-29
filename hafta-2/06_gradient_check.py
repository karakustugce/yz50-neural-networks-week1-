import math
import torch

class Value:
    def __init__(self, data, _children=(), _op="", label=""):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self._backward = lambda: None

    def __repr__(self):
        return (
            f"Value(data={self.data}, "
            f"grad={self.grad}, "
            f"label='{self.label}')"
        )

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        out = Value(
            self.data + other.data,
            (self, other),
            "+",
        )

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        out = Value(
            self.data * other.data,
            (self, other),
            "*",
        )

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
        if not isinstance(other, (int, float)):
            raise TypeError("Üs int veya float olmalıdır.")

        out = Value(
            self.data**other,
            (self,),
            f"**{other}",
        )

        def _backward():
            local_derivative = other * self.data ** (other - 1)
            self.grad += local_derivative * out.grad

        out._backward = _backward
        return out

    def exp(self):
        out = Value(
            math.exp(self.data),
            (self,),
            "exp",
        )

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1

    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (
            math.exp(2 * x) + 1
        )

        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(node):
            if node not in visited:
                visited.add(node)

                for child in node._prev:
                    build_topo(child)

                topo.append(node)

        build_topo(self)

        self.grad = 1.0

        for node in reversed(topo):
            node._backward()


# Nöronun girdileri
x1 = Value(2.0, label="x1")
x2 = Value(0.0, label="x2")

# Nöronun ağırlıkları
w1 = Value(-3.0, label="w1")
w2 = Value(1.0, label="w2")

# Bias
b = Value(6.8813735870195432, label="b")

# Çarpma işlemleri
x1w1 = x1 * w1
x1w1.label = "x1*w1"

x2w2 = x2 * w2
x2w2.label = "x2*w2"

# Toplama işlemleri
x1w1x2w2 = x1w1 + x2w2
x1w1x2w2.label = "x1*w1 + x2*w2"

n = x1w1x2w2 + b
n.label = "n"

# tanh fonksiyonunu temel işlemlere ayırıyoruz
e = (2 * n).exp()
e.label = "e"

o = (e - 1) / (e + 1)
o.label = "o"

print("Forward pass:")
print(x1w1)
print(x2w2)
print(x1w1x2w2)
print(n)
print(o)

# Bütün gradient'leri otomatik hesapla
o.backward()

print("\nOtomatik hesaplanan gradient'ler:")
print("o.grad:", o.grad)
print("n.grad:", n.grad)
print("b.grad:", b.grad)
print("x1.grad:", x1.grad)
print("w1.grad:", w1.grad)
print("x2.grad:", x2.grad)
print("w2.grad:", w2.grad)

# Aynı değişkenin birden fazla kullanılması testi
a = Value(3.0, label="a")
result = a + a
result.label = "result"

result.backward()

print("\nGradient toplama testi:")
print("result.data:", result.data)
print("result.grad:", result.grad)
print("a.grad:", a.grad)

# Normal Python sayılarıyla işlem testi
number_test = Value(3.0, label="number_test")
addition_test = number_test + 2
multiplication_test = 2 * number_test

print("\nNormal sayılarla işlem testi:")
print("3 + 2 =", addition_test.data)
print("2 * 3 =", multiplication_test.data)

# -------------------------------------------------
# Üçlü gradient doğrulaması
# 1. Kendi backward sistemimiz
# 2. Numerical derivative
# 3. PyTorch autograd
# -------------------------------------------------

def neuron_output(values):
    x1_value, x2_value, w1_value, w2_value, b_value = values

    n_value = (
        x1_value * w1_value
        + x2_value * w2_value
        + b_value
    )

    return math.tanh(n_value)


base_values = [
    2.0,
    0.0,
    -3.0,
    1.0,
    6.8813735870195432,
]


def numerical_gradient(parameter_index, h=1e-6):
    values_plus = base_values.copy()
    values_minus = base_values.copy()

    values_plus[parameter_index] += h
    values_minus[parameter_index] -= h

    output_plus = neuron_output(values_plus)
    output_minus = neuron_output(values_minus)

    return (output_plus - output_minus) / (2 * h)


parameter_names = ["x1", "x2", "w1", "w2", "b"]

custom_gradients = [
    x1.grad,
    x2.grad,
    w1.grad,
    w2.grad,
    b.grad,
]

numerical_gradients = [
    numerical_gradient(index)
    for index in range(len(base_values))
]

# Aynı nöronu PyTorch ile oluşturuyoruz
x1_torch = torch.tensor(
    2.0,
    dtype=torch.float64,
    requires_grad=True,
)

x2_torch = torch.tensor(
    0.0,
    dtype=torch.float64,
    requires_grad=True,
)

w1_torch = torch.tensor(
    -3.0,
    dtype=torch.float64,
    requires_grad=True,
)

w2_torch = torch.tensor(
    1.0,
    dtype=torch.float64,
    requires_grad=True,
)

b_torch = torch.tensor(
    6.8813735870195432,
    dtype=torch.float64,
    requires_grad=True,
)

n_torch = (
    x1_torch * w1_torch
    + x2_torch * w2_torch
    + b_torch
)

o_torch = torch.tanh(n_torch)
o_torch.backward()

torch_gradients = [
    x1_torch.grad.item(),
    x2_torch.grad.item(),
    w1_torch.grad.item(),
    w2_torch.grad.item(),
    b_torch.grad.item(),
]

print("\nÜçlü gradient karşılaştırması:")

for name, custom, numerical, pytorch in zip(
    parameter_names,
    custom_gradients,
    numerical_gradients,
    torch_gradients,
):
    print(
        f"{name}: "
        f"backward={custom:.8f}, "
        f"numerical={numerical:.8f}, "
        f"pytorch={pytorch:.8f}"
    )