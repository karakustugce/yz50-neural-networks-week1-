import math
class Value:
    def __init__(self, data, _children=(), _op="", label=""):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad}, label='{self.label}')"

    def __add__(self, other):
        out = Value(
            self.data + other.data,
            (self, other),
            "+",
        )
        return out

    def __mul__(self, other):
        out = Value(
            self.data * other.data,
            (self, other),
            "*",
        )
        return out
    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)

        out = Value(t, (self,), "tanh")
        return out


# Karpathy videosundaki basit ifade
# Nöronun girdileri
x1 = Value(2.0, label="x1")
x2 = Value(0.0, label="x2")

# Nöronun ağırlıkları
w1 = Value(-3.0, label="w1")
w2 = Value(1.0, label="w2")

# Bias
b = Value(6.8813735870195432, label="b")

# x1*w1 ve x2*w2
x1w1 = x1 * w1
x1w1.label = "x1*w1"

x2w2 = x2 * w2
x2w2.label = "x2*w2"

# Ağırlıklı toplam
x1w1x2w2 = x1w1 + x2w2
x1w1x2w2.label = "x1*w1 + x2*w2"

n = x1w1x2w2 + b
n.label = "n"

# Aktivasyon
o = n.tanh()
o.label = "o"

print(x1w1)
print(x2w2)
print(x1w1x2w2)
print(n)
print(o)

# Gradient'leri elle hesaplıyoruz
o.grad = 1.0

# o = tanh(n)
n.grad = (1 - o.data**2) * o.grad

# n = x1w1x2w2 + b
x1w1x2w2.grad = 1.0 * n.grad
b.grad = 1.0 * n.grad

# x1w1x2w2 = x1w1 + x2w2
x1w1.grad = 1.0 * x1w1x2w2.grad
x2w2.grad = 1.0 * x1w1x2w2.grad

# x1w1 = x1 * w1
x1.grad = w1.data * x1w1.grad
w1.grad = x1.data * x1w1.grad

# x2w2 = x2 * w2
x2.grad = w2.data * x2w2.grad
w2.grad = x2.data * x2w2.grad

print("\nElle hesaplanan gradient'ler:")
print("o.grad:", o.grad)
print("n.grad:", n.grad)
print("b.grad:", b.grad)
print("x1.grad:", x1.grad)
print("w1.grad:", w1.grad)
print("x2.grad:", x2.grad)
print("w2.grad:", w2.grad)