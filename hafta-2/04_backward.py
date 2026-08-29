import math


class Value:
    def __init__(self, data, _children=(), _op="", label=""):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad}, label='{self.label}')"

    def __add__(self, other):
        out = Value(
            self.data + other.data,
            (self, other),
            "+",
        )

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
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

    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)

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

# Aktivasyon
o = n.tanh()
o.label = "o"

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