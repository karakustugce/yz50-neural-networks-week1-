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


# Karpathy videosundaki basit ifade
a = Value(2.0, label="a")
b = Value(-3.0, label="b")
c = Value(10.0, label="c")

e = a * b
e.label = "e"

d = e + c
d.label = "d"

f = Value(-2.0, label="f")

L = d * f
L.label = "L"

print(a)
print(b)
print(c)
print(e)
print(d)
print(f)
print(L)
# Gradient'leri L'den geriye doğru elle hesaplıyoruz
L.grad = 1.0

d.grad = f.data * L.grad
f.grad = d.data * L.grad

e.grad = 1.0 * d.grad
c.grad = 1.0 * d.grad

a.grad = b.data * e.grad
b.grad = a.data * e.grad

print("\nElle hesaplanan gradient'ler:")
print("L.grad:", L.grad)
print("d.grad:", d.grad)
print("f.grad:", f.grad)
print("e.grad:", e.grad)
print("c.grad:", c.grad)
print("a.grad:", a.grad)
print("b.grad:", b.grad)