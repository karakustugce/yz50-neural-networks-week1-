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


a = Value(2.0, label="a")
b = Value(3.0, label="b")
c = Value(4.0, label="c")

d = a * b
d.label = "d"

e = d + c
e.label = "e"

print(a)
print(b)
print(c)
print(d)
print(e)

print("d işlemi:", d._op)
print("d önceki değerleri:", d._prev)

print("e işlemi:", e._op)
print("e önceki değerleri:", e._prev)

# Gradient'leri şimdilik elle hesaplıyoruz
e.grad = 1.0
d.grad = 1.0
c.grad = 1.0
a.grad = b.data * d.grad
b.grad = a.data * d.grad

print("\nElle hesaplanan gradient'ler:")
print("a.grad:", a.grad)
print("b.grad:", b.grad)
print("c.grad:", c.grad)
print("d.grad:", d.grad)
print("e.grad:", e.grad)