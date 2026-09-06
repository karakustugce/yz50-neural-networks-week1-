# ============================================================
# YZ50 - Week 1: Neural Networks
# ============================================================
#
# Bu projede sinir ağlarının temel çalışma mantığını anlamaya
# çalışıyorum. Hazır bir machine learning kütüphanesi kullanmadan
# tek bir nöronun forward pass'inden başlayıp numerical gradient
# descent'e kadar temel yapıları kendim oluşturuyorum.
#
# Ana kavramlar:
# input -> weight -> weighted sum -> bias -> activation -> output
# ============================================================


# ============================================================
# 1. TEK NÖRON - FORWARD PASS
# ============================================================
#
# Bir nöron, kendisine gelen input değerlerini weight'lerle çarpar.
# Daha sonra bu sonuçları toplar ve bias ekler.
#
# Matematiksel olarak:
#
# z = x1*w1 + x2*w2 + x3*w3 + bias
#
# Ardından istersek z değerini bir activation function'dan geçiririz.
# Burada basit olması için ReLU kullanıyorum.
# ============================================================


def relu(x):
    # ReLU negatif değerleri 0 yapar,
    # pozitif değerleri ise değiştirmeden bırakır.
    if x > 0:
        return x
    return 0


def single_neuron_forward(inputs, weights, bias):

    weighted_sum = 0

    # Her input'u kendi weight'i ile çarpıp topluyorum.
    for i in range(len(inputs)):
        weighted_sum += inputs[i] * weights[i]

    # Bias değerini ekliyorum.
    weighted_sum += bias

    # Sonucu activation function'dan geçiriyorum.
    output = relu(weighted_sum)

    return output


inputs = [1.0, 2.0, 3.0]
weights = [0.2, -0.5, 1.0]
bias = 0.1

neuron_output = single_neuron_forward(inputs, weights, bias)

print("1) Tek nöron output:")
print(neuron_output)


# ============================================================
# 2. BİRDEN FAZLA NÖRON - KÜÇÜK BİR LAYER
# ============================================================
#
# Bir layer aslında birden fazla nöronun yan yana bulunmasıdır.
#
# Aynı input değerleri her nörona gider fakat her nöronun
# kendine ait farklı weight ve bias değerleri vardır.
#
# Bu yüzden aynı input'tan farklı output'lar elde edebiliriz.
# ============================================================


def layer_forward(inputs, layer_weights, layer_biases):

    layer_outputs = []

    # Layer içerisindeki her nöronu tek tek hesaplıyorum.
    for neuron_index in range(len(layer_weights)):

        weights = layer_weights[neuron_index]
        bias = layer_biases[neuron_index]

        output = single_neuron_forward(
            inputs,
            weights,
            bias
        )

        layer_outputs.append(output)

    return layer_outputs


# Layer'da 3 tane nöron var.
layer_weights = [
    [0.2, -0.5, 1.0],
    [0.5, 0.1, -0.2],
    [-0.3, 0.8, 0.4]
]

layer_biases = [
    0.1,
    0.2,
    -0.1
]

layer_output = layer_forward(
    inputs,
    layer_weights,
    layer_biases
)

print("\n2) Layer output:")
print(layer_output)


# ============================================================
# 3. LOSS FUNCTION
# ============================================================
#
# Modelin ürettiği prediction ile gerçek target arasındaki farkı
# ölçebilmek için loss function kullanıyoruz.
#
# Loss ne kadar küçükse modelin tahmini gerçek değere o kadar yakındır.
#
# Burada Mean Squared Error (MSE) kullanıyorum.
#
# MSE:
# (prediction - target)^2 değerlerinin ortalamasıdır.
# ============================================================


def mse_loss(predictions, targets):

    total_error = 0

    for i in range(len(predictions)):

        error = predictions[i] - targets[i]

        squared_error = error ** 2

        total_error += squared_error

    average_error = total_error / len(predictions)

    return average_error


predictions = [2.5, 4.5, 5.5]
targets = [2.0, 4.0, 6.0]

loss = mse_loss(predictions, targets)

print("\n3) MSE Loss:")
print(loss)


# ============================================================
# 4. PARAMETRE DEĞİŞTİREREK LOSS'U GÖZLEMLEME
# ============================================================
#
# Şimdi çok basit bir model oluşturuyorum:
#
# prediction = x * weight + bias
#
# Dataset'teki gerçek ilişki:
#
# y = 2x
#
# Weight değerini manuel olarak değiştirerek loss'un nasıl
# değiştiğini gözlemlemek istiyorum.
#
# Model doğru weight'e, yani yaklaşık 2'ye yaklaştıkça
# loss'un düşmesini bekliyorum.
# ============================================================


x_train = [
    1.0,
    2.0,
    3.0,
    4.0
]

y_train = [
    2.0,
    4.0,
    6.0,
    8.0
]


def predict(x, weight, bias):

    return x * weight + bias


def calculate_model_loss(weight, bias):

    predictions = []

    for x in x_train:

        prediction = predict(
            x,
            weight,
            bias
        )

        predictions.append(prediction)

    loss = mse_loss(
        predictions,
        y_train
    )

    return loss


# Manuel olarak deneyeceğim weight değerleri.
manual_weights = [
    -1.0,
    -0.5,
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    2.5,
    3.0,
    3.5,
    4.0
]

manual_losses = []

print("\n4) Weight değiştikçe loss:")

for weight in manual_weights:

    loss = calculate_model_loss(
        weight,
        0.0
    )

    manual_losses.append(loss)

    print(
        "Weight:",
        weight,
        "| Loss:",
        loss
    )


# ============================================================
# 5. LOSS EĞRİSİNİ ÇİZME
# ============================================================
#
# Yukarıda farklı weight değerleri için hesapladığım loss'ları
# grafik üzerinde gösteriyorum.
#
# Grafikte minimum loss'un weight=2 civarında oluşmasını
# bekliyorum çünkü gerçek dataset y = 2x ilişkisine sahip.
#
# Bu bölümde sadece grafik çizmek için matplotlib kullanıyorum.
# ============================================================


import matplotlib.pyplot as plt


plt.figure()

plt.plot(
    manual_weights,
    manual_losses,
    marker="o"
)

plt.xlabel("Weight")
plt.ylabel("Loss")
plt.title("Weight Değiştikçe Loss")

plt.grid()

plt.show()


# ============================================================
# 6. NUMERICAL DERIVATIVE
# ============================================================
#
# Gradient descent yapabilmek için loss'un parametreye göre
# hangi yönde değiştiğini bilmem gerekiyor.
#
# Bunun için numerical derivative kullanıyorum.
#
# Bir parametrenin biraz sağındaki ve biraz solundaki loss'u
# hesaplayarak eğimin yaklaşık değerini buluyorum.
#
# Formül:
#
# f'(x) ≈ [f(x+h) - f(x-h)] / 2h
#
# h burada çok küçük bir sayıdır.
# ============================================================


def numerical_derivative(function, x, h=0.00001):

    right_side = function(x + h)

    left_side = function(x - h)

    derivative = (
        right_side - left_side
    ) / (2 * h)

    return derivative


# Bias'ı şimdilik 0 olarak sabit tutuyorum.
def loss_for_weight(weight):

    return calculate_model_loss(
        weight,
        0.0
    )


example_weight = 1.0

gradient = numerical_derivative(
    loss_for_weight,
    example_weight
)

print("\n5) Numerical derivative:")

print(
    "Weight:",
    example_weight
)

print(
    "Gradient:",
    gradient
)


# ============================================================
# 7. GRADIENT DESCENT
# ============================================================
#
# Gradient bize loss'un hangi yönde arttığını gösteriyor.
#
# Biz loss'u azaltmak istediğimiz için gradient'in ters yönünde
# hareket ediyoruz.
#
# Güncelleme formülü:
#
# weight = weight - learning_rate * gradient
#
# Learning rate her adımda ne kadar ilerleyeceğimizi belirliyor.
#
# Burada weight'i -1'den başlatıyorum.
# Model gradient descent ile doğru değer olan 2'ye yaklaşmaya
# çalışacak.
# ============================================================


weight = -1.0

learning_rate = 0.05

epochs = 30

gradient_descent_losses = []

print("\n6) Gradient Descent başlıyor:")


for epoch in range(epochs):

    # Şu anki weight için loss hesaplanıyor.
    current_loss = loss_for_weight(weight)

    gradient_descent_losses.append(
        current_loss
    )

    # Numerical derivative ile gradient hesaplanıyor.
    gradient = numerical_derivative(
        loss_for_weight,
        weight
    )

    # Weight gradient'in ters yönünde güncelleniyor.
    weight = weight - learning_rate * gradient

    print(
        "Epoch:",
        epoch,
        "| Weight:",
        round(weight, 5),
        "| Loss:",
        round(current_loss, 5),
        "| Gradient:",
        round(gradient, 5)
    )


print("\nTraining tamamlandı.")

print(
    "Öğrenilen Weight:",
    weight
)

print(
    "Final Loss:",
    loss_for_weight(weight)
)


# ============================================================
# GRADIENT DESCENT LOSS GRAFİĞİ
# ============================================================
#
# Son olarak gradient descent ilerledikçe loss'un gerçekten
# düşüp düşmediğini grafik üzerinde gösteriyorum.
#
# Eğitim başarılıysa loss zamanla sıfıra yaklaşmalıdır.
# ============================================================


plt.figure()

plt.plot(
    range(epochs),
    gradient_descent_losses
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "Gradient Descent Sırasında Loss"
)

plt.grid()

plt.show()