import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
        self.weights = np.zeros(input_size + 1)  # +1 для bias
        self.learning_rate = learning_rate

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        return 1 if summation >= 0 else 0

    def train(self, training_inputs, labels, epochs=10):
        for _ in range(epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)  # Bias

# Навчальні дані для функції XOR
training_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
labels = np.array([0, 1, 1, 0])

# Додавання нових ознак (добуток x1 * x2)
extended_inputs = np.hstack((training_inputs, (training_inputs[:, 0] * training_inputs[:, 1]).reshape(-1, 1)))

# Навчання моделі на розширених даних
perceptron_extended = Perceptron(input_size=3, learning_rate=0.1)
perceptron_extended.train(extended_inputs, labels, epochs=10)

# Візуалізація результатів для розширених даних
x_vals = np.linspace(-0.5, 1.5, 100)
y_vals = -(perceptron_extended.weights[1] * x_vals + perceptron_extended.weights[0]) / perceptron_extended.weights[2]

# Візуалізація
plt.figure(figsize=(6, 5))
plt.scatter(training_inputs[:, 0], training_inputs[:, 1], c=labels, cmap='bwr', edgecolors='k', label="Точки даних")
plt.plot(x_vals, y_vals, label="Decision Boundary", color="blue")
plt.title("Персептрон для функції XOR (з новими ознаками)")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.show()
