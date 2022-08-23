import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
D = np.array([[0], [0], [1], [1]])
W = np.random.rand(1, 3)
ALPHA = 0.9
for i in range(100):
    for j in range(D.shape[0]):
        x = np.reshape(X[j], (3, 1))
        d = D[j, :]
        v = np.dot(W, x)
        y = sigmoid(v)
        e = d - y
        delta = y * (1 - y) * e
        dW = ALPHA * delta * x
        W = W + np.transpose(dW)

for i in range(D.shape[0]):
    x = np.reshape(X[i], (3, 1))
    v = np.dot(W, x)
    y = sigmoid(v)
    print(np.round(y))
