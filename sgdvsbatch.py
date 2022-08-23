import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def DeltaSGD(W, X, D):
    for j in range(D.shape[1]):
        x = np.reshape(X[j], (3, 1))
        d = D[:, j]
        v = np.dot(W, x)
        y = sigmoid(v)
        e = d - y
        delta = y * (1 - y) * e
        dW = ALPHA * delta * x
        W = W + np.transpose(dW)
    return W


def DeltaBatch(W, X, D):
    dWsum = np.zeros((3, 1))
    for j in range(D.shape[1]):
        x = np.reshape(X[j], (3, 1))
        d = D[:, j]
        v = np.dot(W, x)
        y = sigmoid(v)
        e = d - y
        delta = y * (1 - y) * e
        dW = ALPHA * delta * x
        dWsum = dWsum + dW

    dWavg = np.divide(dWsum, 4)
    W = W + np.transpose(dWavg)
    return W


X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
D = np.array([[0, 0, 1, 1]])
E1 = np.zeros((1000, 1))
E2 = np.zeros((1000, 1))
ALPHA = 0.9
W1 = np.random.randn(1, 3)
W2 = W1
epoch = 1000
for i in range(epoch):
    W1 = DeltaSGD(W1, X, D)
    W2 = DeltaBatch(W2, X, D)
    es1 = 0
    es2 = 0
    N = 4
    for k in range(N):
        x = np.reshape(X[k], (3, 1))
        d = D[:, k]
        v1 = np.dot(W1, x)
        y1 = sigmoid(v1)
        es1 = es1 + np.square(d - y1)

        v2 = np.dot(W2, x)
        y2 = sigmoid(v2)
        es2 = es2 + np.square(d - y2)

    E1[i] = es1 / N
    E2[i] = es2 / N

plt.plot(E1, 'r')
plt.plot(E2, 'b:')
plt.xlabel('Epoch')
plt.ylabel('Average of Training error')
plt.legend('SGD', 'Batch')
plt.show()
