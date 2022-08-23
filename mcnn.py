import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


one = np.array([[0, 1, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 1, 1, 1, 0]])
two = np.array([[1, 1, 1, 1, 0],
                [0, 0, 0, 0, 1],
                [0, 1, 1, 1, 0],
                [1, 0, 0, 0, 0],
                [1, 1, 1, 1, 1]
                ])
three = np.array([[1, 1, 1, 1, 0],
                  [0, 0, 0, 0, 1],
                  [0, 1, 1, 1, 0],
                  [0, 0, 0, 0, 1],
                  [1, 1, 1, 1, 0]])
X = np.array([one, two, three])
D = np.array([[1, 0, 0, 0, 0],
              [0, 1, 0, 0, 0],
              [0, 0, 1, 0, 0],
              ])
assert X.shape[0] == D.shape[0], "Your are Fucked up!!"

W1 = np.random.rand(50, 25)
W2 = np.random.rand(5, 50)
ALPHA = 0.9
epchos = 100
# print(D)
for i in range(epchos+1):
    for j, val in enumerate(D):
        x = np.reshape(X[j], (25, 1))
        d = np.reshape(val, (5, 1))
        v1 = np.dot(W1, x)
        y1 = sigmoid(v1)
        v = np.dot(W2, y1)
        y = softmax(v)
        e = d - y
        delta = e
        e1 = np.dot(np.transpose(W2), delta)
        delta1 = y1 * (1 - y1) * e1
        dW1 = ALPHA * delta1 * np.transpose(x)
        W1 = W1 + dW1
        dW2 = ALPHA * delta * np.transpose(y1)
        W2 = W2 + dW2

# prediction Stage

for j, val in enumerate(D):
    x = np.reshape(X[j], (25, 1))
    v1 = np.dot(W1, x)
    y1 = sigmoid(v1)
    v = np.dot(W2, y1)
    y = np.round(softmax(v)).transpose()[0]
    print(y)
    print()
