import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sse(val):
    return np.sum(np.square(np.subtract(np.mean(val), val)))


dc_bx = np.array([-1, 0, 1, 2])
dc_bd = []
weight_list = []
error_list = []
eta = 0.3
input = np.array([[0, 0, 1, 1], [0, 1, 0, 1], [-1, -1, -1, -1]])
target = np.array([[0, 1, 1, 0]])
print(input.shape[1])
W = np.random.rand(1, input.shape[0])
# W = np.array([[0.004501292851472, -0.002688614864257, 0.001068425835418]])
print(W)
# W= np.array([[0.0045,-0.0027,0.0011]])
# init finish
slop = -1 * W[0][0] / W[0][1]
dc_by = []
for i in dc_bx:
    dc_by.append(slop * i + W[0][2] / W[0][1])
dc_bd.append(dc_by)

print(dc_bd)
epochs = 100
print(f"Epochs \t\t error")
for i in range(epochs + 1):
    y_k = np.dot(W, input)
    y = sigmoid(y_k)
    error = target - y
    # f_d = 0.5 * (error * (1 - np.square(y)))
    f_d = error
    delta = np.dot(f_d, np.transpose(input))
    W = W + eta * delta
    weight_list.append(W[0])
    error_list.append(sse(error[0] / 4))
    if i != 0 and i % epochs == 0:
        print(f"{i} \t\t {error_list[i]}")
        slop = -1 * (weight_list[i][0] / weight_list[i][1])
        dc_by = []
        for x in dc_bx:
            temp = weight_list[i][2]
            dc_by.append(slop * x + weight_list[i][2] / weight_list[i][1])
        dc_bd.append(dc_by)

print("Input")
print(input)
print("Output")
print(target)
print("Error")
print(error)

# print((dc_bx,dc_bd[i]) for i in range(5))
fig, ax = plt.subplots()
ax.plot([0, 1], [0, 1], 'bo')
ax.plot([1, 0], [0, 1], 'r+')
for i, x in enumerate(dc_bd):
    ax.plot(dc_bx, dc_bd[i], label=f"{i}th")
ax.set(xlabel='x-label', ylabel='y-label')
ax.legend()
fig.suptitle("Decision Boundary")
# ax.axis([-0.25, 1.5, -0.25, 1.5])

fig, ax = plt.subplots()
fig.suptitle("Error convergence curve")
ax.plot([i for i in range(epochs + 1)], error_list)

plt.show()
