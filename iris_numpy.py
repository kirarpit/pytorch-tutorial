from sklearn.datasets import load_iris
import numpy as np
np.random.seed(5)

X, y = load_iris(return_X_y=True)
data = np.hstack([X, y[:, None]])
np.random.shuffle(data)
X = data[:, :-1]
y = data[:, -1]
X_train = X[:100]
y_train = y[:100]
X_test = X[100:]
y_test = y[100:]

y_train = y_train.astype("int")
lr = 1e-3

# weight initialization
w1 = np.random.normal(0, 1e-0, (4, 10))
w2 = np.random.normal(0, 1e-0, (10, 10))
w3 = np.random.normal(0, 1e-0, (10, 3))

for _ in range(100):
    # forward pass
    z1 = X_train.dot(w1)
    a1 = (z1>0)*z1
    z2 = a1.dot(w2)
    a2 = (z2>0)*z2
    z3 = a2.dot(w3)

    # softmax
    z3 = np.exp(z3 - np.max(z3))
    a3 = z3/np.sum(z3, axis=1, keepdims=True)

    # compute loss
    loss = -np.mean(np.log(a3[np.arange(len(X_train)), y_train]))

    # print acc and loss
    print("accuracy {}, loss {}".format(np.mean(np.argmax(a3, axis=1) == y_train), loss))

    # compute grads
    a3[np.arange(X_train.shape[0]), y_train] -= 1
    a3 /= len(X_train)
    dw3 = a2.T.dot(a3)
    da2 = a3.dot(w3.T)
    dz2 = (z2>0)*da2
    dw2 = a1.T.dot(dz2)
    da1 = dz2.dot(w2.T)
    dz1 = (z1>0)*da1
    dw1 = X_train.T.dot(dz1)

    # update weights
    w1 -= lr*dw1
    w2 -= lr*dw2
    w3 -= lr*dw3

# test accuracy
z1 = X_test.dot(w1)
a1 = (z1>0)*z1
z2 = a1.dot(w2)
a2 = (z2>0)*z2
z3 = a2.dot(w3)
a3 = z3/np.sum(z3, axis=1, keepdims=True)
print("accuracy {}".format(np.mean(np.argmax(a3, axis=1) == y_test)))
