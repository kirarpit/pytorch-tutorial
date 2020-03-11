import torch
from torch import Tensor
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

lr = 1e-3
# convert numpy to Tensor
X_train = torch.from_numpy(X_train)
y_train = torch.from_numpy(y_train)

# intialize weights
w1 = torch.from_numpy(np.random.normal(0, 1e-0, (4, 10)))
w2 = torch.from_numpy(np.random.normal(0, 1e-0, (10, 10)))
w3 = torch.from_numpy(np.random.normal(0, 1e-0, (10, 3)))

[w.requires_grad_() for w in [w1, w2, w3]]

for _ in range(100):
    # forward pass
    z1 = X_train.mm(w1)
    a1 = z1*(z1>0)
    z2 = a1.mm(w2)
    a2 = z2*(z2>0)
    z3 = a2.mm(w3)

    # softmax
    z3 = z3 - z3.max()
    z3 = z3.exp()
    a3 = z3/z3.sum(dim=1, keepdim=True)

    # a3 = X_train.mm(w1).clamp(min=0).mm(w2).clamp(min=0).mm(w3).softmax(dim=1)

    # compute acc
    preds = a3.argmax(dim=1)
    print(Tensor.eq(preds, y_train).float().mean())

    # compute loss
    loss = -a3[np.arange(len(X_train)), y_train.tolist()].log().mean()

    # compute gradients
    loss.backward()

    # update weights
    with torch.no_grad():
        w1 -= lr*w1.grad
        w2 -= lr*w2.grad
        w3 -= lr*w3.grad

        # zero the gradients
        [w.grad.zero_() for w in [w1, w2, w3]]
