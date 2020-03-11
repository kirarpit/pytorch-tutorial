import torch
import torch.nn.functional as F
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

# initialize weights
w1 = torch.from_numpy(np.random.normal(0, 1e-0, (4, 10)))
w2 = torch.from_numpy(np.random.normal(0, 1e-0, (10, 10)))
w3 = torch.from_numpy(np.random.normal(0, 1e-0, (10, 3)))

[w.requires_grad_() for w in [w1, w2, w3]]

loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
opt = torch.optim.Adam([w1, w2, w3], lr=lr)

for _ in range(100):
    # feed forward
    z1 = F.linear(X_train, w1.T)
    a1 = F.relu(z1)
    a3 = F.softmax(F.linear(F.relu(F.linear(a1, w2.T)), w3.T), dim=1)

    # print acc
    print(torch.Tensor.eq(a3.argmax(dim=1), y_train).float().mean())

    # compute loss
    loss = -a3[np.arange(len(X_train)), y_train.tolist()].log().mean()

    # compute grads
    opt.zero_grad()
    loss.backward()

    # update weights
    opt.step()
