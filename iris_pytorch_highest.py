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

lr = 1e-2

# converting np to tensor
X_train = torch.from_numpy(X_train)
y_train = torch.from_numpy(y_train)

# initialize model
model = torch.nn.Sequential(
    torch.nn.Linear(4, 10, bias=False),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 10, bias=False),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 3, bias=False),
    torch.nn.Softmax(dim=1)
)

# initialize loss
loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")

# initialize opt
opt = torch.optim.Adam(model.parameters(), lr=lr)

for _ in range(100):
    # forward pass
    a3 = model(X_train.float())

    # print accuracy
    print(Tensor.eq(a3.argmax(dim=1), y_train).float().mean())

    # get loss
    loss = loss_fn(a3, y_train.long())

    # get gradients
    model.zero_grad()
    loss.backward()

    # use opt to update weights
    opt.step()
