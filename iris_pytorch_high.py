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

# convert numpy to Tensor
X_train = torch.from_numpy(X_train)
y_train = torch.from_numpy(y_train)

# initialize model instead of weights
model = torch.nn.Sequential(
    torch.nn.Linear(4, 10, bias=False),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 10, bias=False),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 3, bias=False),
    torch.nn.Softmax(dim=1)
)

# can custom initialize weights as well
for param in model.parameters():
    torch.nn.init.xavier_normal_(param.data)

# print(param)
# print(model[-2].weight, model[-2].bias)
# assert param is model[-2].bias

loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

for _ in range(500):
    # feed forward
    probs = model(X_train.float())

    # print acc
    print(Tensor.eq(probs.argmax(dim=1), y_train).float().mean())

    # get loss and acc
    loss = loss_fn(probs, y_train.long())

    # get gradients
    # model.zero_grad()
    loss.backward()

    # update weights
    with torch.no_grad():
        for param in model.parameters():
            param.data -= lr*param.grad
            param.grad.zero_()
