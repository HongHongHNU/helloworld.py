import torch.cuda
from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from time import perf_counter

start = perf_counter()

x = torch.unsqueeze(torch.linspace(-3, 3, 1000), dim=1)
y = x.pow(3) + 0.3 * torch.rand(x.size())


class Net(nn.Module):
    def __init__(self, input_feature, num_hidden, output_feature):
        super(Net, self).__init__()
        self.hidden = nn.Linear(input_feature, num_hidden)
        self.out = nn.Linear(num_hidden, output_feature)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.out(x)
        return x



CUDA = torch.cuda.is_available()
if CUDA:
    net = Net(input_feature=1, num_hidden=100, output_feature=1).cuda()
    inputs = x.cuda()
    target = y.cuda()
else:
    net = Net(1, 20, 1)
    inputs = x
    target = y

optimizer = optim.Adam(net.parameters(), 0.01)
criterion = nn.MSELoss()


def draw(output, loss):
    if CUDA:
        output=output.cpu()
    plt.cla()
    plt.scatter(x.numpy(), y.numpy(),c='b')
    plt.plot(x.numpy(), output.data.numpy(), '-r',lw=5)
    plt.text(0.5,0,"loss is%s" % (loss.item()))
    plt.pause(0.005)


def train(model, criterion, optimizer, epochs):
    for epoch in range(epochs):
        output = model(inputs)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        if epoch % 80 == 0:
            draw(output, loss)
    return model, loss


net, loss = train(net, criterion, optimizer, 10000)

final = perf_counter()
print("loss is", loss.item())
print("all time is %d", final - start)
