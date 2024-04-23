import torch
import torch.nn as nn

# Set parameters of Genetic_Algorithm_Network
imput_size = 32
layer1_size = 24
layer2_size = 16
layer3_size = 8
Out_size = 4

class Genetic_Algorithm_Network(nn.Module):
    def __init__(self, n_input, n_hidden1, n_hidden2, n_hidden3, n_output, weights):
        super(Genetic_Algorithm_Network, self).__init__()

        self.num_in = n_input
        self.num_1 = n_hidden1
        self.num_2 = n_hidden2
        self.num_3 = n_hidden3
        self.num_out = n_output

        self.fc1 = nn.Linear(n_input, n_hidden1)
        self.fc2 = nn.Linear(n_hidden1, n_hidden2)
        self.fc3 = nn.Linear(n_hidden2, n_hidden3)
        self.out = nn.Linear(n_hidden3, n_output)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.update_weights(weights)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.sigmoid(self.out(x))
        return x

    # 更新权重，不过这里不再是反向传播了，而是直接从表中提取的
    # weight大小为基因长度也就是神经元的权重数量之和
    def update_weights(self, weights):
        weights = torch.FloatTensor(weights)
        with torch.no_grad():
            x = self.num_in * self.num_1
            xx = x + self.num_1
            y = xx + self.num_1 * self.num_2
            yy = y + self.num_2
            z = yy + self.num_2 * self.num_3
            zz = z + self.num_3
            self.fc1.weight.data = weights[0:x].reshape(self.num_1, self.num_in)
            self.fc1.bias.data = weights[x:xx]
            self.fc2.weight.data = weights[xx:y].reshape(self.num_2, self.num_1)
            self.fc2.bias.data = weights[y:yy]
            self.fc3.weight.data = weights[yy:z].reshape(self.num_3, self.num_2)
            self.fc3.bias.data = weights[z:zz]
            self.out.weight.data = weights[zz:zz + self.num_3 * self.num_out].reshape(self.num_out, self.num_3)
            self.out.bias.data = weights[zz + self.num_3 * self.num_out:]

    def predict(self, input):
        input = torch.tensor([input]).float()
        y = self(input)
        return torch.argmax(y, dim=1).tolist()[0]
