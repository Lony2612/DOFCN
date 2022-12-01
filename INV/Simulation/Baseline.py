import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from utils import load_preprocess

import numpy as np
import torch.nn as nn
import torch
import torch.utils.data as Data
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torch.nn import functional as F


class BaselineNet(nn.Module):
    def __init__(self):
        super(BaselineNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=128, kernel_size=3, padding=0, bias=False)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=0, bias=False)
        
        self.fc1 = nn.Linear(15360, 1024)
        self.fc2 = nn.Linear(1024, 4)        
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        x = x.view(-1, 15360)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


dataset = load_preprocess('./dataset/dataset.csv', mode='S')
np.random.seed(3090)
indices = np.random.permutation(dataset.shape[0])
train_idx = int(0.8 * dataset.shape[0])
dataset = dataset[indices, :]

train_dataset = dataset[:train_idx, :]
test_dataset = dataset[train_idx:, :]

criterion = nn.MSELoss()
batch_size = 32  # training batch size
lr = 3e-5

model = BaselineNet()
path = './models/Baseline.pth.tar'
model.cuda()


def train(model, criterion, batch_size, path, lr=3e-5):
    nb_epoch = 200     # number of epochs to train on
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    torch.manual_seed(2025)
    X_train = train_dataset[:, 4:]
    Y_train = train_dataset[:, :4]
    X_train = np.reshape(X_train, [-1, 2, 64])
    X_train = torch.tensor(torch.from_numpy(X_train), dtype=torch.float32)
    Y_train = torch.tensor(torch.from_numpy(Y_train), dtype=torch.float32)
    dataset = Data.TensorDataset(X_train, Y_train)
    loader = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    
    X_test = test_dataset[:, 4:]
    Y_test = test_dataset[:, :4]
    X_test = np.reshape(X_test, [-1, 2, 64])
    X_test = torch.tensor(torch.from_numpy(X_test), dtype=torch.float32)
    Y_test = torch.tensor(torch.from_numpy(Y_test), dtype=torch.float32)
    X_test, Y_test = X_test.cuda(), Y_test.cuda()
    print('Begin!!!')
    try:
        for epoch in range(0, nb_epoch):
            running_loss = 0.0
            for i, data in enumerate(loader):
                inputs, labels = data
                inputs, labels = Variable(inputs), Variable(labels)
                inputs, labels = inputs.cuda(), labels.cuda()
                
                # forward+backward
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 10 == 9:
                    print(str([lr])+str([epoch+1, i+1])+' loss:'+str(running_loss/10))
                    running_loss = 0.0
                
    except KeyboardInterrupt:
        save_model(path, model)
    print('Done!')
    save_model(path, model)

def test(model, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    X_test = test_dataset[:, 4:]
    Y_test = test_dataset[:, :4]
    X_test = np.reshape(X_test, [-1, 2, 64])
    X_test = torch.tensor(torch.from_numpy(X_test), dtype=torch.float32)
    X_test = X_test.cuda()
    Y_hat  = model(X_test).cpu().detach().numpy()
    for ii in range(Y_hat.shape[0]):
        lbl = Y_test[ii,:]
        pre = Y_hat[ii,:]
        plt.figure()
        plt.plot(range(4), lbl[:], "r")
        plt.plot(range(4), pre[:], "r:")
        plt.xlabel("wavelength")
        plt.ylabel("Intensity")
        plt.title("Chiral Material")
        plt.savefig("./img/%d.png"%ii)
        plt.close()
        plt.clf()


def save_model(path, model):
    torch.save({'state_dict': model.state_dict()}, path)
            

if __name__ == "__main__":
    train(model, criterion, batch_size, path, lr)
    print("I'm done!")