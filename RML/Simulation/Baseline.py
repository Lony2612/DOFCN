import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import torch
from torch.autograd import Variable as V
import numpy as np
import matplotlib.pyplot as plt
from torch.utils import data
import pickle as pk
import torch.optim as optim
import torch.utils.data as Data
from torch import nn
import math


class BaselineNet(nn.Module):
    def __init__(self):
        super(BaselineNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=(1,3), padding=(2,0))
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=80, kernel_size=(2,3), padding=(2,0))
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
 
        self.fc1 = nn.Linear(89280, 256)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 10)
        self.log_softmax = nn.LogSoftmax(dim=1)
 
    def forward(self, x):        
        x = self.dropout1(self.relu1(self.conv1(x)))
        
        x = self.dropout2(self.relu2(self.conv2(x)))
        
        x = x.view(-1, 89280)
        x = self.dropout3(self.relu3(self.fc1(x)))
        x = self.log_softmax(self.fc2(x))
        return x
    

###############Data######################################
f = open('RML2016.10b.dat', 'rb')
data = pk.load(f, encoding='latin1')

global snrs
snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], data.keys())))), [1,0])
X = []  
lbl = []
for mod in mods:
    for snr in snrs:
        X.append(data[(mod, snr)])
        for i in range(data[(mod, snr)].shape[0]):
            lbl.append((mod, snr))
X = np.vstack(X)

X = (X - np.mean(X)) / np.std(X)

n_examples = X.shape[0]
n_train = n_examples * 0.8
train_idx = np.random.choice(range(0, int(n_examples)),
								size=int(n_train), replace=False)

test_idx = list(set(range(0, n_examples))-set(train_idx))
X_train = X[train_idx]
X_test = X[test_idx]


def to_onehot(yy):
    yy1 = np.zeros([len(yy), max(yy)+1])
    yy1[np.arange(len(yy)),yy] = 1
    return yy1


Y_train = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), train_idx)))
Y_test = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), test_idx)))

in_shp = list(X_train.shape[1:])  #(2,128)
classes = mods
X_train = torch.from_numpy(X_train)
X_train = X_train.unsqueeze(1)
Y_train = torch.from_numpy(Y_train)
X_test = torch.from_numpy(X_test)
X_test = X_test.unsqueeze(1)
Y_test = torch.from_numpy(Y_test)

############### Net ####################################
model = BaselineNet()
model = model.cuda()
    
criterion = nn.CrossEntropyLoss()
batch_size =  1024 # training batch size

def train(model_path='./models/mynet_Base.pth.tar', \
          loss_path='./logs/Base_train_loss.txt',\
          acc_path='./logs/Base_train_acc.txt'):
    
    acc_plot = []
    loss_plot = []
    nb_epoch = 40     # number of epochs to train on
    
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    torch.manual_seed(2025)
    label = torch.argmax(Y_train, -1)  # One-hot to label
    dataset = Data.TensorDataset(X_train, label)  # TensorData Type
    loader = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    print('Begin!!!')
    try:
        for epoch in range(0, nb_epoch):
            running_loss = 0.0
            for i, data in enumerate(loader):
                inputs, labels = data
                inputs, labels = V(inputs), V(labels)
                inputs, labels = inputs.cuda(), labels.cuda()
                
                # forward+backward
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss_plot.append(loss)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 100 == 0:
                    print(str([epoch+1, i+1])+' loss:'+str(running_loss/100))
                    _, pred = torch.max(outputs, 1)
                    num_correct = (pred == labels).sum()
                    acc = int(num_correct) / batch_size
                    acc_plot.append(acc)
                    print('Acc:'+str(acc))
                    acc = 0.0
                    running_loss = 0.0
            if epoch % 5 == 4:
                torch.save({'state_dict': model.state_dict()}, model_path)
    except KeyboardInterrupt:
        save_model(model, model_path, loss_plot, loss_path, acc_plot, acc_path)
    save_model(model, model_path, loss_plot, loss_path, acc_plot, acc_path)
    print('Train Done!')


def save_model(model, model_path, loss_plot, loss_path, acc_plot, acc_path):
    torch.save({'state_dict': model.state_dict()}, model_path)
    f = open(acc_path, 'w')
    for i in acc_plot:
        f.write(str(i)+'\n')
    f.close()
    f = open(loss_path,'w')
    for i in loss_plot:
        f.write(str(i)+'\n')
    f.close()

def test(model_path='./models/mynet_Base.pth.tar', \
         acc_path='./logs/Base_test_acc.txt', 
         plot_confused_matrix=False):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    def plot_confusion_matrix(filepath, cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=[]):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels, rotation=45)
        plt.yticks(tick_marks, labels)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(filepath)
        plt.close()

    # Plot confusion matrix 画图
    acc = {}
    for snr in snrs:
        # extract classes @ SNR
        test_SNRs = list(map(lambda x: lbl[x][1], test_idx))
        test_X_i = X_test[np.where(np.array(test_SNRs)==snr)]
        test_Y_i = Y_test[np.where(np.array(test_SNRs)==snr)]    

        # estimate classes
        test_X_size = test_X_i.shape[0]
        seg_num = 10
        seg_len = math.ceil(test_X_size/seg_num)
        
        conf = np.zeros([len(classes),len(classes)])
        confnorm = np.zeros([len(classes),len(classes)])
        for jj in range(seg_num):
            test_X_i_j = test_X_i[jj*seg_len:(jj+1)*seg_len,:,:]
            test_Y_i_j = test_Y_i[jj*seg_len:(jj+1)*seg_len,:]
            test_Y_i_hat = model(test_X_i_j.cuda())
            
            for i in range(0,test_X_i_j.shape[0]):
                j = list(test_Y_i_j[i,:]).index(1)
                k = int(np.argmax(test_Y_i_hat.cpu().detach().numpy()[i,:]))
                conf[j,k] = conf[j,k] + 1
            for i in range(0,len(classes)):
                confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
        if plot_confused_matrix:
            plt.figure()
            plot_confusion_matrix("./img/snr_%d.png"%snr, confnorm, labels=classes, title="ConvNet Confusion Matrix (SNR=%d)"%(snr))
        
        cor = np.sum(np.diag(conf))
        ncor = np.sum(conf) - cor
        print ("SNR: %3d"%snr, "Overall Accuracy: ", cor / (cor+ncor))
        acc[snr] = 1.0*cor/(cor+ncor)
    
    f = open(acc_path, 'w')
    for key in acc.keys():
        f.write("%3d\t%.6f\n"%(key, acc[key]))
    f.close()
    # Plot accuracy curve
    plt.figure()
    plt.plot(snrs, list(map(lambda x: acc[x], snrs)))
    plt.xlabel("Signal to Noise Ratio")
    plt.ylabel("Classification Accuracy")
    plt.title("Classification Accuracy on RadioML 2016.10 Beta")
    plt.close()
    
    
if __name__ == "__main__":
    # train()
    test()
    print("done!")