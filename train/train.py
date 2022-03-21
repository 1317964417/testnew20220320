from model.resnet18 import Resnet18
from cifar10 import cifar10_data_download
from torch import nn
from torch import optim
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = Resnet18().to(device)
modelsavedpath = "F:/pycharm_project/20220315/modelsaved"

EPOCHS = 30
BATCHSIZE = 128
lr = 0.002
momentum = 0.8
weight_decay=4e-4
data_loader_train, data_loader_test = cifar10_data_download(BATCHSIZE)

classes = ('airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def train():
    print("麻木不仁的我开始新的训练，真烦啊！")
    print("\n首先得保存训练过程中一些参数，真烦啊！，神啊救救我吧！")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    with open("F:/pycharm_project/20220315/log/training/resnet18/resnet18_0.txt", "w") as f1:
        print("\n开始炼丹了，好无聊！")
        for epoch in range(EPOCHS):
            net.train()
            sum_loss = 0.0
            correct = 0.0
            total = 0.0
            print("\n当前跑到第%d个EPOCH"%(epoch+1))
            for index,data in enumerate(data_loader_train,0):
                length = len(data_loader_train)
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                sum_loss = loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += predicted.eq(labels.data).cpu().sum()
                print("===============================================")
                print('[当前Epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                      % (epoch + 1, (index + 1 + epoch * length), sum_loss / (index + 1), 100. * correct / total))
                f1.write('%03d  %05d |Loss: %.03f | Acc: %.3f%% '
                         % (epoch + 1, (index + 1 + epoch * length), sum_loss / (index + 1), 100. * correct / total))
                f1.write('\n')
                f1.flush()
            if epoch==10:
                print('Saving model......')
                torch.save(net.state_dict(), '%s/resnet18_%03d_20220315.pth' % (modelsavedpath, epoch + 1))
            if epoch==20:
                print('Saving model......')
                torch.save(net.state_dict(), '%s/resnet18_%03d_20220315.pth' % (modelsavedpath, epoch + 1))
        print('Saving model......')
        torch.save(net.state_dict(), '%s/resnet18_%03d_20220315.pth' % (modelsavedpath, epoch + 1))


if __name__ == "__main__":

    train()

