import torch
import torch.nn as nn
from torch import optim
from model.resnet18 import Resnet18
from cifar10 import cifar10_data_download

"""
ϵ（epsilon）的值通常是人为设定 ，可以理解为学习率，一旦扰动值超出阈值，该对抗样本会被人眼识别。
"""
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fgsm_attack(model,images,labels,eps):
    criterion = nn.CrossEntropyLoss()
    images = images.to(device)
    labels = labels.to(device)
    # 设置张量的requires_grad属性，这对于攻击很关键
    images.requires_grad = True
    # 通过模型前向传递数据
    outputs,out5,out2 = model(images)
    # _,init_pred = outputs.max(1, keepdim=True)  # get the index of the max log-probability
    # # 如果初始预测是错误的，不打断攻击，继续
    # if init_pred.item() != labels.item():
    #     print("如果初始预测是错误的，不打断攻击，继续")
    # 计算损失
    loss = criterion(outputs, labels)
    # 将所有现有的渐变归零
    model.zero_grad()
    # 计算后向传递模型的梯度
    loss.backward()
    # 收集datagrad
    data_grad = images.grad.data
    # 使用sign（符号）函数，将对x求了偏导的梯度进行符号化
    sign_data_grad = data_grad.sign()
    # 通过epsilon生成对抗样本
    perturbed_image = images + eps * sign_data_grad
    # 做一个剪裁的工作，将torch.clamp内部大于1的数值变为1，小于0的数值等于0，防止image越界
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # 返回对抗样本
    return perturbed_image

def train():
    print("麻木不仁的我开始新的训练，真烦啊！")
    print("\n首先得保存训练过程中一些参数，真烦啊！，神啊救救我吧！")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    with open("F:/pycharm_project/20220316/log/training/resnet18/resnet18_1.txt", "w") as f1:
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
                inputs = fgsm_attack(net,inputs,labels,eps=0.3)
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
                torch.save(net.state_dict(), '%s/resnet18_%03d_20220316fgsm.pth' % (modelsavedpath, epoch + 1))
            if epoch==20:
                print('Saving model......')
                torch.save(net.state_dict(), '%s/resnet18_%03d_20220315fgsm.pth' % (modelsavedpath, epoch + 1))
        print('Saving model......')
        torch.save(net.state_dict(), '%s/resnet18_%03d_20220315fgsm.pth' % (modelsavedpath, epoch + 1))

if __name__ == "__main__":
    net = Resnet18().to(device)
    modelsavedpath = "F:/pycharm_project/20220315/modelsaved"

    EPOCHS = 30
    BATCHSIZE = 128
    lr = 0.002
    momentum = 0.8
    weight_decay = 4e-4
    data_loader_train, data_loader_test = cifar10_data_download(BATCHSIZE)

    classes = ('airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    train()