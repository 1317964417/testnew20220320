import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch.nn as nn
from attack.cw import cw_attack
from attack.fgsm import fgsm_attack
from attack.pgd import pgd_attack
from model.resnet18 import Resnet18
from Randomly_divide_the_dataset import newData
criterion = nn.CrossEntropyLoss()

device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = Resnet18().to(device)
net.load_state_dict(torch.load("./modelsaved/resnet18_030_20220315.pth"))

classes = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

data_loader = newData(BATCHSIZE=1,name='deer')


def imshow(img, title):
    npimg = img.numpy()
    fig = plt.figure()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.title(title)
    plt.show()

def cv_for_pgd():
    print("True Image & Predicted Label")
    net.eval()
    correct = 0
    total = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)
        images = pgd_attack(net,images,labels,0.3,0.03,10)
        outputs = net(images)
        _, pre = torch.max(outputs.data, 1)

        total += 1
        correct += (pre == labels).sum()
        imshow(torchvision.utils.make_grid(images.cpu().data, normalize=True),[classes[i] for i in pre])
        print('累计成功率Accuracy of test text: %f %%' % (100 * float(correct) / total))

def cv_for_fgsm():
    print("True Image & Predicted Label")
    net.eval()
    correct = 0
    total = 0
    for images, labels in data_loader:
        # 把数据和标签发送到设备
        images = images.to(device)
        labels = labels.to(device)
        # 唤醒FGSM进行攻击
        perturbed_data = fgsm_attack(net,images,labels,eps=0.3)
        # 重新分类受扰乱的图像
        output = net(perturbed_data)
        _, pre = torch.max(output.data, 1)
        total += 1
        correct += (pre == labels).sum()
        imshow(torchvision.utils.make_grid(perturbed_data.cpu().data, normalize=True), [classes[i] for i in pre])
        print('累计成功率Accuracy of test text: %f %%' % (100 * float(correct) / total))

def cv_for_cw():
    print("True Image & Predicted Label")
    net.eval()
    correct = 0
    total = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)
        images = cw_attack(net,images,labels,1)
        outputs = net(images)
        _, pre = torch.max(outputs.data, 1)

        total += 1
        correct += (pre == labels).sum()
        imshow(torchvision.utils.make_grid(images.cpu().data, normalize=True),[classes[i] for i in pre])
        print('累计成功率Accuracy of test text: %f %%' % (100 * float(correct) / total))



if __name__ == "__main__":

    cv_for_pgd()
    # cv_for_fgsm()
    # cv_for_cw()





