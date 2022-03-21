import torch
from model.resnet18 import Resnet18
from cifar10 import cifar10_data_download
from Randomly_divide_the_dataset import newData

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
data_loader = newData(BATCHSIZE=100, name=classes[0])
device = torch.device("cuda")
net = Resnet18().to(device)
# net = vgg16().to(device)
net.load_state_dict(torch.load("./modelsaved/resnet18_010_20220315.pth"))
net.eval()

with torch.no_grad():
    correct = 0
    total = 0
    for data in data_loader:

        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        # 取得分最高的那个类 (outputs.data的索引号)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    print('测试分类准确率为：%.3f%%' % (100 * correct / total))



