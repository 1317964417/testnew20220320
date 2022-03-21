import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def cifar10_data_download(batchsize):
    # 训练集数据变换
    trainset_transform = transforms.Compose([
        transforms.ToTensor(),
        # 归一化处理
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # 测试集数据变化
    testset_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # 数据加载
    data_train = torchvision.datasets.CIFAR10(root='F:/pycharm_project/20220315/data/cifar10/',
                                              train=True,
                                              transform=trainset_transform,
                                              download=True)
    data_test = torchvision.datasets.CIFAR10(root='F:/pycharm_project/20220315/data/cifar10/',
                                              train=False,
                                              transform=testset_transform,
                                              download=True)
    # 数据装载和数据预览
    data_loader_train = DataLoader(dataset=data_train,
                                   batch_size=batchsize,
                                   shuffle=True,
                                   )
    data_loader_test = DataLoader(dataset=data_test,
                                  batch_size=batchsize,
                                  shuffle=False,
                                  )
    return data_loader_train, data_loader_test


if __name__ == "__main__":

    cifar10_data_download(128)




