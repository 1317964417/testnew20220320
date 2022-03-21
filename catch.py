import os
import csv
import numpy
import torch
from pandas import *
from numpy import *
from model.resnet18 import  Resnet18
from Randomly_divide_the_dataset import newData

from attack.fgsm import fgsm_attack
from attack.pgd import pgd_attack
from attack.cw import cw_attack



def Catch_Cracteristics_in_some_layers(BATCHISIZE,name):
    # 传入数据batchsize和name
    data_loader = newData(BATCHSIZE=BATCHISIZE, name=name)
    for index,data in enumerate(data_loader,0):
        inputs ,labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        # 对当前的inputs进行攻击扰动
        # inputs = fgsm_attack(net,inputs,labels,0.3)
        # inputs = pgd_attack(net,inputs,labels,eps=0.3,alpha=8/255,iters=10)
        # inputs = cw_attack(net,inputs,labels,BATCHSIZE=100)
        # 输出倒数第二层和倒数第一层
        outputs,out5,out2  = net(inputs)
        out2 = out2.data.cpu().detach().numpy()

    return out2

# 按照通数获得目标层中对应通道中的最大平均值和最小平均值
def Cracteristics_to_list(inputs,dataname,layer,flag):
    print("当前输入的维度数：",inputs)
    # 创建三个列表来接收最大平均值和平均值、最小平均值
    max = []
    ave = []
    min = []
    max_average = []
    ave_average = []
    min_average = []
    # 最外层置通道数,第二层置块数
    for channels in range(inputs.shape[1]):
        for blocks in range(inputs.shape[0]):
            a,b = numpy.max(inputs[blocks][channels][:][:]),numpy.min(inputs[blocks][channels][:][:])
            average = numpy.average(inputs[blocks][channels][:][:])
            max.append(a)
            min.append(b)
            ave.append(average)
            # 如此就可以获得其中每一个通道，100个block中分别的最大值和最小值,然后对这200个数值做平均
        max_average.append(mean(max))
        min_average.append(mean(min))
        ave_average.append(mean(ave))
    print(max_average)
    print(min_average)
    print(ave_average)
    # 将max_average和min_average写入对应的csv文件中
    # 创建文件名
    if os.path.isdir("./layersaved/20220316/"+dataname+"/"):
        print("文件夹已经存在")
        pass
    else:
        os.mkdir(path="./layersaved/20220316/"+dataname+"/")
    with open("./layersaved/20220316/"+dataname+"/"+flag+"_%d_layer.csv" % layer,'w',newline='',encoding='utf-8') as f:
        csv_writer = csv.writer(f)
        # 写入文件
        csv_writer.writerow(max_average)
        csv_writer.writerow(ave_average)
        csv_writer.writerow(min_average)
    # 返回max_average和min_average
    return max_average,min_average,ave_average

# 绘图并保存到对应文件夹中









if __name__ == "__main__":

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    net = Resnet18().to(device)
    net.load_state_dict(torch.load("./modelsaved/resnet18_010_20220315.pth"))
    net.eval()
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    flag = ["original","fgsm","pgd","cw"]
    # print(type(classes))
    # <class 'list'>
    # 导入测试图像的数量和类别
    for i in range(10):
        inputs = Catch_Cracteristics_in_some_layers(100,classes[i])
        Cracteristics_to_list(inputs,dataname=classes[i],layer=2,flag=flag[2])








































































































































































#
# data_loader = newData(BATCHSIZE=50, name='airplane')
# # list用来存储数据
# list = []
#
#
# def hook(module, input, output):
#     print(output.data.shape)
#     list.append(output.data.tolist())
#
#
# handle = net.layer4[0].bn_2.register_forward_hook(hook)
#
# for inputs, labels in data_loader:
#     inputs = inputs.to(device)
#     labels = labels.to(device)
#     # inputs = fgsm_attack(net, inputs, labels, eps=0.3)
#
#     outputs = net(inputs)
#
# handle.remove()
# print(len(list))  # 代表样本数量
# print(len(list[0]))  # 代表通道数
# print(len(list[0][0]))  # 代表矩阵池长宽
# data1 = list[0]  # 例如，如果是第一层layer的conv1，它的输出是64*32*32
# total = 0
# total_list = []
# channel_list = []
#
# # 第一层循环遍历通道数
# for channels in range(len(data1[0])):
#     # 第二层循环遍历batch即样本数
#     for batchs in range(len(data1)):
#         # 第三层用来遍历当前通道下的所有行
#         for rows in range(len(data1[0][0])):
#             # 同理第四行也是如此
#             for cols in range(len(data1[0][0])):
#                 total += data1[batchs][channels][rows][cols]
#         total = total / (len(data1[0][0]) * len(data1[0][0]))
#         total_list.append(total)
#         total = 0
#     average_total_list = np.mean(total_list)
#     channel_list.append(average_total_list)
#     average_total_list = 0
#
# print(channel_list)
# test = pd.DataFrame(data=channel_list)
# test.to_csv('./layersaved/20220315/airplane/4.csv', encoding='gbk')
#
# plt.title("output for channels")
# plt.xlabel("channels")
# plt.ylabel("The averagevalue of every channels")
# plt.plot(channel_list)
# plt.show()
#




