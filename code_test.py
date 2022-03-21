# -*- coding: utf-8 -*-
# 测试代码
# 遍历递归实现遍历所有文件夹
import csv
import matplotlib.pyplot as plt

def func():
    with open("F:/pycharm_project/20220315/layersaved/20220316/airplane/fgsm_1_layer.csv",'r+') as csv_file:
         reader = csv.reader(csv_file)
         print(reader)
         for row in reader:
             row = list(map(float, row))
             print(row)



if __name__ == "__main__":
    func()
