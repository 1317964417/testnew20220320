import torch
import torch.nn as nn
"""
暂时没啥想说的，不过后面还是要把参数统一一下
"""
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def pgd_attack(model, images, labels, eps, alpha, iters):
    criterion = nn.CrossEntropyLoss()
    # 将输入和标签转换为GPU可用的类型
    images = images.to(device)
    labels = labels.to(device)
    loss = criterion
    ori_images  = images.data
    for i in range(iters):
        images.requires_grad = True
        outputs,out5,out2 = model(images)
        model.zero_grad()
        cost = loss(outputs,labels).to(device)
        cost.backward()
        adv_images = images + alpha * images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()
    return images
