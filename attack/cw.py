import torch
from cleverhans.torch.attacks.carlini_wagner_l2 import carlini_wagner_l2
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def cw_attack(model,images,labels,BATCHSIZE):

    adv_image = carlini_wagner_l2(model, images, n_classes=10, y=torch.tensor([labels] * BATCHSIZE, device=device),
                                targeted=False)
    return adv_image

