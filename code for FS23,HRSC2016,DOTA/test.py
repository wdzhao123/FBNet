from __future__ import print_function, division

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import os,argparse
from PIL import Image

from resnet50 import  Bottleneck
from dataset import SHIP_FG
from torchvision import transforms
from torchvision.utils import save_image
from test_net import SI_help_res, SI_help_res_copy
import cv2

os.environ['CUDA_VISIBLE_DEVICES']='1'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description="PyTorch recongnition")
parser.add_argument("--lr", type=float, default=0.0001, help="Learning Rate. Default=0.1")
parser.add_argument("--BATCH_SIZE", type=int, default=8, help="Training batch size")
# parser.add_argument("--", type=int, default=100, help="Training epoch")
parser.add_argument("--step", type=int, default=30, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
opt = parser.parse_args()

# epoch = 10
testset = SHIP_FG(root='./Images', is_train=False, data_len=None)
testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                         shuffle=False, num_workers=8, drop_last=False)

cuda_avail = torch.cuda.is_available()

AuliNet = SI_help_res_copy(Bottleneck, [3, 4, 6, 3]).to(device)
AuliNet.load_state_dict(torch.load('./分类网络模型参数'))

AuliNet = nn.DataParallel(AuliNet)

#加载优化器
optimizer = torch.optim.SGD(AuliNet.parameters(), lr=opt.lr, momentum=0.9, weight_decay=1e-4)

#测试每个文件夹的准确度
def test_dir():
    test_acc = 0.0
    AuliNet.eval()
    m = 0
    num = 0
    all = 0
    for i, (images, labels) in enumerate(testloader):
        if cuda_avail:
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

        outputs,yuan= AuliNet(images)
        _, prediction = torch.max(outputs.data, 1)
        all = all + torch.sum(prediction==labels.data)
        a = torch.from_numpy(np.array([m]))
        a = Variable(a.cuda())
        if a == labels:
            num += 1
            test_acc = test_acc + torch.sum(prediction == labels.data)
            Accu = test_acc.item() / num
        if a != labels:
            num = 1
            m += 1
            f = open('Accuracy.txt', 'a')
            f.write(str(m - 1) + ":" + str(Accu) + "\n")
            test_acc = 1

    f = open('Accuracy.txt', 'a')
    f.write(str(m) + ":" + str(Accu) + "\n")
    f.write("测试集整体准确度:" + str(all.item()/len(testloader)) + "\n")
    return test_acc

def test_pic():
    AuliNet.eval()
    image = readImage('./7_4_38_11762.jpg',224)
    image = image.view(1, *image.size())
    images = Variable(image.cuda())
    outputs, yuan = AuliNet(images)
    _, prediction = torch.max(outputs.data, 1)
    print("预测类别是:", prediction.item())


def readImage(path='../mode.jpg',size=224):
    img = cv2.imread(path)
    # if len(img.shape) == 2:
    #     img = np.stack([img] * 3, 2)
    img = Image.fromarray(img, mode='RGB')
    transform1 = transforms.Compose([
        transforms.Resize((size, size), Image.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    mode = transform1(img)
    return mode



if __name__ == "__main__":
    test_dir() #预测测试集中所有类别的准确度
    # test_pic()  #预测单张图片的准确度









