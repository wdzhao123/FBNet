from __future__ import print_function, division
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import os,argparse
from torch.optim import Adam
from resnet50 import resnet50,CNN,Bottleneck
from dataset import SISR
from SISR_with_gradcam_net import SI_help_res, RFDN2, RFDN1
from Grad_cam import GradCam


os.environ['CUDA_VISIBLE_DEVICES']='0'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#加载损失函数
def Loss(LR,HR,map,batch_size):
    loss = torch.mul(map*100.0, torch.abs(torch.sub(LR, HR)))
    loss = torch.sum(loss) / (batch_size * 128 * 256)

    return loss


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    # lr = opt.lr * (0.1 ** (epoch // opt.step))
    if epoch>200:
        lr = 1/2*opt.lr
    return lr

#写出函数的保存模型和评估模型
def save_models(epoch,L2S,opt):
    if not os.path.isdir(opt.save_path):
        os.makedirs(opt.save_path)
    savepath = os.path.join(opt.save_path,"model_{}".format(epoch))
    torch.save(L2S.module.state_dict(),savepath)
    print('checkpoint saved')

def train(opt):
    print(opt)
    trainset = SISR(root=opt.image_path, is_train=True, data_len=None)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.BATCH_SIZE,
                                              shuffle=False, num_workers=8, drop_last=True)

    ## SISR
    L2S = RFDN2().to(device)
    L2S.load_state_dict(torch.load(opt.SISR_param),strict=False)
    L2S = nn.DataParallel(L2S)

    AuliNet = SI_help_res(Bottleneck, [3, 4, 6, 3]).to(device)
    AuliNet.load_state_dict(torch.load(opt.AuliNet_param), strict=False)
    grad_cam = GradCam(model=AuliNet, feature_module=AuliNet.layer4,target_layer_names=["2"], use_cuda=True)

    # 加载优化器
    optimizer = Adam(L2S.parameters(), lr=0.0001, betas=(0.9, 0.99), eps=1e-08, weight_decay=0.0001)

    for epoch in range(0,opt.epoch):
        L2S.train()
        # L2S1.eval()
        AuliNet.eval()
        SR_loss = 0.0
        for i, (images,HR) in enumerate(trainloader):
            # if cuda_avail:
            images = Variable(images.to(device))
            HR = Variable(HR.to(device))

            optimizer.zero_grad()
            SR,_,_ = L2S(images)

            target_index = None
            _,_,F = AuliNet(images)
            map = grad_cam(F,target_index)

            loss_SISR = Loss(SR, HR, map, opt.BATCH_SIZE)
            loss_SISR.backward()
            optimizer.step()

            SR_loss += loss_SISR.item()*images.size(0)

            if i % 100 == 0:
                print("===> Epoch[{}]({}/{}): loss_SISR:{:.10f}".format(epoch, i, len(trainloader),
                                                                    loss_SISR.item()))

        train_loss = SR_loss/len(trainloader)
        if epoch%10==0:
            save_models(epoch,L2S,opt)
        print("Epoch:{} ,Train_loss:{}".format(epoch,train_loss))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch recongnition")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning Rate. Default=0.1")
    parser.add_argument("--BATCH_SIZE", type=int, default=8, help="Training batch size")
    parser.add_argument("--epoch", type=int, default= 200, help="Training batch size")
    parser.add_argument("--save_path", type=str, default="",
                        help="Sets the save path of model param")
    parser.add_argument("--image_path", type=str, default="",
                        help="Sets the image path")
    parser.add_argument("--AuliNet_param", type=str, default="",
                        help="The last iteration parameters of the classification model")
    parser.add_argument("--SISR_param", type=str, default="",
                        help="The param of SISR model")
    parser.add_argument("--step", type=int, default=10,
                        help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
    opt = parser.parse_args()
    train(opt)











