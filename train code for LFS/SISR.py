from __future__ import print_function, division

import torch
import torch.nn as nn
from torch.autograd import Variable
import os,argparse
from dataset import SISR
from SISR_net import RFDN

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
cuda_avail = torch.cuda.is_available()





def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = opt.lr * (0.1 ** (epoch // opt.step))
    return lr

#保存模型
def save_models(epoch,model):
    if not os.path.isdir(opt.save_path):
        os.makedirs(opt.save_path)
    savepath = os.path.join(opt.save_path,"model_{}".format(epoch))
    torch.save(model.module.state_dict(),savepath)
    print('checkpoint saved')



def train(opt):
    print(opt)

    trainset = SISR(root=opt.image_path, is_train=True, data_len=None)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.BATCH_SIZE,
                                              shuffle=True, num_workers=8, drop_last=False)


    # 损失函数
    loss_fn = nn.CrossEntropyLoss()
    loss_L1 = nn.L1Loss()
    # DISR
    model = RFDN().to(device)
    model = nn.DataParallel(model)
    # 加载优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=1e-4)

    for epoch in range(opt.epoch):
        model.train()
        SR_loss = 0.0
        for i, (images,HR) in enumerate(trainloader):
            images = Variable(images.to(device))
            HR = Variable(HR.to(device))

            optimizer.zero_grad()
            SR = model(images)
            loss_SISR = loss_L1(SR, HR)
            loss_SISR.backward()

            optimizer.step()
            SR_loss += loss_SISR.item()*images.size(0)

            if i % 100 == 0:
                print("===> Epoch[{}]({}/{}): loss_SISR:{:.10f}".format(epoch, i, len(trainloader),
                                                                    loss_SISR.item()))

        train_loss = SR_loss/len(trainloader)

        if epoch%10==0:
            save_models(epoch,model)
        print("Epoch:{} ,Train_loss:{}".format(epoch,train_loss))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="PyTorch recongnition")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning Rate. Default=0.1")
    parser.add_argument("--BATCH_SIZE", type=int, default=8, help="Training batch size")
    parser.add_argument("--epoch", type=int, default=8, help="Training batch size")
    parser.add_argument("--save_path", type=str, default=" ", help="Saving model path")
    parser.add_argument("--step", type=int, default=30,
                        help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
    parser.add_argument("--image_path", type=str, default=" ", help="Saving model path")
    opt = parser.parse_args()
    train(opt)











