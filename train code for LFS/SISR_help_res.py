from __future__ import print_function, division
import torch
import torch.nn as nn
from torch.autograd import Variable
import os,argparse
from resnet50 import Bottleneck
from dataset import Ship
from SISR_help_res_net import SI_help_res

os.environ['CUDA_VISIBLE_DEVICES']='1'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
cuda_avail = torch.cuda.is_available()


#加载损失函数
loss_fn = nn.CrossEntropyLoss()
loss_map = nn.L1Loss()

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = opt.lr * (0.1 ** (epoch // opt.step))
    return lr

#写出函数的保存模型和评估模型
def save_models(epoch,AuliNet):
    if not os.path.isdir(opt.save_path):
        os.makedirs(opt.save_path)
    savepath = os.path.join(opt.save_path,"model_{}".format(epoch))
    torch.save(AuliNet.module.state_dict(),savepath)
    print('checkpoint saved')

#测试总共的准确度
def test(AuliNet,testloader):
    test_acc = 0.0
    AuliNet.eval()

    for i, (images,labels) in enumerate(testloader):

        if cuda_avail:
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

        outputs,yuan = AuliNet(images)
        _,prediction = torch.max(outputs.data,1)
        test_acc = test_acc + torch.sum(prediction==labels.data)
    test_acc = test_acc/len(testloader)

    return test_acc

def train(opt):
    best_acc =0.0
    print(opt)
    f = open('record.txt','a')
    ## resnet with SISR

    trainset = Ship(root=opt.image_path, is_train=True, data_len=None)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.BATCH_SIZE,
                                              shuffle=True, num_workers=8, drop_last=False)
    testset = Ship(root=opt.image_path, is_train=False, data_len=None)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                             shuffle=False, num_workers=8, drop_last=False)

    AuliNet = SI_help_res(opt,Bottleneck, [3, 4, 6, 3]).to(device)
    AuliNet = nn.DataParallel(AuliNet)

    # 加载优化器
    optimizer = torch.optim.SGD(AuliNet.parameters(), lr=opt.lr, momentum=0.9, weight_decay=1e-4)

    for epoch in range(0,opt.epochs):
        AuliNet.train()

        train_acc = 0.0
        train_loss = 0.0
        refactor_loss = 0.0

        for i, (images,labels) in enumerate(trainloader):

            if cuda_avail:
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())

            optimizer.zero_grad()
            outputs,yuan = AuliNet(images)
            loss = loss_fn(outputs,labels)
            loss_r = loss_map(yuan,images)

            loss_all = loss + loss_r
            loss_all.backward()

            optimizer.step()

            train_loss += loss.item()*images.size(0)
            refactor_loss += loss_r.item()*images.size(0)

            _, prediction = torch.max(outputs.data,1)
            train_acc += torch.sum(prediction == labels.data)

        train_acc = train_acc / opt.train_num
        train_loss = train_loss / opt.train_num
        refactor_loss = refactor_loss / opt.train_num


        test_acc = test(AuliNet,testloader)
        save_models(epoch,AuliNet)

        f.write("epoch"+":"+str(epoch)+","+"train_acc"+":"+str(train_acc)+','+"test_acc"+":"+str(test_acc)+"\n")
        print("Epoch:{} ,Train_loss:{},Refactor_loss:{},Train_acc:{},test_acc:{}".format(epoch,train_loss,refactor_loss,train_acc,test_acc))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch recongnition")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning Rate")
    parser.add_argument("--BATCH_SIZE", type=int, default=8, help="Training batch size")
    parser.add_argument("--epoch", type=int, default=8, help="Training batch size")
    parser.add_argument("--save_path", type=int, default="",
                        help="Sets the save path of model param")
    parser.add_argument("--image_path", type=int, default="",
                        help="Sets the image path")
    parser.add_argument("--SISR_path", type=int, default="",
                        help="Sets the image path")
    parser.add_argument("--train_num", type=int, default=8, help="number of train set")
    parser.add_argument("--step", type=int, default=30,
                        help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
    opt = parser.parse_args()
    train(opt)