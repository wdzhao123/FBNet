import torch
import torch.nn as nn
import block as B
from torch.autograd import Variable
import numpy as np

####transfer######

class ResBlock(nn.Module):
    """
    Want transfer road map to resnet

    Basic residual block for transfer.

    Parameters
    ---
    n_filters : int, optional
        a number of filters.
    """

    def __init__(self, n_filters=64):
        super(ResBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_filters, n_filters, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(n_filters, n_filters, 3, 1, 1),
        )

    def forward(self, x):
        return self.body(x) + x



######resnet#########
class SI_help_res(nn.Module):

    def __init__(self, block, layers,  num_classes=12,nf=50, num_modules=4, out_nc=3):#
        self.inplanes = 64
        super(SI_help_res, self).__init__()
        self.RFDN1 = RFDN1()

        self.B1 = B.RFDB(in_channels=nf)
        self.B2 = B.RFDB(in_channels=nf)
        self.B3 = B.RFDB(in_channels=nf)
        self.B4 = B.RFDB(in_channels=nf)
        self.c = B.conv_block(nf * num_modules, nf, kernel_size=1, act_type='lrelu')

        self.LR_conv = B.conv_layer(nf, nf, kernel_size=3)

        upsample_block = B.pixelshuffle_block1
        self.upsampler = upsample_block(nf, out_nc, upscale_factor=4)
        self.upsampler1 = upsample_block(nf, 1, upscale_factor=4)
        self.scale_idx = 0

        self.cos = cosine_simi()

        #resnet
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
        #                        bias=False)
        # self.conv2 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
        #                        bias=False)
        self.conv3 = nn.Conv2d(50, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # 全局平均池化
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        # 去掉原来的fc层，新增一个fclass层
        self.fclass = nn.Linear(2048, num_classes)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, input):

        out,SI_map,B = self.RFDN1(input)
        SI_map = SI_map.detach()
        B = B.detach()

        out_B1 = self.B1(B)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)

        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4], dim=1))
        out_lr = self.LR_conv(out_B) + B  # out_fea
        map = self.cos(out_lr, SI_map)
        F = map * SI_map + out_lr

        output = self.upsampler1(out_lr)

        x = self.conv3(F)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.global_pooling(x)
        # 新加层的forward
        x = x.view(x.size(0), -1)
        x = self.fclass(x)


        return  x,output,F


class RFDN1(nn.Module):
    def __init__(self, in_nc=3, nf=50, num_modules=4, out_nc=3, upscale=4):
        super(RFDN1, self).__init__()

        # self.fea_conv = B.conv_layer(in_nc, nf, kernel_size=3)
        self.fea_conv1 = B.conv_layer(1, nf, kernel_size=3)
        self.B1 = B.RFDB(in_channels=nf)
        self.B2 = B.RFDB(in_channels=nf)
        self.B3 = B.RFDB(in_channels=nf)
        self.B4 = B.RFDB(in_channels=nf)
        self.c = B.conv_block(nf * num_modules, nf, kernel_size=1, act_type='lrelu')

        self.LR_conv = B.conv_layer(nf, nf, kernel_size=3)

        upsample_block = B.pixelshuffle_block1
        self.upsampler = upsample_block(nf, out_nc, upscale_factor=4)
        self.upsampler1 = upsample_block(nf, 1, upscale_factor=4)
        self.scale_idx = 0


    def forward(self, input):
        out_fea = self.fea_conv1(input)

        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)

        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4], dim=1))
        out_lr = self.LR_conv(out_B) + out_fea

        output = self.upsampler1(out_lr)

        return output,out_lr,out_fea

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx


class cosine_simi(nn.Module):

    def __init__(self):
        super(cosine_simi, self).__init__()

    def forward(self,x,y):
        m_batchsize, C, width, height = x.size()
        output = torch.cosine_similarity(x, y, dim=1)
        output = output.view([m_batchsize, 1, width, height])
        output = torch.sigmoid(output)
        # mid1 = self.display(output, tag='map')
        valid = Variable(torch.FloatTensor(np.ones((m_batchsize, 1, width, height))).cuda(), requires_grad=False)
        map = valid - output

        return map



class RFDN2(nn.Module):
    def __init__(self, in_nc=3, nf=50, num_modules=4, out_nc=3, upscale=4):
        super(RFDN2, self).__init__()

        # self.fea_conv = B.conv_layer(in_nc, nf, kernel_size=3)
        self.fea_conv1 = B.conv_layer(1, nf, kernel_size=3)
        self.B1 = B.RFDB(in_channels=nf)
        self.B2 = B.RFDB(in_channels=nf)
        self.B3 = B.RFDB(in_channels=nf)
        self.B4 = B.RFDB(in_channels=nf)
        self.c = B.conv_block(nf * num_modules, nf, kernel_size=1, act_type='lrelu')

        self.LR_conv = B.conv_layer(nf, nf, kernel_size=3)

        upsample_block = B.pixelshuffle_block1
        self.upsampler = upsample_block(nf, out_nc, upscale_factor=4)
        self.upsampler1 = upsample_block(nf, 1, upscale_factor=4)
        self.scale_idx = 0

    def forward(self, input):
        out_fea = self.fea_conv1(input)

        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)

        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4], dim=1))
        out_lr = self.LR_conv(out_B) + out_fea
        # mid = self.display(out_lr,'F')
        output = self.upsampler1(out_lr)

        return output, out_lr, out_fea

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx

