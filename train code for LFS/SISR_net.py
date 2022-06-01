import torch
import torch.nn as nn
import block as B
from torch.autograd import Variable
import numpy as np



class RFDN(nn.Module):
    def __init__(self, in_nc=3, nf=50, num_modules=4, out_nc=3, upscale=4):
        super(RFDN, self).__init__()

        self.RFDN1 = RFDN1()
        self.cos = cosine_simi()
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
        out, SI_map, B = self.RFDN1(input)

        SI_map = SI_map.detach()
        B = B.detach()
        out_B1 = self.B1(B)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)

        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4], dim=1))
        out_lr = self.LR_conv(out_B) + B

        map = self.cos(out_lr, SI_map)
        F = map * SI_map + out_lr
        output = self.upsampler1(out_lr)

        return output

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx




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

        valid = Variable(torch.FloatTensor(np.ones((m_batchsize, 1, width, height))).cuda(), requires_grad=False)

        map = valid - output

        return map


