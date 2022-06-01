import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn



class FeatureExtractor():
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        for name, module in self.model._modules.items():
            if module == self.feature_module:
                target_activations, x = self.feature_extractor(x)
            elif "global_pooling" in name.lower():
                x = module(x)
                x = x.view(x.size(0), -1)
            elif name in ['RFDN1','B1','B2','B3','B4','c','LR_conv','upsampler','upsampler1','scale_idx','conv2', 'cos']:
                continue
            else:
                # print('ttt:', name)
                x = module(x)

        return target_activations, x


def preprocess_image(preprocessed_img):
    # preprocessed_img = img.copy()[:, :, ::-1]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))

    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = preprocessed_img.requires_grad_(True)
    return input


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # .cpu().detach().numpy()
    cam = heatmap * 0.2 + img  # .reshape([256,128,1])
    cam = cam / np.max(cam)
    cv2.imwrite("cam.jpg", np.uint8(255 * cam))


###
class GradCam:
    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        m_batchsize, C, width, height = input.size()
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index == None:
            one_hot = np.zeros((m_batchsize,12), dtype=np.float32)
            for i in range(m_batchsize):
                m = output[i].reshape(1,12)
                index = np.argmax(m.detach().cpu().numpy())
                # one_hot[i] = np.zeros((1, output[i].size()[-1]), dtype=np.float32)
                one_hot[i][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1]  # .cpu().data.numpy()

        target = features[-1]
        # target = target.cpu().data.numpy()[0, :]
        weights = torch.mean(grads_val, dim=(2, 3), keepdim=False, out=None)
        weights = weights.reshape([m_batchsize,2048, 1, 1])

        cam = weights * target
        cam = torch.sum(cam, dim=1)
        cam = cam.reshape([m_batchsize, 1, 4, 8])



        # 增加的relu和min—max normalize，这里resize应该在relu之后
        Fc = nn.ReLU(inplace=True)
        cam = Fc(cam)
        cam = F.interpolate(cam, scale_factor=32, mode='bicubic')

        max = torch.max(cam[0])
        min = torch.min(cam[0])
        map1 = cam[0]
        # b,c = map.shape
        # result = np.zeros(shape=(b,c))
        result1 = (map1 - min) / (max - min + 1e-9)
        result1 = result1.view([1, 1, width, height])

        for i in range(1, m_batchsize):
            max = torch.max(cam[i])
            min = torch.min(cam[i])

            map = cam[i]
            # b,c = map.shape
            # result = np.zeros(shape=(b,c))
            result = (map - min) / (max - min + 1e-9)
            result = result.view([1, 1, width, height])
            result1 = torch.cat([result1, result], dim=0)

        # m = self.display(result1,tag='cam1')
        return result1

    def display(self,x, tag):

        m_batchsize, C, width, height = x.size()
        x = x.view([m_batchsize, width, height])

        for i in range(m_batchsize):
            # max = torch.max(x[i]).detach().cpu().numpy()
            # min = torch.min(x[i]).detach().cpu().numpy()

            cam = x[i].detach().cpu().numpy()
            # cam = np.maximum(cam, 0)
            # # cam = cv2.resize(cam, (height, width))  # input.shape[2:]
            # cam = cam - np.min(cam)
            # cam = cam / np.max(cam)
            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            # cv2.imwrite('./some_result/{}/mask_mid{}{}.tif'.format(tag1, tag2, i + 1), heatmap)
            cv2.imwrite('/media/ttt/data/SISR_HELP_CLASSIFY/some_result/feature/gradcam/{}{}.tif'.format(tag, i + 1),heatmap)
            # cv2.imwrite('/media/ttt/data/SISR_HELP_CLASSIFY/some_result/feature/SISR/gradcam/{}.tif'.format(tag),
            #             heatmap)

###第三层加入relu后的
# class GradCam1:
#     def __init__(self, model, feature_module, target_layer_names, use_cuda):
#         self.model = model
#         self.feature_module = feature_module
#         self.model.eval()
#         self.cuda = use_cuda
#         if self.cuda:
#             self.model = model.cuda()
#
#         self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)
#
#     def forward(self, input):
#         return self.model(input)
#
#     def __call__(self, input, index=None):
#         m_batchsize, C, width, height = input.size()
#         if self.cuda:
#             features, output = self.extractor(input.cuda())
#         else:
#             features, output = self.extractor(input)
#
#         if index == None:
#             one_hot = np.zeros((8,12), dtype=np.float32)
#             for i in range(m_batchsize):
#                 m = output[i].reshape(1,12)
#                 index = np.argmax(m.detach().cpu().numpy())
#                 # one_hot[i] = np.zeros((1, output[i].size()[-1]), dtype=np.float32)
#                 one_hot[i][index] = 1
#         one_hot = torch.from_numpy(one_hot).requires_grad_(True)
#         if self.cuda:
#             one_hot = torch.sum(one_hot.cuda() * output)
#         else:
#             one_hot = torch.sum(one_hot * output)
#
#         self.feature_module.zero_grad()
#         self.model.zero_grad()
#         one_hot.backward(retain_graph=True)
#
#         grads_val = self.extractor.get_gradients()[-1]  # .cpu().data.numpy()
#
#         target = features[-1]
#         # target = target.cpu().data.numpy()[0, :]
#         weights = torch.mean(grads_val, dim=(2, 3), keepdim=False, out=None)
#         weights = weights.reshape([8, 1024, 1, 1])
#
#         cam = weights * target
#         cam = torch.sum(cam, dim=1)
#         cam = cam.reshape([8, 1, 8, 16])
#         # cam = F.interpolate(cam, scale_factor=32, mode='bicubic')
#
#         ############
#         #增加的relu和min—max normalize，这里resize应该在relu之后
#         Fc = nn.ReLU(inplace=True)
#         cam = Fc(cam)
#         cam = F.interpolate(cam, scale_factor=16, mode='bicubic')
#
#         max = torch.max(cam[0])
#         min = torch.min(cam[0])
#         map1 = cam[0]
#         # b,c = map.shape
#         # result = np.zeros(shape=(b,c))
#         result1 = (map1 - min) / (max - min + 1e-9)
#         result1 = result1.view([1, 1, width, height])
#
#         for i in range(1, m_batchsize):
#             max = torch.max(cam[i])
#             min = torch.min(cam[i])
#
#             map = cam[i]
#             # b,c = map.shape
#             # result = np.zeros(shape=(b,c))
#             result = (map - min) / (max - min + 1e-9)
#             result = result.view([1, 1, width, height])
#             result1 = torch.cat([result1, result], dim=0)
#
#         return result1




class yuanshi:
    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        m_batchsize, C, width, height = input.size()
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index == None:
            one_hot = np.zeros((8,12), dtype=np.float32)
            for i in range(m_batchsize):
                m = output[i].reshape(1,12)
                index = np.argmax(m.detach().cpu().numpy())
                # one_hot[i] = np.zeros((1, output[i].size()[-1]), dtype=np.float32)
                one_hot[i][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1]  # .cpu().data.numpy()

        target = features[-1]
        # target = target.cpu().data.numpy()[0, :]
        weights = torch.mean(grads_val, dim=(2, 3), keepdim=False, out=None)
        weights = weights.reshape([8, 2048, 1, 1])

        cam = weights * target
        cam = torch.sum(cam, dim=1)
        cam = cam.reshape([8, 1, 4, 8])
        cam = F.interpolate(cam, scale_factor=32, mode='bicubic')

        return cam
