import numpy as np
import scipy.misc
import scipy.io
import os

from torchvision import transforms
import PIL.Image as Image

#baseline_SISR
class SISR():
    def __init__(self, root, is_train=True, data_len=None):
        self.root = root
        self.is_train = is_train
        # 获得train_file_list
        train_file_txt = open(os.path.join(self.root, 'anno', 'SISR_train.txt'))
        train_file_list = []
        train_label_list = []
        for line in train_file_txt:
            train_file_list.append(line[:-1].split(' ')[0])
            train_label_list.append(int(line[:-1].split(' ')[-1]))

        # 获得HR图片
        HR_file_txt = open(os.path.join(self.root, 'anno', 'SISR_truth.txt'))

        HR_file_list = []
        for line in HR_file_txt:
            HR_file_list.append(line[:-1].split(' ')[0])


        # 获得图片和标签
        if self.is_train:
            self.train_img = [scipy.misc.imread(os.path.join(self.root, 'SISR_train', train_file)) for train_file in
                              train_file_list[:data_len]]
            self.train_label = train_label_list[:data_len]


            self.HR_img = [scipy.misc.imread(os.path.join(self.root, 'SISR_truth', train_file)) for train_file in
                              HR_file_list[:data_len]]


    def __getitem__(self, index):
        if self.is_train:
            img,  HR = self.train_img[index], self.HR_img[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            # img = img[:, :, :3]
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Grayscale(num_output_channels=1)(img)
            img = transforms.Resize((128, 256), Image.BICUBIC)(img)#
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([.5], [.5])(img)

            if len(HR.shape) == 2:
                HR = np.stack([HR] * 3, 2)
            # HR = HR[:, :, :3]
            HR = Image.fromarray(HR, mode='RGB')
            HR = transforms.Grayscale(num_output_channels=1)(HR)
            HR = transforms.Resize((128, 256), Image.BICUBIC)(HR)#DISR
            HR = transforms.ToTensor()(HR)
            HR = transforms.Normalize([.5], [.5])(HR)


        else:
            img, target = self.test_img[index], self.test_label[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            # img = img[:, :, :3]
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Grayscale(num_output_channels=1)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([.5], [.5])(img)
            HR = img


        return img, HR

    def __len__(self):
        if self.is_train:
            return len(self.train_label)
        else:
            return len(self.test_label)


class Ship():
    def __init__(self,root, is_train = True, data_len = None):
        self.root = root
        self.is_train = is_train
        # 获得train_file_list
        train_file_txt = open(os.path.join(self.root, 'anno', 'train.txt'))
        train_file_list = []
        train_label_list = []
        for line in train_file_txt:
            train_file_list.append(line[:-1].split(' ')[0])
            train_label_list.append(int(line[:-1].split(' ')[-1]))

        # 获得test_file_list
        test_file_txt = open(os.path.join(self.root, 'anno', 'test.txt'))
        test_file_list = []
        test_label_list = []
        for line in test_file_txt:
            test_file_list.append(line[:-1].split(' ')[0])
            test_label_list.append(int(line[:-1].split(' ')[1]))

        # 获得图片和标签
        if self.is_train:
            self.train_img = [scipy.misc.imread(os.path.join(self.root, 'train', train_file)) for train_file in
                              train_file_list[:data_len]]
            self.train_label = train_label_list[:data_len]
        if not self.is_train:
            self.test_img = [scipy.misc.imread(os.path.join(self.root, 'test', test_file)) for test_file in
                             test_file_list[:data_len]]
            self.test_label = test_label_list[:data_len]

    def __getitem__(self, index):
        if self.is_train:
            img, target = self.train_img[index], self.train_label[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            # img = img[:,:,:3]
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Grayscale(num_output_channels=1)(img)
            img = transforms.Resize((128,256),Image.BICUBIC)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([.5], [.5])(img)

        else:
            img, target = self.test_img[index], self.test_label[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)

            img = Image.fromarray(img, mode='RGB')
            img = transforms.Grayscale(num_output_channels=1)(img)
            img = transforms.Resize((128, 256), Image.BICUBIC)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([.5], [.5])(img)

        return img,  target

    def __len__(self):
        if self.is_train:
            return len(self.train_label)
        else:
            return len(self.test_label)