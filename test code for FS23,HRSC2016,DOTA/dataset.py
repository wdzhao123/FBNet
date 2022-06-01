import numpy as np
import scipy.misc
import scipy.io
import os

from torchvision import transforms
import PIL.Image as Image


###FG_resnet
class SHIP_FG():
    def __init__(self, root, is_train=True, data_len=None):
        self.root = root
        self.is_train = is_train
        #获得train_file_list
        train_file_txt = open(os.path.join(self.root,'anno','FG_train.txt'))
        train_file_list = []
        train_label_list = []
        for line in train_file_txt:
            train_file_list.append(line[:-1].split(' ')[0])
            train_label_list.append(int(line[:-1].split(' ')[-1]))

        #获得test_file_list
        test_file_txt = open(os.path.join(self.root, 'anno', 'FG_test.txt'))
        test_file_list = []
        test_label_list = []
        for line in test_file_txt:
            test_file_list.append(line[:-1].split(' ')[0])
            test_label_list.append(int(line[:-1].split(' ')[1]))

        #获得图片和标签
        if self.is_train:
            self.train_img = [scipy.misc.imread(os.path.join(self.root, 'FG_train', train_file)) for train_file in
                              train_file_list[:data_len]]
            self.train_label = train_label_list[:data_len]
        if not self.is_train:
            self.test_img = [scipy.misc.imread(os.path.join(self.root, 'FG_test', test_file)) for test_file in
                             test_file_list[:data_len]]
            self.test_label = test_label_list[:data_len]

    def __getitem__(self, index):
        if self.is_train:
            img, target = self.train_img[index], self.train_label[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            # img = transforms.Grayscale(num_output_channels=1)(img)
            img = transforms.Resize((224,224),Image.BICUBIC)(img)
            img = transforms.RandomHorizontalFlip(p=0.5)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))(img)

        else:
            img, target = self.test_img[index], self.test_label[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)

            img = Image.fromarray(img, mode='RGB')
            # img = transforms.Grayscale(num_output_channels=1)(img)
            img = transforms.Resize((224, 224), Image.BICUBIC)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))(img)

        return img,  target

    def __len__(self):
        if self.is_train:
            return len(self.train_label)
        else:
            return len(self.test_label)
