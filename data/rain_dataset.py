import os
from torch.utils.data import Dataset
import numpy as np
import os.path
#from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
#import random

#corrupted_files = open('corrupted_fils', 'w')


class RainDataset(Dataset):
    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """        
        super(RainDataset, self).__init__()

        self.opt = opt
        self.dir_A = os.path.join(opt.dataroot, opt.phase + '/data')  # create a path '/path/to/data/train/data'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + '/gt')    # create a path '/path/to/data/train/gt'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/train/data'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))   # load images from '/path/to/data/train/gt'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

   
    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of.
        Make sure two dataset have the same size.
        """
        return max(self.A_size, self.B_size)

    def __getitem__(self, index):

        A_path = self.A_paths[index % self.A_size]   # make sure index is within then range 
        B_path = self.B_paths[index % self.B_size]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        img = A_img.resize((224,224),Image.BILINEAR)
        gt = B_img.resize((224,224),Image.BILINEAR)
        
        img = (np.array(img) / 255.0).astype('float32') #normalization
        gt = (np.array(gt) / 255.0).astype('float32')

        return [img,gt]
