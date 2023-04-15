"""This module implements an abstract base class (ABC) 'BaseDataset' for datasets.

It also includes common transformation functions (e.g., get_transform, __scale_width), which can be later used in subclasses.
"""
import random
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from abc import ABC, abstractmethod
from scipy import stats

class BaseDataset(data.Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.

    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the class; save the options in the class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        self.root = opt.dataroot

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset."""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """
        pass


def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    
    #hardcoding input size after padding, else cannot get random shift
    new_h=286
    new_w=286
    teta=0
    
    ###Not using these options; only using 'crop'###
    if opt.preprocess == 'resize_and_crop':
        new_h = new_w = opt.load_size
    elif opt.preprocess == 'scale_width_and_crop':
        new_w = opt.load_size
        new_h = opt.load_size * h // w
    
    if opt.preprocess == 'none_nobasechg':
       x = 0
       y = 0
    else:  
       x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
       y = random.randint(0, np.maximum(0, new_h - opt.crop_size))
       teta = random.randint(0,45)
       direction=random.choice([True,False])
       if direction==False:
          teta=-teta

    #print(f'new_h {new_h}')
    #print(f'opt.crop_size {opt.crop_size}')
    #print(f'x y {x} {y}')
    #print(f'teta {teta}')
        
    flip = random.random() > 0.5

    return {'crop_pos': (x, y), 'flip': flip, 'rotangle': teta}


def get_transform(opt, params=None, grayscale=False, method=transforms.InterpolationMode.BICUBIC, convert=True):
    transform_list = []
    if grayscale and convert==True:
        transform_list.append(transforms.Grayscale(1))
    if 'resize' in opt.preprocess:
        osize = [opt.load_size, opt.load_size]
        transform_list.append(transforms.Resize(osize, method))
    elif 'scale_width' in opt.preprocess:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.load_size, opt.crop_size, method)))
        
        
    #Include padding for arbitrary crop and rotation for training
    if 'none' not in opt.preprocess: 
       #print('came into pad') 
       padsize=15 
       transform_list.append(transforms.Pad(padding=padsize))    

    if 'crop' in opt.preprocess:
        if params is None:
            print('came into random crop')
            transform_list.append(transforms.RandomCrop(opt.crop_size))
        else:
            transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.crop_size)))
            transform_list.append(transforms.Lambda(lambda img: __rotate(img, params['rotangle'])))
            
    if opt.preprocess == 'none':
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))
    if opt.preprocess == 'none_nobasechg': #In the case of test images of size 256
        osize = [opt.load_size, opt.load_size]
        transform_list.append(transforms.Resize(osize, method))
        transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.crop_size)))

        
    if not opt.no_flip:
        if params is None:
            transform_list.append(transforms.RandomHorizontalFlip())
        elif params['flip']:
            transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    if convert:
        transform_list += [transforms.ToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    else:
        if grayscale:
            numch=opt.input_nc
        else:
            numch=opt.output_nc
        tuplen=tuple(0.5 for _ in range(numch))
        transform_list += [transforms.Normalize(tuplen, tuplen)]         
    return transforms.Compose(transform_list)


def __make_power_2(img, base, method=transforms.InterpolationMode.BICUBIC):
    
    
    if isinstance(img,Image.Image)==True:
       ow, oh = img.size
    else:
       if len(img.shape)==3: 
          ow =  img.shape[2]
          oh =  img.shape[1]
       else:
          ow = img.shape[1]
          oh = img.shape[0]  
        
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if h == oh and w == ow:
        return img

    __print_size_warning(ow, oh, w, h)
    if isinstance(img,Image.Image)==True:
      return img.resize((w, h), method)
    else:
      return F.resize(img,size=(w,h),InterpolationMode=method)


def __scale_width(img, target_size, crop_size, method=transforms.InterpolationMode.BICUBIC):

    if isinstance(img,Image.Image)==True:
       ow, oh = img.size
    else:
       if len(img.shape)==3: 
          ow =  img.shape[2]
          oh =  img.shape[1]
       else:
          ow = img.shape[1]
          oh = img.shape[0]
        
    if ow == target_size and oh >= crop_size:
        return img
    w = target_size
    h = int(max(target_size * oh / ow, crop_size))
    
    
    if isinstance(img,Image.Image)==True:
      return img.resize((w, h), method)
    else:
      return F.resize(img,size=(w,h),InterpolationMode=method) 


def __crop(img, pos, size):

    if isinstance(img,Image.Image)==True:
       ow, oh = img.size
    else:
       if len(img.shape)==3: 
          ow =  img.shape[2]
          oh =  img.shape[1]
       else:
          ow = img.shape[1]
          oh = img.shape[0]
        
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        if isinstance(img,Image.Image)==True:
           return img.crop((x1, y1, x1 + tw, y1 + th))
        else:
           return F.crop(img,x1,y1,tw,th)
    return img

def __flip(img, flip):
    if flip:
        if isinstance(img,Image.Image)==True:
           return img.transpose(Image.FLIP_LEFT_RIGHT)
        else:
           return torch.flip(img,[2]) 
    return img

#def __rotate(img,rotangle):
#    if isinstance(img,Image.Image)==True:
#       return img.rotate(rotangle,PIL.Image.NEAREST)
#    else:
#       return F.rotate(img,rotangle) 

def __rotate(img,rotangle):
    if isinstance(img,Image.Image)==True:
       temp=img.rotate(rotangle,PIL.Image.NEAREST)
       #Replace all 0s after rotation with modal value
       modevalue=stats.mode(temp,keepdims=True,axis=None).mode[0]
       idx0=(temp==0).nonzero(as_tuple=True)
       temp[idx0]=torch.from_numpy(modevalue)
        
    
    else:
       temp=F.rotate(img,rotangle) 
       #Replace all 0s after rotation with modal value
       modevalue=stats.mode(temp,keepdims=True,axis=None).mode[0]
       idx0=(temp==0).nonzero(as_tuple=True)
       temp[idx0]=torch.tensor(modevalue)
        
    return temp 
    

def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True
