import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset, make_dataset_temporal
from skimage import color  # require skimage
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import cv2
from pathlib import Path

class Nir12bit1chtorgb8bit3chhumanfacetensorDataset(BaseDataset):
    """This dataset class can load a set of natural images in RGB, and convert RGB format into (L, ab) pairs in Lab color space.

    This dataset is required by pix2pix-based colorization model ('--model colorization')
    """
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        By default, the number of channels for input image  is 1 (Grey) and
        the number of channels for output image is 3 (RGB). The direction is from A to B
        """
        parser.set_defaults(input_nc=1, output_nc=3, direction='AtoB')
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir = os.path.join(opt.dataroot, opt.phase)
        self.AB_paths = sorted(make_dataset_temporal(self.dir, opt.max_dataset_size))
        assert(opt.input_nc == 1 and opt.output_nc == 3 and opt.direction == 'AtoB')
        #self.transform = get_transform(self.opt, convert=False)
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - the L channel of an image
            B (tensor) - - the ab channels of the same image
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        path = self.AB_paths[index]
        

        if path[0]=='.':
            path=os.getcwd()+path[1:]

        nirdir=Path(path)

        currentframestr=str(nirdir.stem)
        currentframesplitstr=currentframestr.split('e')
        currentframeprefixstr=currentframesplitstr[0]
        currentframenum=int(currentframesplitstr[1])
        
        nirdironlystr='/'
        rgbdironlystr='/'
        for count in range(1,len(nirdir.parts)-3):
           nirdironlystr=nirdironlystr+str(nirdir.parts[count])+'/'
           rgbdironlystr=rgbdironlystr+str(nirdir.parts[count])+'/'
        
        
        #Normalise all grey inputs
        
        greyimnp=np.right_shift(cv2.imread(str(nirdir),-1),6)

        #Normalise all grey inputs between 0 and 1
        #Greyimage has 10bits, 1024 graylevels
        #0<g1<1023
        
        h, w = np.shape(greyimnp)
        
        A_np=np.zeros((1,h,w),dtype=np.float32)
        A_np[0,:,:]=np.float32(greyimnp/1023.0)
        A=torch.from_numpy(A_np)
        
        rgbdirstr=rgbdironlystr+'RGBregisteredbgremoved'+'/'+str(nirdir.parts[-2]) \
                  +'/'+str(nirdir.stem)+'.jpg'
        
        #In case jpg file does not exist (as in the case of test of cropped 256 patches)
        if not os.path.isfile(rgbdirstr):
            rgbdirstr=rgbdironlystr+'RGBregisteredbgremoved'+'/'+str(nirdir.parts[-2]) \
                  +'/'+str(nirdir.stem)+'.png'
            
        rgbdir=Path(rgbdirstr)
        rgbimnp = np.asarray(Image.open(rgbdir).convert('RGB'))
        
  
        
        #Normalise all colour inputs between 0 and 1
        #Colour image has 8 bits, 256 colour levels
        #0<g1<255
        
        B_np=np.zeros((3,h,w),dtype=np.float32)
        B_np[0,:,:]=np.float32(rgbimnp[:,:,0]/255.0)
        B_np[1,:,:]=np.float32(rgbimnp[:,:,1]/255.0)
        B_np[2,:,:]=np.float32(rgbimnp[:,:,2]/255.0)
        B=torch.from_numpy(B_np)
         
        
        # apply the same transform to both A and B
        #transform_params = get_params(self.opt, A.size)
        imgsizetuple=(A.shape[2],A.shape[1])
        transform_params = get_params(self.opt, imgsizetuple)
        A_transform = get_transform(self.opt, transform_params, convert=False, grayscale=True)
        B_transform = get_transform(self.opt, transform_params, convert=False, grayscale=False)

        A = A_transform(A)
        B = B_transform(B)
        
        return {'A': A, 'B': B, 'A_paths': path, 'B_paths': path}
        

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
