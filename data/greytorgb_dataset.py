import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from skimage import color  # require skimage
from PIL import Image
import numpy as np
import torchvision.transforms as transforms


class GreytoRGBDataset(BaseDataset):
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
        self.AB_paths = sorted(make_dataset(self.dir, opt.max_dataset_size))
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
        rgbim = Image.open(path).convert('RGB')
        greyim = Image.open(path).convert('L')
        
        w, h = rgbim.size
        w2 = int(w / 2)
        
        #rgbim_t = self.transform(rgbim)
        #rgbim_t = np.array(im)
        #rgb_t = transforms.ToTensor()(rgbim_t.astype(np.float32))
        
        B=rgbim.crop((0, 0, w2, h))
        A=greyim.crop((0, 0, w2, h))
        
        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A)
        B = B_transform(B)
        
        return {'A': A, 'B': B, 'A_paths': path, 'B_paths': path}
        

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
