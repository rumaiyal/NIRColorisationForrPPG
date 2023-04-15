"""A modified image folder class

We modify the official PyTorch image folder (https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py)
so that this class can load images from both current directory and its subdirectories.
"""

import torch.utils.data as data

from PIL import Image
import os

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF','.PGM','.pgm',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    if dir[0]=='.':
            dir=os.getcwd()+dir[1:]
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    
    list_of_files = sorted( filter( lambda x: os.path.isfile(os.path.join(dir, x)),os.walk(dir) ) )
    for root, _, fnames in list_of_files:
    #for root, _, fnames in sorted(os.walk(dir)):
    
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]

def make_dataset_temporal(dir, max_dataset_size=float("inf")):
    images = []
    print(os.path.abspath(dir))
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    #Filenames must be in FrameXXXXX.pgm or FrameXXXXX.jpg format

    fnames=sorted(os.listdir(dir))
    for fname in fnames: 
       if is_image_file(fname):
          currentframestr=fname.split('.')
          currentframesplitstr=currentframestr[0].split('e')
          if '_' not in fname:
             currentframeprefixstr=currentframesplitstr[0]
             currentframenum=int(currentframesplitstr[1])
             endstr=''
          else:
             splitagainstr=currentframesplitstr[1].split('_')
             currentframenum=int(splitagainstr[0])
             endstr='_'+splitagainstr[1]
            
            
          #fnamefront=fname.split('.')[0]
          currentframeextension=fname.split('.')[1]
          currentframeprefixstr=currentframesplitstr[0]+'e'
          nirprev1str=currentframeprefixstr+(str(currentframenum-1)).zfill(5)+endstr+'.'+currentframeextension
          nirprev2str=currentframeprefixstr+(str(currentframenum-2)).zfill(5)+endstr+'.'+currentframeextension
          if is_image_file(fname) and (nirprev1str in fnames) and (nirprev2str in fnames):
             path = os.path.join(dir, fname)
             images.append(path)
    return images[:min(max_dataset_size, len(images))]



def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)
