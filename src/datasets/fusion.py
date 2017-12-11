import torch.utils.data as data
import numpy as np
import ref
import torch
import cv2
from mpii import MPII
from h36m import H36M

class Fusion(data.Dataset):
  def __init__(self, opt, split):
    self.ratio3D = opt.ratio3D
    self.split = split
    self.dataset3D = H36M(opt, split)
    self.dataset2D = MPII(opt, split, returnMeta = True)
    self.nImages2D = len(self.dataset2D)
    self.nImages3D = min(len(self.dataset3D), int(self.nImages2D * self.ratio3D))
    
    print '#Images2D {}, #Images3D {}'.format(self.nImages2D, self.nImages3D)
  def __getitem__(self, index):
    if index < self.nImages3D:
      return self.dataset3D[index]
    else:
      return self.dataset2D[index - self.nImages3D]
    
  def __len__(self):
    return self.nImages2D + self.nImages3D

