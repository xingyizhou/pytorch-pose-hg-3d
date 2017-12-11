import torch.utils.data as data
import numpy as np
import ref
import torch
from h5py import File
import cv2
from utils.utils import Rnd, Flip, ShuffleLR
from utils.img import Crop, DrawGaussian, Transform

class MPII(data.Dataset):
  def __init__(self, opt, split, returnMeta = False):
    print '==> initializing 2D {} data.'.format(split)
    annot = {}
    tags = ['imgname','part','center','scale']
    f = File('{}/mpii/annot/{}.h5'.format(ref.dataDir, split), 'r')
    for tag in tags:
      annot[tag] = np.asarray(f[tag]).copy()
    f.close()

    print 'Loaded 2D {} {} samples'.format(split, len(annot['scale']))
    
    self.split = split
    self.opt = opt
    self.annot = annot
    self.returnMeta = returnMeta
  
  def LoadImage(self, index):
    path = '{}/{}'.format(ref.mpiiImgDir, self.annot['imgname'][index])
    img = cv2.imread(path)
    return img
  
  def GetPartInfo(self, index):
    pts = self.annot['part'][index].copy()
    c = self.annot['center'][index].copy()
    s = self.annot['scale'][index]
    s = s * 200
    return pts, c, s
      
  def __getitem__(self, index):
    img = self.LoadImage(index)
    pts, c, s = self.GetPartInfo(index)
    r = 0
    
    if self.split == 'train':
      s = s * (2 ** Rnd(ref.scale))
      r = 0 if np.random.random() < 0.6 else Rnd(ref.rotate)
    inp = Crop(img, c, s, r, ref.inputRes) / 256.
    out = np.zeros((ref.nJoints, ref.outputRes, ref.outputRes))
    Reg = np.zeros((ref.nJoints, 3))
    for i in range(ref.nJoints):
      if pts[i][0] > 1:
        pt = Transform(pts[i], c, s, r, ref.outputRes)
        out[i] = DrawGaussian(out[i], pt, ref.hmGauss) 
        Reg[i, :2] = pt
        Reg[i, 2] = 1
    if self.split == 'train':
      if np.random.random() < 0.5:
        inp = Flip(inp)
        out = ShuffleLR(Flip(out))
        Reg[:, 1] = Reg[:, 1] * -1
        Reg = ShuffleLR(Reg)
      #print 'before', inp[0].max(), inp[0].mean()
      inp[0] = np.clip(inp[0] * (np.random.random() * (0.4) + 0.6), 0, 1)
      inp[1] = np.clip(inp[1] * (np.random.random() * (0.4) + 0.6), 0, 1)
      inp[2] = np.clip(inp[2] * (np.random.random() * (0.4) + 0.6), 0, 1)
      #print 'after', inp[0].max(), inp[0].mean()
      
    inp = torch.from_numpy(inp)
    if self.returnMeta:
      return inp, out, Reg, np.zeros((ref.nJoints, 3))
    else:
      return inp, out
    
  def __len__(self):
    return len(self.annot['scale'])

