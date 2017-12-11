import torch.utils.data as data
import numpy as np
import ref
import torch
from h5py import File
import cv2
from utils.utils import Rnd, Flip, ShuffleLR
from utils.img import Crop, DrawGaussian, Transform3D

class H36M(data.Dataset):
  def __init__(self, opt, split):
    print '==> initializing 3D {} data.'.format(split)
    annot = {}
    tags = ['action', 'bbox', 'camera', 'id', 'joint_2d', 'joint_3d_mono', 'subaction', 'subject', 'istrain']
    f = File('../data/h36m/annotSampleTest.h5', 'r')
    for tag in tags:
      annot[tag] = np.asarray(f[tag]).copy()
    f.close()
    
    ids = np.arange(annot['id'].shape[0])[annot['istrain'] == (1 if split == 'train' else 0)]
    for tag in tags:
      annot[tag] = annot[tag][ids]
    
    self.root = 7
    self.split = split
    self.opt = opt
    self.annot = annot
    self.nSamples = len(self.annot['id'])
    
    print 'Loaded 3D {} {} samples'.format(split, len(self.annot['id']))
  
  def LoadImage(self, index):
    folder = 's_{:02d}_act_{:02d}_subact_{:02d}_ca_{:02d}'.format(self.annot['subject'][index], self.annot['action'][index], \
              self.annot['subaction'][index], self.annot['camera'][index])
    path = '{}/{}/{}_{:06d}.jpg'.format(ref.h36mImgDir, folder, folder, self.annot['id'][index])
    #print 'path', path
    img = cv2.imread(path)
    return img
  
  def GetPartInfo(self, index):
    pts = self.annot['joint_2d'][index].copy()
    pts_3d_mono = self.annot['joint_3d_mono'][index].copy()
    pts_3d = self.annot['joint_3d_mono'][index].copy()
    c = np.ones(2) * ref.h36mImgSize / 2
    s = ref.h36mImgSize * 1.0
    
    pts_3d = pts_3d - pts_3d[self.root]
    
    s2d, s3d = 0, 0
    for e in ref.edges:
      s2d += ((pts[e[0]] - pts[e[1]]) ** 2).sum() ** 0.5
      s3d += ((pts_3d[e[0], :2] - pts_3d[e[1], :2]) ** 2).sum() ** 0.5
    scale = s2d / s3d
      
    for j in range(ref.nJoints):
      pts_3d[j, 0] = pts_3d[j, 0] * scale + pts[self.root, 0]
      pts_3d[j, 1] = pts_3d[j, 1] * scale + pts[self.root, 1]
      pts_3d[j, 2] = pts_3d[j, 2] * scale + ref.h36mImgSize / 2
    return pts, c, s, pts_3d, pts_3d_mono
  
      
  def __getitem__(self, index):
    if self.split == 'train':
      index = np.random.randint(self.nSamples)
    img = self.LoadImage(index)
    pts, c, s, pts_3d, pts_3d_mono = self.GetPartInfo(index)   
    pts_3d[7] = (pts_3d[12] + pts_3d[13]) / 2
      
    inp = Crop(img, c, s, 0, ref.inputRes) / 256.
    outMap = np.zeros((ref.nJoints, ref.outputRes, ref.outputRes))
    outReg = np.zeros((ref.nJoints, 3))
    for i in range(ref.nJoints):
      pt = Transform3D(pts_3d[i], c, s, 0, ref.outputRes)
      if pts[i][0] > 1:
        outMap[i] = DrawGaussian(outMap[i], pt[:2], ref.hmGauss) 
      outReg[i, 2] = pt[2] / ref.outputRes * 2 - 1
    
    inp = torch.from_numpy(inp)
    return inp, outMap, outReg, pts_3d_mono
    
  def __len__(self):
    return self.nSamples

