import os
import torch.utils.data as data
import numpy as np
import torch
import json
import cv2
from utils.image import flip, shuffle_lr
from utils.image import draw_gaussian, adjust_aspect_ratio
from utils.image import get_affine_transform, affine_transform
from utils.image import transform_preds

class MPII(data.Dataset):
  def __init__(self, opt, split):
    print('==> initializing 2D {} data.'.format(split))
    self.num_joints = 16
    self.acc_idxs = [0, 1, 2, 3, 4, 5, 10, 11, 14, 15]
    self.shuffle_ref = [[0, 5], [1, 4], [2, 3], 
                        [10, 15], [11, 14], [12, 13]]
    self.edges = [[0, 1], [1, 2], [2, 6], [6, 3], [3, 4], [4, 5], 
                  [10, 11], [11, 12], [12, 8], [8, 13], [13, 14], [14, 15], 
                  [6, 8], [8, 9]]
    self.mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
    self.std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)
    annot = {}
    tags = ['image','joints','center','scale']
    self.data_path = os.path.join(opt.data_dir, 'mpii')
    f = json.load(open('{}/annot/{}.json'.format(self.data_path, split), 'r'))
    self.num_samples = len(f)
    for tag in tags:
      annot[tag] = []
      for i in range(self.num_samples):
        annot[tag].append(f[i][tag])
      annot[tag] = np.array(annot[tag])
    annot['pts_all'] = []
    for i in range(self.num_samples):
      pts_all = [annot['joints'][i]]
      imgname = annot['image'][i]
      cur = i - 1
      while cur >= 0 and annot['image'][cur] == imgname:
        pts_all.append(annot['joints'][cur])
        cur -= 1
      cur = i + 1
      while cur < self.num_samples and annot['image'][cur] == imgname:
        pts_all.append(annot['joints'][cur])
        cur += 1
      annot['pts_all'].append(pts_all)
    print('Loaded 2D {} {} samples'.format(split, self.num_samples))
    self.aspect_ratio = 1.0 * opt.input_w / opt.input_h
    self.split = split
    self.opt = opt
    self.annot = annot
  
  def _load_image(self, index):
    path = '{}/images/{}'.format(
      self.data_path, self.annot['image'][index])
    img = cv2.imread(path)
    return img
  
  def _get_part_info(self, index):
    pts_all = np.array(self.annot['pts_all'][index])
    pts = self.annot['joints'][index].copy().astype(np.float32)
    c = self.annot['center'][index].copy().astype(np.float32)
    s = self.annot['scale'][index]
    c[1] = c[1] + 15 * s
    c -= 1
    s = s * 1.25
    s = s * 200
    return pts_all, pts, c, s
      
  def __getitem__(self, index):
    img = self._load_image(index)
    _, pts, c, s = self._get_part_info(index)
    r = 0
    
    if self.split == 'train':
      sf = self.opt.scale
      rf = self.opt.rotate
      s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
      r = np.clip(np.random.randn()*rf, -rf*2, rf*2) \
          if np.random.random() <= 0.6 else 0
    s = min(s, max(img.shape[0], img.shape[1])) * 1.0
    s = np.array([s, s])
    s = adjust_aspect_ratio(s, self.aspect_ratio, self.opt.fit_short_side)

    flipped = (self.split == 'train' and np.random.random() < self.opt.flip)
    if flipped:
      img = img[:, ::-1, :]
      c[0] = img.shape[1] - 1 - c[0]
      pts[:, 0] = img.shape[1] - 1 - pts[:, 0]
      for e in self.shuffle_ref:
        pts[e[0]], pts[e[1]] = pts[e[1]].copy(), pts[e[0]].copy()

    trans_input = get_affine_transform(
      c, s, r, [self.opt.input_h, self.opt.input_w])
    inp = cv2.warpAffine(img, trans_input, (self.opt.input_h, self.opt.input_w),
                         flags=cv2.INTER_LINEAR)
    inp = (inp.astype(np.float32) / 256. - self.mean) / self.std
    inp = inp.transpose(2, 0, 1)

    trans_output = get_affine_transform(
      c, s, r, [self.opt.output_h, self.opt.output_w])
    out = np.zeros((self.num_joints, self.opt.output_h, self.opt.output_w), 
                    dtype=np.float32)
    pts_crop = np.zeros((self.num_joints, 2), dtype=np.int32)
    for i in range(self.num_joints):
      if pts[i, 0] > 0 or pts[i, 1] > 0:
        pts_crop[i] = affine_transform(pts[i], trans_output)
        out[i] = draw_gaussian(out[i], pts_crop[i], self.opt.hm_gauss) 
    
    meta = {'index' : index, 'center' : c, 'scale' : s, \
            'pts_crop': pts_crop}
    return {'input': inp, 'target': out, 'meta': meta}
    
  def __len__(self):
    return self.num_samples

  def convert_eval_format(self, pred, conf, meta):
    ret = np.zeros((pred.shape[0], pred.shape[1], 2))
    for i in range(pred.shape[0]):
      ret[i] = transform_preds(
        pred[i], meta['center'][i].numpy(), meta['scale'][i].numpy(), 
        [self.opt.output_h, self.opt.output_w])
    return ret

