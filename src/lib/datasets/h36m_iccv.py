import os
import torch.utils.data as data
import numpy as np
import torch
import json
import cv2
import pickle
from utils.image import flip, shuffle_lr
from utils.image import draw_gaussian, adjust_aspect_ratio
from utils.image import get_affine_transform, affine_transform
from utils.image import transform_preds
import scipy.io as sio

class H36M(data.Dataset):
  def __init__(self, opt, split):
    print('==> initializing 3D {} data.'.format(split))
    self.num_joints = 16
    self.num_eval_joints = 16
    self.h36m_to_mpii = [3, 2, 1, 4, 5, 6, 0, 7, 8, 10, 16, 15, 14, 11, 12, 13]
    self.mpii_to_h36m = [6, 2, 1, 0, 3, 4, 5, 7, 8, 9, 9, \
                         13, 14, 15, 12, 11, 10]
    self.acc_idxs = [0, 1, 2, 3, 4, 5, 10, 11, 14, 15]
    self.shuffle_ref = [[0, 5], [1, 4], [2, 3], 
                        [10, 15], [11, 14], [12, 13]]
    self.edges = [[0, 1], [1, 2], [2, 6], [6, 3], [3, 4], [4, 5], 
                  [10, 11], [11, 12], [12, 8], [8, 13], [13, 14], [14, 15], 
                  [6, 8], [8, 9]]
    self.edges_3d = [[3, 2], [2, 1], [1, 0], [0, 4], [4, 5], [5, 6], \
                     [0, 7], [7, 8], [8, 9], [9, 10],\
                     [16, 15], [15, 14], [14, 8], [8, 11], [11, 12], [12, 13]]
    self.mean_bone_length = 4072.3544 # train: 4072.3544, val: 4252.4707
    self.mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
    self.std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)
    self.aspect_ratio = 1.0 * opt.input_w / opt.input_h
    self.split = split
    self.opt = opt
    self.image_path = os.path.join(
      self.opt.data_dir, 'h36m', 'images')
    cache_path = os.path.join(
      opt.data_dir, 'h36m', 'cache', 'iccv_gt2d_{}.json'.format(split))
    if os.path.exists(cache_path):
      self.annot = json.load(open(cache_path, 'r'))
    else:
      self.annot = self._preprocess(split)
      # import pdb; pdb.set_trace()
      json.dump(self.annot, open(cache_path, 'w'))
    # dowmsample validation data
    self.idxs = np.arange(len(self.annot))
    self.num_samples = len(self.idxs)
    print('Loaded 3D {} {} samples'.format(split, self.num_samples))
  
  def _preprocess(self, split):
    subject_list = {'train': [1, 5, 6, 7, 8], 'val': [9, 11]}
    action_list = np.arange(2, 17)
    subaction_list = np.arange(1, 3)
    camera_list = np.arange(1, 5)
    annot_name = 'matlab_meta.mat'
    ret = []
    print('Preprocessing ...')
    sum_bone_length = 0
    for k, subject in enumerate(subject_list[split]):
      print('[{}/{}]'.format(k, len(subject_list[split])))
      for action in action_list:
        for subaction in subaction_list:
          for camera in camera_list:
            folder_name = \
              's_{:02d}_act_{:02d}_subact_{:02d}_ca_{:02d}'.format(
              subject, action, subaction, camera)
            annot_file = os.path.join(self.image_path, folder_name, annot_name)
            try:
              data = sio.loadmat(annot_file)
            except:
              continue
            n = int(data['num_images'][0][0])
            meta_Y2d = data['Y2d'].reshape(17, 2, n)
            meta_Y3d_mono = data['Y3d_mono'].reshape(17, 3, n)
            bbox = data['bbox'].transpose(1, 0)
            for i in range(n):
              if i % 5 != 0:
                continue
              if split == 'val' and i % 200 != 0:
                continue
              sample = {}
              sample['id'] = int(i + 1)
              sample['uvd'] = self._get_mpii_uvd(
                meta_Y2d[:, :, i], meta_Y3d_mono[:, :, i]).tolist()
              sample['gt_3d'] = meta_Y3d_mono[:, :, i].tolist()
              sample['bbox'] = bbox[i].tolist()
              sample['subject'] = int(subject)
              sample['action'] = int(action)
              sample['subaction'] = int(subaction)
              sample['camera'] = int(camera)
              sum_bone_length += self._get_bone_length(
                meta_Y3d_mono[:, :, i][self.h36m_to_mpii])
              ret.append(sample)
    print('sum_bone_length', sum_bone_length / len(ret))
    return ret
  
  
  def _get_mpii_uvd(self, pts_2d, pts_3d):
    pts_2d = pts_2d[self.h36m_to_mpii]
    pts_3d = pts_3d[self.h36m_to_mpii]
    pts_2d[7] = (pts_2d[12] + pts_2d[13]) / 2
    pts_3d[7] = (pts_3d[12] + pts_3d[13]) / 2
    root = 7
    pts_3d = pts_3d - pts_3d[root:root+1]
    
    s2d, s3d = 0, 0
    for e in self.edges:
      s2d += ((pts_2d[e[0]] - pts_2d[e[1]]) ** 2).sum() ** 0.5
      s3d += ((pts_3d[e[0], :2] - pts_3d[e[1], :2]) ** 2).sum() ** 0.5
    scale = s2d / s3d
    
    uvd = np.zeros((self.num_joints, 3), dtype=np.float32)
    for j in range(self.num_joints):
      uvd[j, 0] = pts_2d[j, 0]
      uvd[j, 1] = pts_2d[j, 1]
      uvd[j, 2] = pts_3d[j, 2] * scale
    return uvd

  def _load_image(self, index):
    ann = self.annot[self.idxs[index]]
    folder = 's_{:02d}_act_{:02d}_subact_{:02d}_ca_{:02d}'.format(
      ann['subject'], ann['action'], ann['subaction'], ann['camera'])
    path = '{}/{}/{}_{:06d}.jpg'.format(
      self.image_path, folder, folder, ann['id'])
    img = cv2.imread(path)
    return img
  
  def _get_part_info(self, index):
    ann = self.annot[self.idxs[index]]
    gt_3d = np.array(ann['gt_3d'], dtype=np.float32)
    gt_3d = gt_3d - gt_3d[:1]
    gt_3d = gt_3d[self.h36m_to_mpii]
    pts = np.array(ann['uvd'], dtype=np.float32)
    c = np.array([112, 112], dtype=np.float32)
    s = 224.
    return gt_3d, pts, c, s
      
  def __getitem__(self, index):
    if index < 10 and self.split == 'train':
      self.idxs = np.random.choice(
        self.num_samples, self.num_samples, replace=False)
    img = self._load_image(index)
    gt_3d, pts, c, s = self._get_part_info(index)
    
    r = 0
    s = np.array([s, s])
    s = adjust_aspect_ratio(s, self.aspect_ratio, self.opt.fit_short_side)
    
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
    reg_target = np.zeros((self.num_joints, 1), dtype=np.float32)
    reg_ind = np.zeros((self.num_joints), dtype=np.int64)
    reg_mask = np.zeros((self.num_joints), dtype=np.uint8)
    pts_crop = np.zeros((self.num_joints, 2), dtype=np.int32)
    for i in range(self.num_joints):
      pt = affine_transform(pts[i, :2], trans_output).astype(np.int32)
      if pt[0] >= 0 and pt[1] >=0 and pt[0] < self.opt.output_w \
        and pt[1] < self.opt.output_h:
        pts_crop[i] = pt
        out[i] = draw_gaussian(out[i], pt, self.opt.hm_gauss)
        reg_target[i] = pts[i, 2] / s[0] # assert not fit_short
        reg_ind[i] = pt[1] * self.opt.output_w * self.num_joints + \
                     pt[0] * self.num_joints + i # note transposed
        reg_mask[i] = 1
    
    meta = {'index' : self.idxs[index], 'center' : c, 'scale' : s,
            'gt_3d': gt_3d, 'pts_crop': pts_crop}

    ret = {'input': inp, 'target': out, 'meta': meta, 
           'reg_target': reg_target, 'reg_ind': reg_ind, 'reg_mask': reg_mask}
    
    return ret

  def __len__(self):
    return self.num_samples

  def convert_eval_format(self, pred):
    sum_bone_length = self._get_bone_length(pred)
    pred_h36m = pred
    pred_h36m[7] = (pred_h36m[6] + pred_h36m[8]) / 2
    # pred_h36m = pred[self.mpii_to_h36m]
    # pred_h36m[7] = (pred_h36m[0] + pred_h36m[8]) / 2
    # pred_h36m[9] = (pred_h36m[8] + pred_h36m[10]) / 2
    pred_h36m = pred_h36m * self.mean_bone_length / sum_bone_length
    return pred_h36m

  def _get_bone_length(self, pts):
    sum_bone_length = 0
    for e in self.edges:
      sum_bone_length += ((pts[e[0]] - pts[e[1]]) ** 2).sum() ** 0.5
    return sum_bone_length