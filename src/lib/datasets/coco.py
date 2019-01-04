import os
import numpy as np
import json
import cv2
import torch.utils.data as data
import torch
from utils.image import flip, shuffle_lr
from utils.image import draw_gaussian
from utils.image import get_affine_transform, affine_transform
from pycocotools.coco import COCO as COCOApi
from utils.image import transform_preds

class COCO(data.Dataset):
  def __init__(self, opt, split):
    self.num_joints = 17
    self.edges = [[0, 1], [0, 2], [1, 3], [2, 4], 
                  [4, 6], [3, 5], [5, 6], 
                  [5, 7], [7, 9], [6, 8], [8, 10], 
                  [6, 12], [5, 11], [11, 12], 
                  [12, 14], [14, 16], [11, 13], [13, 15]]
    self.shuffle_ref = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], 
                        [11, 12], [13, 14], [15, 16]]
    self.acc_idxs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    self.mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
    self.std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)
    self.aspect_ratio = 1.0 * opt.input_w / opt.input_h

    self.split = split
    self.data_dir = os.path.join(opt.data_dir, 'COCO')
    self.img_dir = os.path.join(self.data_dir, '{}2017'.format(split))
    self.annot_path = os.path.join(self.data_dir,
      'annotations', 'person_keypoints_{}2017.json'.format(split))
    self.opt = opt
    
    print('==> initializing coco {} data.'.format(split))
    self.coco = COCOApi(self.annot_path)
    img_idxs = self.coco.getImgIds()
    
    self.idxs = []
    self.clean_bbox = []

    for img_idx in img_idxs:
      idxs = self.coco.getAnnIds(imgIds=[img_idx])
      img_info = self.coco.loadImgs(ids=[img_idx])[0]
      width, height= img_info['width'], img_info['height']
      for idx in idxs:
        ann = self.coco.loadAnns(ids = idx)[0]
        num_keypoints = ann['num_keypoints']
        x, y, w, h = ann['bbox']
        x1 = np.max((0, x))
        y1 = np.max((0, y))
        x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
        y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
        if ann['area'] > 0 and x2 >= x1 and y2 >= y1 and num_keypoints > 0:
          self.idxs.append(idx)
          self.clean_bbox.append([x1, y1, x2-x1, y2-y1])
    print('Loaded COCO {} {} samples'.format(split, len(self.idxs)))


  def _box2cs(self, box):
    x, y, w, h = box[:4]
    return self._xywh2cs(x, y, w, h)

  def _xywh2cs(self, x, y, w, h):
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5

    if w > self.aspect_ratio * h:
        h = w * 1.0 / self.aspect_ratio
    elif w < self.aspect_ratio * h:
        w = h * self.aspect_ratio
    scale = np.array([w, h], dtype=np.float32)
    if center[0] != -1:
      scale = scale * 1.25

    return center, scale

  def __getitem__(self, index):
    ann = self.coco.loadAnns(ids = [self.idxs[index]])[0]
    clean_bbox = self.clean_bbox[index]
    img_info = self.coco.loadImgs(ids = [ann['image_id']])[0]
    img_path = os.path.join(self.img_dir, img_info['file_name'])
    img = cv2.imread(img_path)
    ids_all = self.coco.getAnnIds(imgIds = [ann['image_id']])
    ann_all = self.coco.loadAnns(ids = ids_all)
    pts_all = []
    for k in range(len(ann_all)):
      pts_k = np.array(ann_all[k]['keypoints'])
      pts_k = pts_k.reshape(self.num_joints, 3).astype(np.float32)
      pts_all.append(pts_k.copy())

    pts = np.array(ann['keypoints']).reshape(
      self.num_joints, 3).astype(np.float32)

    c, s = self._box2cs(clean_bbox)
    r = 0 
    
    if self.split == 'train':
      sf = self.opt.scale
      rf = self.opt.rotate
      s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
      r = np.clip(np.random.randn()*rf, -rf*2, rf*2) \
          if np.random.random() <= 0.6 else 0

    trans_input = get_affine_transform(
      c, s, r, [self.opt.input_w, self.opt.input_h])
    inp = cv2.warpAffine(img, trans_input, (self.opt.input_w, self.opt.input_h),
                         flags=cv2.INTER_LINEAR)
    inp = (inp.astype(np.float32) / 256. - self.mean) / self.std
    inp = inp.transpose(2, 0, 1)

    trans_output = get_affine_transform(
      c, s, r, [self.opt.output_w, self.opt.output_h])
    out = np.zeros((self.num_joints, self.opt.output_h, self.opt.output_w), 
                    dtype=np.float32)
    for i in range(self.num_joints):
      if pts[i, 2] > 0:
        pt = affine_transform(pts[i], trans_output)
        out[i] = draw_gaussian(out[i], pt, self.opt.hm_gauss) 

    '''
    out_all = np.zeros((self.num_joints, self.opt.output_w, self.opt.output_h), 
                       dtype=np.float32)
    for k in range(len(pts_all)):
      pts = pts_all[k]
      for i in range(self.num_joints):
        if pts[i, 2] > 0:
          pt = affine_transform(pts[i], trans_output)
          out_all[i] = np.maximum(
            out_all[i], draw_gaussian(out_all[i], pt, self.opt.hm_gauss))
    '''

    if self.split == 'train':
      if np.random.random() < self.opt.flip:
        inp = flip(inp)
        out = shuffle_lr(flip(out), self.shuffle_ref)
        # out_all = shuffle_lr(flip(out_all), self.shuffle_ref)
    
    meta = {'index' : index, 'id': self.idxs[index], 'center' : c, 
            'scale' : s, 'rotate': r, 'image_id': ann['image_id'], 
            'vis': pts[:, 2], 'score': 1}

    return {'input': inp, 'target': out, 'meta': meta}
    
  def __len__(self):
    return len(self.idxs)

  def convert_eval_format(self, pred, conf, meta):
    preds = np.zeros((pred.shape[0], pred.shape[1], 2))
    for i in range(pred.shape[0]):
      preds[i] = transform_preds(
        pred[i], meta['center'][i].numpy(), meta['scale'][i].numpy(), 
        [self.opt.output_h, self.opt.output_w])

    ret = []
    for i in range(pred.shape[0]):
      kpts = np.concatenate([preds[i], conf[i]], axis=1).astype(
        np.int32).reshape(self.num_joints * 3).tolist()
      score = int(meta['score'][i])
      ret.append({'category_id': 1, 'image_id': int(meta['image_id'].numpy()), \
                  'keypoints': kpts, 'score': score})
    return ret
