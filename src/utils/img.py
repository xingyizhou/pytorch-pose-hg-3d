import torch
import numpy as np
import cv2
import ref

sigma_inp = ref.hmGaussInp
n = sigma_inp * 6 + 1
g_inp = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        g_inp[i, j] = np.exp(-((i - n / 2) ** 2 + (j - n / 2) ** 2) / (2. * sigma_inp * sigma_inp))
        
def GetTransform(center, scale, rot, res):
  h = scale
  t = np.eye(3)

  t[0, 0] = res / h
  t[1, 1] = res / h
  t[0, 2] = res * (- center[0] / h + 0.5)
  t[1, 2] = res * (- center[1] / h + 0.5)
  
  if rot != 0:
    rot = -rot 
    r = np.eye(3)
    ang = rot * np.math.pi / 180
    s = np.math.sin(ang)
    c = np.math.cos(ang)
    r[0, 0] = c
    r[0, 1] = - s
    r[1, 0] = s
    r[1, 1] = c
    t_ = np.eye(3)
    t_[0, 2] = - res / 2
    t_[1, 2] = - res / 2
    t_inv = np.eye(3)
    t_inv[0, 2] = res / 2
    t_inv[1, 2] = res / 2
    t = np.dot(np.dot(np.dot(t_inv,  r), t_), t)
  
  return t


def Transform(pt, center, scale, rot, res, invert = False):
  pt_ = np.ones(3)
  pt_[0], pt_[1] = pt[0], pt[1]
  
  t = GetTransform(center, scale, rot, res)
  if invert:
    t = np.linalg.inv(t)
  new_point = np.dot(t, pt_)[:2]
  new_point = new_point.astype(np.int32)
  return new_point 


def getTransform3D(center, scale, rot, res):
  h = 1.0 * scale
  t = np.eye(4)
  
  t[0][0] = res / h
  t[1][1] = res / h
  t[2][2] = res / h
  
  t[0][3] = res * (- center[0] / h + 0.5)
  t[1][3] = res * (- center[1] / h + 0.5)
  
  if rot != 0:
    raise Exception('Not Implement')
  
  return t
  

def Transform3D(pt, center, scale, rot, res, invert = False):
  pt_ = np.ones(4)
  pt_[0], pt_[1], pt_[2] = pt[0], pt[1], pt[2]
  #print 'c s r res', center, scale, rot, res
  t = getTransform3D(center, scale, rot, res)
  if invert:
    t = np.linalg.inv(t)
  #print 't', t
  #print 'pt_', pt_
  new_point = np.dot(t, pt_)[:3]
  #print 'new_point', new_point
  #if not invert:
  #  new_point = new_point.astype(np.int32)
  return new_point
  

def Crop(img, center, scale, rot, res):
  ht, wd = img.shape[0], img.shape[1]
  tmpImg, newImg = img.copy(), np.zeros((res, res, 3), dtype = np.uint8)

  scaleFactor = scale / res
  if scaleFactor < 2:
    scaleFactor = 1
  else:
    newSize = int(np.math.floor(max(ht, wd) / scaleFactor))
    newSize_ht = int(np.math.floor(ht / scaleFactor))
    newSize_wd = int(np.math.floor(wd / scaleFactor))
    if newSize < 2:
      return torch.from_numpy(newImg.transpose(2, 0, 1).astype(np.float32) / 256.)
    else:
      tmpImg = cv2.resize(tmpImg, (newSize_wd, newSize_ht)) #TODO
      ht, wd = tmpImg.shape[0], tmpImg.shape[1]
    
  c, s = 1.0 * center / scaleFactor, scale / scaleFactor
  c[0], c[1] = c[1], c[0]
  ul = Transform((0, 0), c, s, 0, res, invert = True)
  br = Transform((res, res), c, s, 0, res, invert = True)
  
  if scaleFactor >= 2:
    br = br - (br - ul - res)
    
  pad = int(np.math.ceil((((ul - br) ** 2).sum() ** 0.5) / 2 - (br[0] - ul[0]) / 2))
  if rot != 0:
    ul = ul - pad
    br = br + pad
    
  old_ = [max(0, ul[0]),   min(br[0], ht),         max(0, ul[1]),   min(br[1], wd)]
  new_ = [max(0, - ul[0]), min(br[0], ht) - ul[0], max(0, - ul[1]), min(br[1], wd) - ul[1]]
  
  newImg = np.zeros((br[0] - ul[0], br[1] - ul[1], 3), dtype = np.uint8)
  #print 'new old newshape tmpshape center', new_[0], new_[1], old_[0], old_[1], newImg.shape, tmpImg.shape, center
  try:
    newImg[new_[0]:new_[1], new_[2]:new_[3], :] = tmpImg[old_[0]:old_[1], old_[2]:old_[3], :]
  except:
    #print 'ERROR: new old newshape tmpshape center', new_[0], new_[1], old_[0], old_[1], newImg.shape, tmpImg.shape, center
    return np.zeros((3, res, res), np.uint8)
  if rot != 0:
    M = cv2.getRotationMatrix2D((newImg.shape[0] / 2, newImg.shape[1] / 2), rot, 1)
    newImg = cv2.warpAffine(newImg, M, (newImg.shape[0], newImg.shape[1]))
    newImg = newImg[pad+1:-pad+1, pad+1:-pad+1, :].copy()

  if scaleFactor < 2:
    newImg = cv2.resize(newImg, (res, res))
  
  return newImg.transpose(2, 0, 1).astype(np.float32)

def Gaussian(sigma):
  if sigma == 7:
    return np.array([0.0529,  0.1197,  0.1954,  0.2301,  0.1954,  0.1197,  0.0529,
                     0.1197,  0.2709,  0.4421,  0.5205,  0.4421,  0.2709,  0.1197,
                     0.1954,  0.4421,  0.7214,  0.8494,  0.7214,  0.4421,  0.1954,
                     0.2301,  0.5205,  0.8494,  1.0000,  0.8494,  0.5205,  0.2301,
                     0.1954,  0.4421,  0.7214,  0.8494,  0.7214,  0.4421,  0.1954,
                     0.1197,  0.2709,  0.4421,  0.5205,  0.4421,  0.2709,  0.1197,
                     0.0529,  0.1197,  0.1954,  0.2301,  0.1954,  0.1197,  0.0529]).reshape(7, 7)
  elif sigma == n:
    return g_inp
  else:
    raise Exception('Gaussian {} Not Implement'.format(sigma))

def DrawGaussian(img, pt, sigma):
  tmpSize = int(np.math.ceil(3 * sigma))
  ul = [int(np.math.floor(pt[0] - tmpSize)), int(np.math.floor(pt[1] - tmpSize))]
  br = [int(np.math.floor(pt[0] + tmpSize)), int(np.math.floor(pt[1] + tmpSize))]
  
  if ul[0] > img.shape[1] or ul[1] > img.shape[0] or br[0] < 1 or br[1] < 1:
    return img
  
  size = 2 * tmpSize + 1
  g = Gaussian(size)
  
  g_x = [max(0, -ul[0]), min(br[0], img.shape[1]) - max(0, ul[0]) + max(0, -ul[0])]
  g_y = [max(0, -ul[1]), min(br[1], img.shape[0]) - max(0, ul[1]) + max(0, -ul[1])]

  img_x = [max(0, ul[0]), min(br[0], img.shape[1])]
  img_y = [max(0, ul[1]), min(br[1], img.shape[0])]
  
  img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
  return img

