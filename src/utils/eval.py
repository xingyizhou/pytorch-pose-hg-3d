import numpy as np
import ref
from img import Transform

def getPreds(hm):
  assert len(hm.shape) == 4, 'Input must be a 4-D tensor'
  res = hm.shape[2]
  hm = hm.reshape(hm.shape[0], hm.shape[1], hm.shape[2] * hm.shape[3])
  idx = np.argmax(hm, axis = 2)
  preds = np.zeros((hm.shape[0], hm.shape[1], 2))
  for i in range(hm.shape[0]):
    for j in range(hm.shape[1]):
      preds[i, j, 0], preds[i, j, 1] = idx[i, j] % res, idx[i, j] / res
  
  return preds

def calcDists(preds, gt, normalize):
  dists = np.zeros((preds.shape[1], preds.shape[0]))
  for i in range(preds.shape[0]):
    for j in range(preds.shape[1]):
      if gt[i, j, 0] > 0 and gt[i, j, 1] > 0:
        dists[j][i] = ((gt[i][j] - preds[i][j]) ** 2).sum() ** 0.5 / normalize[i]
      else:
        dists[j][i] = -1
  return dists

def distAccuracy(dist, thr = 0.5):
  dist = dist[dist != -1]
  if len(dist) > 0:
    return 1.0 * (dist < thr).sum() / len(dist)
  else:
    return -1

def Accuracy(output, target):
  preds = getPreds(output)
  gt = getPreds(target)
  dists = calcDists(preds, gt, np.ones(preds.shape[0]) * ref.outputRes / 10)
  acc = np.zeros(len(ref.accIdxs))
  avgAcc = 0
  badIdxCount = 0
  
  for i in range(len(ref.accIdxs)):
    acc[i] = distAccuracy(dists[ref.accIdxs[i]])
    if acc[i] >= 0:
      avgAcc = avgAcc + acc[i]
    else:
      badIdxCount = badIdxCount + 1
  
  if badIdxCount == len(ref.accIdxs):
    return 0
  else:
    return avgAcc / (len(ref.accIdxs) - badIdxCount)

    
def finalPreds(output, center, scale, rotate):
  p = getPreds(output).copy()
  hm = output.reshape(output.shape[0], output.shape[1], ref.outputRes, ref.outputRes)
  for i in range(hm.shape[0]):
    for j in range(hm.shape[1]):
      pX, pY = int(p[i, j, 0]), int(p[i, j, 1])
      scores = hm[i, j, pX, pY]
      if pX > 0 and pX < ref.outputRes - 1 and pY > 0 and pY < ref.outputRes - 1:
        diffY = hm[i, j, pX, pY + 1] - hm[i, j, pX, pY - 1]
        diffX = hm[i, j, pX + 1, pY] - hm[i, j, pX - 1, pY]
        p[i, j, 0] = p[i, j, 0] + 0.25 * (1 if diffX >=0 else -1)
        p[i, j, 1] = p[i, j, 1] + 0.25 * (1 if diffY >=0 else -1)
  p = p + 0.5

  preds = np.zeros((p.shape[0], p.shape[1], 2))
  for i in range(p.shape[0]):
    for j in range(p.shape[1]):
      preds[i, j] = Transform(p[i, j], center[i], scale[i], rotate[i], ref.outputRes, invert = True)
  return preds
  
  

