import torch
from torch.autograd import Function
import numpy as np
import ref

class FusionCriterion(Function):
  def __init__(self, regWeight, varWeight):
    super(FusionCriterion, self).__init__()
    self.regWeight = regWeight
    self.varWeight = varWeight
   
    self.skeletonRef = [[[0,1],    [1,2],
                         [3,4],    [4,5]],
                        [[10,11],  [11,12],
                         [13,14],  [14,15]], 
                         [[2, 6], [3, 6]], 
                         [[12,8], [13,8]]]
    self.skeletonWeight = [[1.0085885098415446, 1, 
                            1, 1.0085885098415446], 
                           [1.1375361376887123, 1, 
                            1, 1.1375361376887123], 
                           [1, 1], 
                           [1, 1]]

    
  def forward(self, input, target_):
    target = target_.view(target_.size(0), ref.nJoints, 3)
    xy = target[:, :, :2]
    z = target[:, :, 2]
    batchSize = target.size(0)
    output = torch.FloatTensor(1) * 0
    for t in range(batchSize):
      s = xy[t].sum()
      if s < ref.eps and s > - ref.eps: #Sup data
        loss = ((input[t] - z[t]) ** 2).sum() / ref.nJoints
        output += self.regWeight * loss
      else:
        xy[t] = 2.0 * xy[t] / ref.outputRes - 1
        for g in range(len(self.skeletonRef)):
          E, num = 0, 0
          N = len(self.skeletonRef[g])
          l = np.zeros(N)
          for j in range(N):
            id1, id2 = self.skeletonRef[g][j]
            if z[t, id1] > 0.5 and z[t, id2] > 0.5:
              l[j] = (((xy[t, id1] - xy[t, id2]) ** 2).sum() + (input[t, id1] - input[t, id2]) ** 2) ** 0.5
              l[j] = l[j] * self.skeletonWeight[g][j]
              num += 1
              E += l[j]
          if num < 0.5:
            E = 0
          else:
            E = E / num
          loss = 0
          for j in range(N):
            if l[j] > 0:
              loss += (l[j] - E) ** 2 / num
          output += self.varWeight * loss 
    output = output / batchSize
    self.save_for_backward(input, target_)
    return output.cuda()
    
  def backward(self, grad_output):
    input, target = self.saved_tensors
    target = target.view(target.size(0), ref.nJoints, 3)
    xy = target[:, :, :2]
    z = target[:, :, 2]
    grad_input = torch.zeros(input.size())
    batchSize = target.size(0)
    for t in range(batchSize):
      s = xy[t].sum()
      if s < ref.eps and s > - ref.eps:
        grad_input[t] += grad_output[0] * self.regWeight / batchSize * 2 / ref.nJoints * (input[t] - z[t]).cpu()
      else:
        xy[t] = 2.0 * xy[t] / ref.outputRes - 1
        for g in range(len(self.skeletonRef)):
          E, num = 0, 0
          N = len(self.skeletonRef[g])
          l = np.zeros(N)
          for j in range(N):
            id1, id2 = self.skeletonRef[g][j]
            if z[t, id1] > 0.5 and z[t, id2] > 0.5:
              l[j] = (((xy[t, id1] - xy[t, id2]) ** 2).sum() + (input[t, id1] - input[t, id2]) ** 2) ** 0.5
              l[j] = l[j] * self.skeletonWeight[g][j]
              num += 1
              E += l[j]
          if num < 0.5:
            E = 0
          else:
            E = E / num
          for j in range(N):
            if l[j] > 0:
              id1, id2 = self.skeletonRef[g][j]
              grad_input[t][id1] += 2 * self.varWeight * self.skeletonWeight[g][j] ** 2 / num * (l[j] - E) / l[j] * (input[t, id1] - input[t, id2]) / batchSize
              grad_input[t][id2] += 2 * self.varWeight * self.skeletonWeight[g][j] ** 2 / num * (l[j] - E) / l[j] * (input[t, id2] - input[t, id1]) / batchSize
    return grad_input.cuda(), None
    
    
    
    
