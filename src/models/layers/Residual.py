import torch.nn as nn

class Residual(nn.Module):
  def __init__(self, numIn, numOut):
    super(Residual, self).__init__()
    self.numIn = numIn
    self.numOut = numOut
    self.bn = nn.BatchNorm2d(self.numIn)
    self.relu = nn.ReLU(inplace = True)
    self.conv1 = nn.Conv2d(self.numIn, self.numOut / 2, bias = True, kernel_size = 1)
    self.bn1 = nn.BatchNorm2d(self.numOut / 2)
    self.conv2 = nn.Conv2d(self.numOut / 2, self.numOut / 2, bias = True, kernel_size = 3, stride = 1, padding = 1)
    self.bn2 = nn.BatchNorm2d(self.numOut / 2)
    self.conv3 = nn.Conv2d(self.numOut / 2, self.numOut, bias = True, kernel_size = 1)
    
    if self.numIn != self.numOut:
      self.conv4 = nn.Conv2d(self.numIn, self.numOut, bias = True, kernel_size = 1) 
    
  def forward(self, x):
    residual = x
    out = self.bn(x)
    out = self.relu(out)
    out = self.conv1(out)
    out = self.bn1(out)
    out = self.relu(out)
    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)
    out = self.conv3(out)
    
    if self.numIn != self.numOut:
      residual = self.conv4(x)
    
    return out + residual
    
