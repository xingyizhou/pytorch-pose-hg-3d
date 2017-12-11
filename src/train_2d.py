import torch
import numpy as np
from utils.utils import AverageMeter
from utils.eval import Accuracy, getPreds
import cv2
import ref
from progress.bar import Bar
from utils.debugger import Debugger

def step(split, epoch, opt, dataLoader, model, criterion, optimizer = None):
  if split == 'train':
    model.train()
  else:
    model.eval()
  Loss, Acc = AverageMeter(), AverageMeter()
  
  nIters = len(dataLoader)
  bar = Bar('==>', max=nIters)
  
  for i, (input, target) in enumerate(dataLoader):
    input_var = torch.autograd.Variable(input).float().cuda()
    target_var = torch.autograd.Variable(target.cuda(async = True)).float().cuda()
    output = model(input_var)
    
    if opt.DEBUG >= 2:
      gt = getPreds(target.cpu().numpy()) * 4
      pred = getPreds((output[opt.nStack - 1].data).cpu().numpy()) * 4
      init = getPreds(input.numpy()[:, 3:])
      debugger = Debugger()
      img = (input[0].numpy()[:3].transpose(1, 2, 0)*256).astype(np.uint8).copy()
      debugger.addImg(img)
      debugger.addPoint2D(pred[0], (255, 0, 0))
      debugger.addPoint2D(gt[0], (0, 0, 255))
      debugger.addPoint2D(init[0], (0, 255, 0))
      debugger.showAllImg(pause = True)
      #debugger.saveImg('debug/{}.png'.format(i))
    
    loss = criterion(output[0], target_var)
    for k in range(1, opt.nStack):
      loss += criterion(output[k], target_var)

    Loss.update(loss.data[0], input.size(0))
    Acc.update(Accuracy((output[opt.nStack - 1].data).cpu().numpy(), (target_var.data).cpu().numpy()))
    if split == 'train':
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
 
    Bar.suffix = '{split} Epoch: [{0}][{1}/{2}]| Total: {total:} | ETA: {eta:} | Loss {loss.avg:.6f} | Acc {Acc.avg:.6f} ({Acc.val:.6f})'.format(epoch, i, nIters, total=bar.elapsed_td, eta=bar.eta_td, loss=Loss, Acc=Acc, split = split)
    bar.next()

  bar.finish()
  return Loss.avg, Acc.avg
  

def train(epoch, opt, train_loader, model, criterion, optimizer):
  return step('train', epoch, opt, train_loader, model, criterion, optimizer)
  
def val(epoch, opt, val_loader, model, criterion):
  return step('val', epoch, opt, val_loader, model, criterion)
