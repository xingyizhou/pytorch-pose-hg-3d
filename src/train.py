import torch
import numpy as np
from utils.utils import AverageMeter, Flip, ShuffleLR
from utils.eval import Accuracy, getPreds, finalPreds
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
  preds = []
  
  nIters = len(dataLoader)
  bar = Bar('{}'.format(opt.expID), max=nIters)
  
  for i, (input, target, meta) in enumerate(dataLoader):
    input_var = torch.autograd.Variable(input).float().cuda(opt.GPU)
    target_var = torch.autograd.Variable(target).float().cuda(opt.GPU)
    output = model(input_var)
    
    if opt.DEBUG >= 2:
      gt = getPreds(target.cpu().numpy()) * 4
      pred = getPreds((output[opt.nStack - 1].data).cpu().numpy()) * 4
      debugger = Debugger()
      img = (input[0].numpy().transpose(1, 2, 0)*256).astype(np.uint8).copy()
      debugger.addImg(img)
      debugger.addPoint2D(pred[0], (255, 0, 0))
      debugger.addPoint2D(gt[0], (0, 0, 255))
      debugger.showAllImg(pause = True)
    
    loss = criterion(output[0], target_var)
    for k in range(1, opt.nStack):
      loss += criterion(output[k], target_var)

    Loss.update(loss.data[0], input.size(0))
    Acc.update(Accuracy((output[opt.nStack - 1].data).cpu().numpy(), (target_var.data).cpu().numpy()))
    if split == 'train':
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    else:
      input_ = input.cpu().numpy()
      input_[0] = Flip(input_[0]).copy()
      inputFlip_var = torch.autograd.Variable(torch.from_numpy(input_).view(1, input_.shape[1], ref.inputRes, ref.inputRes)).float().cuda(opt.GPU)
      outputFlip = model(inputFlip_var)
      outputFlip = ShuffleLR(Flip((outputFlip[opt.nStack - 1].data).cpu().numpy()[0])).reshape(1, ref.nJoints, ref.outputRes, ref.outputRes)
      output_ = ((output[opt.nStack - 1].data).cpu().numpy() + outputFlip) / 2
      preds.append(finalPreds(output_, meta['center'], meta['scale'], meta['rotate'])[0])
      
    Bar.suffix = '{split} Epoch: [{0}][{1}/{2}]| Total: {total:} | ETA: {eta:} | Loss {loss.avg:.6f} | Acc {Acc.avg:.6f} ({Acc.val:.6f})'.format(epoch, i, nIters, total=bar.elapsed_td, eta=bar.eta_td, loss=Loss, Acc=Acc, split = split)
    bar.next()

  bar.finish()
  return {'Loss': Loss.avg, 'Acc': Acc.avg}, preds
  

def train(epoch, opt, train_loader, model, criterion, optimizer):
  return step('train', epoch, opt, train_loader, model, criterion, optimizer)
  
def val(epoch, opt, val_loader, model, criterion):
  return step('val', epoch, opt, val_loader, model, criterion)
