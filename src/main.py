import os
import time
import datetime

import torch
import torch.utils.data
from opts import opts
import ref
from models.hg_3d import HourglassNet3D
from utils.utils import adjust_learning_rate
from datasets.fusion import Fusion
from datasets.h36m import H36M
from datasets.mpii import MPII
from utils.logger import Logger
from train import train, val

def main():
  opt = opts().parse()
  now = datetime.datetime.now()
  logger = Logger(opt.saveDir + '/logs_{}'.format(now.isoformat()))

  if opt.loadModel != 'none':
    model = torch.load(opt.loadModel).cuda()
  else:
    model = HourglassNet3D(opt.nStack, opt.nModules, opt.nFeats, opt.nRegModules).cuda()
  
  criterion = torch.nn.MSELoss().cuda()
  optimizer = torch.optim.RMSprop(model.parameters(), opt.LR, 
                                  alpha = ref.alpha, 
                                  eps = ref.epsilon, 
                                  weight_decay = ref.weightDecay, 
                                  momentum = ref.momentum)

  if opt.ratio3D < ref.eps:
    val_loader = torch.utils.data.DataLoader(
        MPII(opt, 'val', returnMeta = True), 
        batch_size = 1, 
        shuffle = False,
        num_workers = int(ref.nThreads)
    )
  else:
    val_loader = torch.utils.data.DataLoader(
        H36M(opt, 'val'), 
        batch_size = 1, 
        shuffle = False,
        num_workers = int(ref.nThreads)
    )
  

  if opt.test:
    val(0, opt, val_loader, model, criterion)
    return

  train_loader = torch.utils.data.DataLoader(
      Fusion(opt, 'train'), 
      batch_size = opt.trainBatch, 
      shuffle = True if opt.DEBUG == 0 else False,
      num_workers = int(ref.nThreads)
  )

  for epoch in range(1, opt.nEpochs + 1):
    loss_train, acc_train, mpjpe_train, loss3d_train = train(epoch, opt, train_loader, model, criterion, optimizer)
    logger.scalar_summary('loss_train', loss_train, epoch)
    logger.scalar_summary('acc_train', acc_train, epoch)
    logger.scalar_summary('mpjpe_train', mpjpe_train, epoch)
    logger.scalar_summary('loss3d_train', loss3d_train, epoch)
    if epoch % opt.valIntervals == 0:
      loss_val, acc_val, mpjpe_val, loss3d_val = val(epoch, opt, val_loader, model, criterion)
      logger.scalar_summary('loss_val', loss_val, epoch)
      logger.scalar_summary('acc_val', acc_val, epoch)
      logger.scalar_summary('mpjpe_val', mpjpe_val, epoch)
      logger.scalar_summary('loss3d_val', loss3d_val, epoch)
      torch.save(model, os.path.join(opt.saveDir, 'model_{}.pth'.format(epoch)))
      logger.write('{:8f} {:8f} {:8f} {:8f} {:8f} {:8f} {:8f} {:8f} \n'.format(loss_train, acc_train, mpjpe_train, loss3d_train, loss_val, acc_val, mpjpe_val, loss3d_val))
    else:
      logger.write('{:8f} {:8f} {:8f} {:8f} \n'.format(loss_train, acc_train, mpjpe_train, loss3d_train))
    adjust_learning_rate(optimizer, epoch, opt.dropLR, opt.LR)
  logger.close()

if __name__ == '__main__':
  main()
