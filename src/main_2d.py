import os
import time
import datetime

import torch
import torch.utils.data
from opts import opts
import ref
from models.hg import HourglassNet
from datasets.mpii import MPII
from utils.logger import Logger
from train_2d import train, val

def main():
  opt = opts().parse()
  now = datetime.datetime.now()
  logger = Logger(opt.saveDir + '/logs_{}'.format(now.isoformat()))

  if opt.loadModel != 'none':
    model = torch.load(opt.loadModel).cuda()
  else:
    model = HourglassNet(opt.nStack, opt.nModules, opt.nFeats).cuda()

  criterion = torch.nn.MSELoss()
  optimizer = torch.optim.RMSprop(model.parameters(), opt.LR, 
                                  alpha = ref.alpha, 
                                  eps = ref.epsilon, 
                                  weight_decay = ref.weightDecay, 
                                  momentum = ref.momentum)

  val_loader = torch.utils.data.DataLoader(
      MPII(opt, 'val'), 
      batch_size = 1, 
      shuffle = False,
      num_workers = int(ref.nThreads)
  )

  if opt.test:
    val(0, opt, val_loader, model, criterion)
    return

  train_loader = torch.utils.data.DataLoader(
      MPII(opt, 'train'), 
      batch_size = opt.trainBatch, 
      shuffle = True if opt.DEBUG == 0 else False,
      num_workers = int(ref.nThreads)
  )

  for epoch in range(1, opt.nEpochs + 1):
    loss_train, acc_train = train(epoch, opt, train_loader, model, criterion, optimizer)
    logger.scalar_summary('loss_train', loss_train, epoch)
    logger.scalar_summary('acc_train', acc_train, epoch)
    if epoch % opt.valIntervals == 0:
      loss_val, acc_val = val(epoch, opt, val_loader, model, criterion)
      logger.scalar_summary('loss_val', loss_val, epoch)
      logger.scalar_summary('acc_val', acc_val, epoch)
      torch.save(model, os.path.join(opt.saveDir, 'model_{}.pth'.format(epoch)))
      logger.write('{:8f} {:8f} {:8f} {:8f}\n'.format(loss_train, acc_train, loss_val, acc_val))
    else:
      logger.write('{:8f} {:8f}\n'.format(loss_train, acc_train))
  logger.close()


if __name__ == '__main__':
  main()
