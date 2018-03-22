import os
import time
import datetime

import torch
import torch.utils.data
from opts import opts
import ref
from model import getModel, saveModel
from datasets.mpii import MPII
from utils.logger import Logger
from train import train, val
import scipy.io as sio

def main():
  opt = opts().parse()
  now = datetime.datetime.now()
  logger = Logger(opt.saveDir, now.isoformat())
  model, optimizer = getModel(opt)
  criterion = torch.nn.MSELoss()

  if opt.GPU > -1:
    print 'Using GPU', opt.GPU
    model = model.cuda(opt.GPU)
    criterion = criterion.cuda(opt.GPU)


  val_loader = torch.utils.data.DataLoader(
      MPII(opt, 'val'), 
      batch_size = 1, 
      shuffle = False,
      num_workers = int(ref.nThreads)
  )

  if opt.test:
    log_dict_train, preds = val(0, opt, val_loader, model, criterion)
    sio.savemat(os.path.join(opt.saveDir, 'preds.mat'), mdict = {'preds': preds})
    return

  train_loader = torch.utils.data.DataLoader(
      MPII(opt, 'train'), 
      batch_size = opt.trainBatch, 
      shuffle = True if opt.DEBUG == 0 else False,
      num_workers = int(ref.nThreads)
  )

  for epoch in range(1, opt.nEpochs + 1):
    log_dict_train, _ = train(epoch, opt, train_loader, model, criterion, optimizer)
    for k, v in log_dict_train.items():
      logger.scalar_summary('train_{}'.format(k), v, epoch)
      logger.write('{} {:8f} | '.format(k, v))
    if epoch % opt.valIntervals == 0:
      log_dict_val, preds = val(epoch, opt, val_loader, model, criterion)
      for k, v in log_dict_val.items():
        logger.scalar_summary('val_{}'.format(k), v, epoch)
        logger.write('{} {:8f} | '.format(k, v))
      #saveModel(model, optimizer, os.path.join(opt.saveDir, 'model_{}.checkpoint'.format(epoch)))
      torch.save(model, os.path.join(opt.saveDir, 'model_{}.pth'.format(epoch)))
      sio.savemat(os.path.join(opt.saveDir, 'preds_{}.mat'.format(epoch)), mdict = {'preds': preds})
    logger.write('\n')
    if epoch % opt.dropLR == 0:
      lr = opt.LR * (0.1 ** (epoch // opt.dropLR))
      print 'Drop LR to', lr
      adjust_learning_rate(optimizer, lr)
  logger.close()
  torch.save(model.cpu(), os.path.join(opt.saveDir, 'model_cpu.pth'))

if __name__ == '__main__':
  main()
