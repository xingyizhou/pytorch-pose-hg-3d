from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os

import torch
import torch.utils.data
from opts import opts
from model import create_model, save_model
from datasets.mpii import MPII
from datasets.coco import COCO
from datasets.fusion_3d import Fusion3D
from logger import Logger
from train import train, val
from train_3d import train_3d, val_3d
import scipy.io as sio

dataset_factory = {
  'mpii': MPII,
  'coco': COCO,
  'fusion_3d': Fusion3D
}

task_factory = {
  'human2d': (train, val), 
  'human3d': (train_3d, val_3d)
}

def main(opt):
  if opt.disable_cudnn:
    torch.backends.cudnn.enabled = False
    print('Cudnn is disabled.')

  logger = Logger(opt)
  opt.device = torch.device('cuda:{}'.format(opt.gpus[0]))

  Dataset = dataset_factory[opt.dataset]
  train, val = task_factory[opt.task]

  model, optimizer, start_epoch = create_model(opt)
 
  if len(opt.gpus) > 1:
    model = torch.nn.DataParallel(model, device_ids=opt.gpus).cuda(opt.device)
  else:
    model = model.cuda(opt.device)

  val_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'val'), 
      batch_size=1, 
      shuffle=False,
      num_workers=1,
      pin_memory=True
  )

  if opt.test:
    log_dict_train, preds = val(0, opt, val_loader, model)
    sio.savemat(os.path.join(opt.save_dir, 'preds.mat'),
                mdict = {'preds': preds})
    return

  train_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'train'), 
      batch_size=opt.batch_size * len(opt.gpus), 
      shuffle=True, # if opt.debug == 0 else False,
      num_workers=opt.num_workers,
      pin_memory=True
  )
  
  best = -1
  for epoch in range(start_epoch, opt.num_epochs + 1):
    mark = epoch if opt.save_all_models else 'last'
    log_dict_train, _ = train(epoch, opt, train_loader, model, optimizer)
    for k, v in log_dict_train.items():
      logger.scalar_summary('train_{}'.format(k), v, epoch)
      logger.write('{} {:8f} | '.format(k, v))
    if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)), 
                 epoch, model, optimizer)
      log_dict_val, preds = val(epoch, opt, val_loader, model)
      for k, v in log_dict_val.items():
        logger.scalar_summary('val_{}'.format(k), v, epoch)
        logger.write('{} {:8f} | '.format(k, v))
      if log_dict_val[opt.metric] > best:
        best = log_dict_val[opt.metric]
        save_model(os.path.join(opt.save_dir, 'model_best.pth'), 
                   epoch, model)
    else:
      save_model(os.path.join(opt.save_dir, 'model_last.pth'), 
                 epoch, model, optimizer)
    logger.write('\n')
    if epoch in opt.lr_step:
      lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
      print('Drop LR to', lr)
      for param_group in optimizer.param_groups:
          param_group['lr'] = lr
  logger.close()

if __name__ == '__main__':
  opt = opts().parse()
  main(opt)
