import torchvision.models as models
import torch
import torch.nn as nn
import os

from models.msra_resnet import get_pose_net

def create_model(opt): 
  if 'msra' in opt.arch:
    print("=> using msra resnet '{}'".format(opt.arch))
    num_layers = int(opt.arch[opt.arch.find('_') + 1:])
    model = get_pose_net(num_layers, opt.heads)
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
  else:
    assert 0, "Model not supported!"
    
  start_epoch = 1
  if opt.load_model != '':
    checkpoint = torch.load(
      opt.load_model, map_location=lambda storage, loc: storage)
    print('loaded {}, epoch {}'.format(opt.load_model, checkpoint['epoch']))
    if type(checkpoint) == type({}):
      state_dict = checkpoint['state_dict']
    else:
      state_dict = checkpoint.state_dict()
    model.load_state_dict(state_dict, strict=False)
    if opt.resume:
      print('resuming optimizer')
      optimizer.load_state_dict(checkpoint['optimizer'])
      start_epoch = checkpoint['epoch'] + 1
      for state in optimizer.state.values():
        for k, v in state.items():
          if isinstance(v, torch.Tensor):
            state[k] = v.cuda(opt.device, non_blocking=True)

  return model, optimizer, start_epoch
  
def save_model(path, epoch, model, optimizer=None):
  if isinstance(model, torch.nn.DataParallel):
    state_dict = model.module.state_dict()
  else:
    state_dict = model.state_dict()
  data = {'epoch': epoch,
          'state_dict': state_dict}
  if not (optimizer is None):
    data['optimizer'] = optimizer.state_dict()
  torch.save(data, path)
