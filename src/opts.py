import argparse
import os
import sys
import ref

class opts():
  def __init__(self):
    self.parser = argparse.ArgumentParser()
    
  def init(self):
    self.parser.add_argument('-expID', default = 'default', help = 'Experiment ID')
    self.parser.add_argument('-GPU', type = int, default = 0, help = 'GPU id')
    self.parser.add_argument('-test', action = 'store_true', help = 'test')
    self.parser.add_argument('-DEBUG', type = int, default = 0, help = 'DEBUG level')
    self.parser.add_argument('-demo', default = '', help = 'path/to/demo/image')
    
    self.parser.add_argument('-loadModel', default = 'none', help = 'Provide full path to a previously trained model')
    self.parser.add_argument('-nFeats', type = int, default = 256, help = '# features in the hourglass')
    self.parser.add_argument('-nStack', type = int, default = 2, help = '# hourglasses to stack')
    self.parser.add_argument('-nModules', type = int, default = 2, help = '# residual modules at each hourglass')
    self.parser.add_argument('-numOutput', type = int, default = ref.nJoints, help = '# ouput')
    
    self.parser.add_argument('-LR', type = float, default = 2.5e-4, help = 'Learning Rate')
    self.parser.add_argument('-dropLR', type = int, default = 1000000, help = 'drop LR')
    self.parser.add_argument('-nEpochs', type = int, default = 60, help = '#training epochs')
    self.parser.add_argument('-valIntervals', type = int, default = 5, help = '#valid intervel')
    self.parser.add_argument('-trainBatch', type = int, default = 6, help = 'Mini-batch size')
    self.parser.add_argument('-arch', default = 'hg', help = 'hg | hg-reg | resnet-xxx')
    
  def parse(self):
    self.init()  
    opt = self.parser.parse_args()
    opt.saveDir = os.path.join(ref.expDir, opt.expID)
    if opt.DEBUG > 0:
      ref.nThreads = 1

    if opt.test:
      opt.expID = opt.expID + 'TEST'
    opt.saveDir = os.path.join(ref.expDir, opt.expID)

    args = dict((name, getattr(opt, name)) for name in dir(opt)
                if not name.startswith('_'))
    refs = dict((name, getattr(ref, name)) for name in dir(ref)
                if not name.startswith('_'))

    if not os.path.exists(opt.saveDir):
      os.makedirs(opt.saveDir)
    file_name = os.path.join(opt.saveDir, 'opt.txt')
    with open(file_name, 'wt') as opt_file:
      opt_file.write('==> Cmd:\n')
      opt_file.write(str(sys.argv))
      opt_file.write('\n==> Opt:\n')
      for k, v in sorted(args.items()):
         opt_file.write('  %s: %s\n' % (str(k), str(v)))
      opt_file.write('==> Ref:\n')
      for k, v in sorted(refs.items()):
         opt_file.write('  %s: %s\n' % (str(k), str(v)))
    return opt
