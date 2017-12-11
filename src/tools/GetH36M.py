import scipy.io as sio
import numpy as np
import h5py
import os
J = 16
inds = [3, 2, 1, 4, 5, 6, 0, 7, 8, 10, 16, 15, 14, 11, 12, 13]
subject_list = [[1, 5, 6, 7, 8], [9, 11]]
action_list = np.arange(2, 17)
subaction_list = np.arange(1, 3)
camera_list = np.arange(1, 5)
IMG_PATH = '/home/zxy/Datasets/Human3.6M/images/'
SAVE_PATH = '../../data/h36m/'
annot_name = 'matlab_meta.mat'

if not os.path.exists(SAVE_PATH):
  os.mkdir(SAVE_PATH)

id = []
joint_2d = []
joint_3d_mono = []
bbox = []
subjects = []
actions = []
subactions = []
cameras = []
istrain = []
num = 0

for split in range(2):

  for subject in subject_list[split]:
    for action in action_list:
      for subaction in subaction_list:
        for camera in camera_list:
          folder_name = 's_{:02d}_act_{:02d}_subact_{:02d}_ca_{:02d}'.format(subject, action, subaction, camera)
          print folder_name
          annot_file = IMG_PATH + folder_name + '/' + annot_name
          try:
            data = sio.loadmat(annot_file)
          except:
            print 'pass', folder_name
            continue
          n = data['num_images'][0][0]
          meta_Y2d = data['Y2d'].reshape(17, 2, n)
          meta_Y3d_mono = data['Y3d_mono'].reshape(17, 3, n)
          bboxx = data['bbox'].transpose(1, 0)
          for i in range(data['num_images']):
            if i % 5 != 0:
              continue
            if split == 1 and i % 200 != 0:
              continue
            id.append(i + 1)
            joint_2d.append(meta_Y2d[inds, :, i])
            joint_3d_mono.append(meta_Y3d_mono[inds, :, i])
            bbox.append(bboxx[i])
            subjects.append(subject)
            actions.append(action)
            subactions.append(subaction)
            cameras.append(camera)
            istrain.append(1 - split)
            num += 1
          
print 'num = ', num
h5name = SAVE_PATH + 'annotSampleTest.h5'
f = h5py.File(h5name, 'w')
f['id'] = id
f['joint_2d'] = joint_2d
f['joint_3d_mono'] = joint_3d_mono
f['bbox'] = bbox
f['subject'] = subjects
f['action'] = actions
f['subaction'] = subactions
f['camera'] = cameras
f['istrain'] = istrain
f.close()
  
