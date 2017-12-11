
import numpy as np
import cv2
import ref
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
from mpl_toolkits.mplot3d import Axes3D

oo = 128
def show3D(ax, points, c = (255, 0, 0)):
  points = points.reshape(ref.nJoints, 3)
  #print 'show3D', c, points
  x, y, z = np.zeros((3, ref.nJoints))
  for j in range(ref.nJoints):
    x[j] = points[j, 0] 
    y[j] = - points[j, 1] 
    z[j] = - points[j, 2] 
  ax.scatter(z, x, y, c = c)
  for e in ref.edges:
    ax.plot(z[e], x[e], y[e], c = c)
  
def show2D(img, points, c):
  points = ((points.reshape(ref.nJoints, -1))).astype(np.int32)
  for j in range(ref.nJoints):
    cv2.circle(img, (points[j, 0], points[j, 1]), 3, c, -1)
  for e in ref.edges:
    cv2.line(img, (points[e[0], 0], points[e[0], 1]),
                  (points[e[1], 0], points[e[1], 1]), c, 2)
  return img

class Debugger(object):
  def __init__(self):
    self.plt = plt
    self.fig = self.plt.figure()
    self.ax = self.fig.add_subplot((111),projection='3d')
    self.ax.set_xlabel('z') 
    self.ax.set_ylabel('x') 
    self.ax.set_zlabel('y')
    self.xmax, self.ymax, self.zmax = oo, oo, oo
    self.xmin, self.ymin, self.zmin = -oo, -oo, -oo
    self.imgs = {}
  
  def addPoint3D(self, point, c = 'b'):
    show3D(self.ax, point, c)
  
  def show3D(self):
    max_range = np.array([self.xmax-self.xmin, self.ymax-self.ymin, self.zmax-self.zmin]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(self.xmax+self.xmin)
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(self.ymax+self.ymin)
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(self.zmax+self.zmin)
    for xb, yb, zb in zip(Xb, Yb, Zb):
      self.ax.plot([xb], [yb], [zb], 'w')
    self.plt.show()
    
  def addImg(self, img, imgId = 0):
    self.imgs[imgId] = img.copy()
  
  def addPoint2D(self, point, c, imgId = 0):
    self.imgs[imgId] = show2D(self.imgs[imgId], point, c)
  
  def showImg(self, pause = False, imgId = 0):
    cv2.imshow('{}'.format(imgId), self.imgs[imgId])
    if pause:
      cv2.waitKey()
      
  def saveImg(self, path = 'debug/debug.png', imgId = 0):
    cv2.imwrite(path, self.imgs[imgId])
    
  def showAllImg(self, pause = False):
    for i, v in self.imgs.items():
      cv2.imshow('{}'.format(i), v)
    if pause:
      cv2.waitKey()
    
