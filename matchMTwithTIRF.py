#https://github.com/toros-astro/astroalign-legacy
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
# import astroalign as aa
from scipy import ndimage as ndi
small = Image.open('03-orange.png')
mat = np.array([[1,0,0],[0,1,0],[0,0,1]])
small2 = ndi.affine_transform(small, mat)
big = Image.open('03-MT.png')
ws, hs= small.size
w, h = big.size
s_x, s_y = 0.961, 0.961 #0.961
mat_scale = np.array([[1/s_x,0,0],[0,1/s_y,0],[0,0,1]])
big2 = ndi.affine_transform(big, mat_scale)
theta = 0.5 * np.pi/360
mat_rotate = np.array([[1,0,w/2],[0,1,h/2],[0,0,1]]) @ np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]]) @ np.array([[1,0,-w/2],[0,1,-h/2],[0,0,1]])
big2 = ndi.affine_transform(big2, mat_rotate)
ty = 110#107
tx = 891#891
big3= big2[ty:ty+hs,tx:tx+ws]
img = np.asarray([small2/256, big3/256, np.zeros([hs,ws])]).transpose(1,2,0)
plt.imshow(img)
plt.show()