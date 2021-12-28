import glrenderer
import numpy as np


import ctypes
from pprint import pprint
from PIL import Image
import glutils.glcontext as glcontext
import OpenGL.GL as GL
import cv2
import numpy as np
from pyassimp import *
from glutils.meshutil import perspective, lookat, xyz2mat, quat2rotmat, mat2xyz, safemat2quat
from transforms3d.quaternions import axangle2quat, mat2quat, qmult, qinverse
from transforms3d.euler import quat2euler, mat2euler, euler2quat
import CppRenderer
import matplotlib.pyplot as plt
import sys


obj_path = "../../test/035_power_drill/textured_simple.obj"
texture_path = "../../test/035_power_drill/texture_map.png"
pose = np.array([0.0, 0.0, 0.3, 1, 0, 0, 0])
pose2 = np.array([0.0, 0.2, 0.3, 1, 0, 0, 0])


renderer = glrenderer.GLRenderer(640, 480, 500.0, 500.0, 320.0, 240.0, 0.01, 10.0, "depth")
p = renderer.load_obj_parameters(obj_path)
renderer.load_object(*p, None)
depth1 = renderer.render([0], [pose])



renderer = glrenderer.GLRenderer(640, 480, 500.0, 500.0, 320.0, 240.0, 0.01, 10.0, "rgb")
p = renderer.load_obj_parameters(obj_path)
renderer.load_object(*p, texture_path)
rgb1, depth = renderer.render([0], [pose], [[1.0,0.0,0.0,1.0]])
rgb1 = cv2.cvtColor(rgb1, cv2.COLOR_BGRA2RGBA)


renderer = glrenderer.GLRenderer(640, 480, 500.0, 500.0, 320.0, 240.0, 0.01, 10.0, "rgb_basic")
p = renderer.load_obj_parameters(obj_path)
renderer.load_object(*p, texture_path)
rgb3, depth = renderer.render([0], [pose], [[1.0,0.0,0.0,1.0]])
rgb3 = cv2.cvtColor(rgb3, cv2.COLOR_BGRA2RGBA)



renderer = glrenderer.GLRenderer(640, 480, 500.0, 500.0, 320.0, 240.0, 0.01, 10.0, "texture")
p = renderer.load_obj_parameters(obj_path)
renderer.load_object(*p, texture_path)
rgb2, depth = renderer.render([0], [pose])
rgb2 = cv2.cvtColor(rgb2, cv2.COLOR_BGRA2RGBA)


renderer = glrenderer.GLRenderer(640, 480, 500.0, 500.0, 320.0, 240.0, 0.01, 10.0, "segmentation")
p = renderer.load_obj_parameters(obj_path)
renderer.load_object(*p, None)
renderer.load_object(*p, None)
depth5, seg5 = renderer.render([0,1], [pose, pose2])



plt.matshow(depth1); plt.colorbar(); plt.show()
plt.imshow(rgb1); plt.colorbar(); plt.show()
plt.imshow(rgb3); plt.colorbar(); plt.show()
plt.imshow(rgb2); plt.colorbar(); plt.show()
plt.matshow(seg5); plt.colorbar(); plt.show()
