import glrenderer
import torch
import numpy as np


import ctypes
import torch
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
import torch.nn.functional as F
import torch.nn as nn
import sys
import matplotlib.pyplot as plt


obj_path = "/home/nishadg/mcs/ThreeDVision.jl/data/ycbv2/models/035_power_drill/textured_simple.obj"
texture_path = "/home/nishadg/mcs/ThreeDVision.jl/data/ycbv2/models/035_power_drill/texture_map.png"
pose = np.array([0.0, 0.0, 0.3, 1, 0, 0, 0])


renderer = glrenderer.GLRenderer(640, 480, 500.0, 500.0, 320.0, 240.0, 0.01, 10.0, "depth")
p = renderer.load_obj_parameters(obj_path)
renderer.load_object(*p, None)
depth = renderer.render([0], [pose])
plt.matshow(depth); plt.colorbar(); plt.show()



renderer = glrenderer.GLRenderer(640, 480, 500.0, 500.0, 320.0, 240.0, 0.01, 10.0, "rgb")
p = renderer.load_obj_parameters(obj_path)
renderer.load_object(*p, texture_path)
rgb, depth = renderer.render([0], [pose], [[1.0,0.0,0.0,1.0]])
rgb = cv2.cvtColor(rgb, cv2.COLOR_BGRA2RGBA)
plt.imshow(rgb); plt.colorbar(); plt.show()


renderer = glrenderer.GLRenderer(640, 480, 500.0, 500.0, 320.0, 240.0, 0.01, 10.0, "texture")
p = renderer.load_obj_parameters(obj_path)
renderer.load_object(*p, texture_path)
rgb, depth = renderer.render([0], [pose])
rgb = cv2.cvtColor(rgb, cv2.COLOR_BGRA2RGBA)
plt.imshow(rgb); plt.colorbar(); plt.show()
