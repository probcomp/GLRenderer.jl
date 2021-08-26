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
import os

renderer = glrenderer.GLRenderer(600, 600,
    300.0, 300.0,
    300.0,300.0,
    0.1, 60.0, "depth")

directory = "/home/nishadg/mcs/GLRenderer.jl/test"

v1 = np.load(os.path.join(directory,"v1.npz"))['arr_0']
v2 = np.load(os.path.join(directory,"v2.npz"))['arr_0']
v3 = np.load(os.path.join(directory,"v3.npz"))['arr_0']
v4 = np.load(os.path.join(directory,"v4.npz"))['arr_0']
f1 = np.load(os.path.join(directory,"f1.npz"))['arr_0']
f2 = np.load(os.path.join(directory,"f2.npz"))['arr_0']
f3 = np.load(os.path.join(directory,"f3.npz"))['arr_0']
f4 = np.load(os.path.join(directory,"f4.npz"))['arr_0']
n1 = np.load(os.path.join(directory,"n1.npz"))['arr_0']
n2 = np.load(os.path.join(directory,"n2.npz"))['arr_0']
n3 = np.load(os.path.join(directory,"n3.npz"))['arr_0']
n4 = np.load(os.path.join(directory,"n4.npz"))['arr_0']
print(f1.shape)
print(f2.shape)
print(f3.shape)
print(f4.shape)


renderer.load_object(v1, n1, f1, None, None)
renderer.load_object(v2, n2, f2, None, None)
renderer.load_object(v3, n3, f3, None, None)
renderer.load_object(v4, n4, f4, None, None)

V = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0, 10.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

renderer.V = V


identity_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
color = np.array([0.0, 0.0, 0.0, 1.0])


# rgb, depth = renderer.render([0,1,2,3], [identity_pose,identity_pose,identity_pose,identity_pose], colors=[color, color, color, color])
# rgb = cv2.cvtColor(rgb, cv2.COLOR_BGRA2RGBA)
# plt.imshow(rgb); plt.colorbar(); plt.show()

depth = renderer.render([0,1,2,3], [identity_pose,identity_pose,identity_pose,identity_pose], colors=[color, color, color, color])
plt.matshow(depth); plt.colorbar(); plt.show()
