
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

try:
    from .get_available_devices import *
except:
    from get_available_devices import *

MAX_NUM_OBJECTS = 3
from glutils.utils import colormap


def loadTexture(path):
    img = Image.open(path).transpose(Image.FLIP_TOP_BOTTOM)
    img_data = np.fromstring(img.tobytes(), np.uint8)
    # print(img_data.shape)
    width, height = img.size
    # glTexImage2D expects the first element of the image data to be the
    # bottom-left corner of the image.  Subsequent elements go left to right,
    # with subsequent lines going from bottom to top.

    # However, the image data was created with PIL Image tostring and numpy's
    # fromstring, which means we have to do a bit of reorganization. The first
    # element in the data output by tostring() will be the top-left corner of
    # the image, with following values going left-to-right and lines going
    # top-to-bottom.  So, we need to flip the vertical coordinate (y).
    texture = GL.glGenTextures(1)
    GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
    GL.glTexParameterf(
        GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
    GL.glTexParameterf(
        GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR_MIPMAP_LINEAR)
    GL.glTexParameterf(
        GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
    GL.glTexParameterf(
        GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
    GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB, width, height, 0,
                    GL.GL_RGB, GL.GL_UNSIGNED_BYTE, img_data)
    GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
    return texture


class GLRenderer:
    def __init__(self, width, height, fx, fy, cx, cy, near, far, render_type, gpu_id=0):
        if render_type is None:
            render_type = "depth"

        self.VAOs = []
        self.VBOs = []
        self.objects = []
        self.width = width
        self.height = height
        self.near = near
        self.far = far
        self.render_type = render_type

        self.data = dict()

        skew = 0.0
        proj = np.eye(5, dtype=np.float32)
        proj[1, 1] = fx
        proj[2, 2] = fy
        proj[1, 2] = skew
        proj[1, 3] = -cx
        proj[2, 3] = -cy
        proj[3, 3] = near + far
        proj[3, 4] = near * far
        proj[4, 4] = 0.0
        proj[4, 3] = -1.0
        proj = proj[1:,1:]


        left, right, bottom, top, near, far = 0, width, 0, height, near, far
        ortho = np.eye(5, dtype=np.float32)
        ortho[1, 1] = 2.0 / (right-left)
        ortho[2, 2] = 2.0 / (top-bottom)
        ortho[3, 3] = - 2.0 / (far - near)
        ortho[1, 4] = - (right + left) / (right - left)
        ortho[2, 4] = - (top + bottom) / (top - bottom)
        ortho[3, 4] = - (far + near) / (far - near)
        ortho = ortho[1:,1:]

        self.option1 = (np.dot(ortho, proj))

        def set_projection_matrix(w, h, fu, fv, u0, v0, znear, zfar):
            L = -(u0) * znear / fu;
            R = +(w - u0) * znear / fu;
            T = -(v0) * znear / fv;
            B = +(h - v0) * znear / fv;

            P = np.zeros((4, 4), dtype=np.float32);
            P[0, 0] = 2 * znear / (R - L);
            P[1, 1] = 2 * znear / (T - B);
            P[2, 0] = (R + L) / (L - R);
            P[2, 1] = (T + B) / (B - T);
            P[2, 2] = (zfar + znear) / (zfar - znear);
            P[2, 3] = 1.0;
            P[3, 2] = (2 * zfar * znear) / (znear - zfar);
            return P
        self.option2 = set_projection_matrix(width, height, fx, fy, cx, cy, near, far)

        self.P = set_projection_matrix(width, height, fx, fy, cx, cy, near, far)
        # self.P = np.ascontiguousarray(np.dot(ortho, proj), np.float32)
        self.P = np.ascontiguousarray(self.P, np.float32)
        self.V = np.ascontiguousarray(np.eye(4, dtype=np.float32), np.float32)
        self.meshes = []
        self.faces = []
        self.textures = []

        self.r = CppRenderer.CppRenderer(self.width, self.height, get_available_devices()[gpu_id])
        self.r.init()

        self.glstring = GL.glGetString(GL.GL_VERSION)
        from OpenGL.GL import shaders

        self.shaders = shaders
        self.colors = [[0.01, 0.1, 0.9], [0.02, 0.2, 0.8], [0.03, 0.3, 0.7], [0.04, 0.4, 0.6], [0.05, 0.5, 0.5], [0.06, 0.6, 0.4], [0.07, 0.7, 0.3],
                       [0.08, 0.8, 0.2], [0.09, 0.9, 0.1], [0.1, 1.0, 0.0], [0.11, 0.1, 0.7], [0.12, 0.2, 0.6], [0.13, 0.3, 0.5], [0.14, 0.4, 0.4],
                       [0.15, 0.5, 0.3], [0.16, 0.6, 0.2], [0.17, 0.7, 0.1], [0.18, 0.8, 0.0], [0.19, 0.5, 0.7], [0.20, 0.4, 0.8], [0.21, 0.3, 0.9]]
        self.lightpos = [0, 0, 0]
        self.lightcolor = [1, 1, 1]

        vertexShader_rgb = self.shaders.compileShader("""
        #version 460
        uniform mat4 V;
        uniform mat4 P;
        uniform mat4 pose_rot;
        uniform mat4 pose_trans;
        uniform vec3 instance_color; 
        uniform vec4 color; 
        layout (location=0) in vec3 position;
        layout (location=1) in vec3 normal;
        out vec4 theColor;
        out vec3 Normal;
        out vec3 FragPos;
        out vec3 Normal_cam;
        out vec3 Instance_color;
        out vec3 Pos_cam;
        out vec3 Pos_obj;
        void main() {
            gl_Position = P * V * pose_trans * pose_rot * vec4(position, 1);
            vec4 world_position4 = pose_trans * pose_rot * vec4(position, 1);
            FragPos = vec3(world_position4.xyz / world_position4.w); // in world coordinate
            Normal = normalize(mat3(pose_rot) * normal); // in world coordinate
            Normal_cam = normalize(mat3(V) * mat3(pose_rot) * normal); // in camera coordinate
            vec4 pos_cam4 = V * pose_trans * pose_rot * vec4(position, 1);
            Pos_cam = pos_cam4.xyz / pos_cam4.w;
            Pos_obj = position;
            theColor = color;
            Instance_color = instance_color;
        }
        """, GL.GL_VERTEX_SHADER)

        fragmentShader_rgb = self.shaders.compileShader("""
        #version 460
        in vec4 theColor;
        in vec3 Normal;
        in vec3 Normal_cam;
        in vec3 FragPos;
        in vec3 Instance_color;
        in vec3 Pos_cam;
        in vec3 Pos_obj;
        layout (location = 0) out vec4 outputColour;
        uniform vec3 light_position;  // in world coordinate
        uniform vec3 light_color; // light color
        void main() {
            float ambientStrength = 0.6;
            vec3 ambient = ambientStrength * light_color;
            vec3 lightDir = normalize(light_position - FragPos);
            float diff = max(dot(Normal, lightDir), 0.0);
            vec3 diffuse = diff * light_color;
            outputColour =  vec4(theColor) * vec4(diffuse + ambient, 1);
        }
        """, GL.GL_FRAGMENT_SHADER)

        self.shaderProgram_rgb = self.shaders.compileProgram(vertexShader_rgb,
                                                                     fragmentShader_rgb)
        vertexShader_depth = self.shaders.compileShader("""
        #version 460
        uniform mat4 V;
        uniform mat4 P;
        uniform mat4 pose_rot;
        uniform mat4 pose_trans;
        layout (location=0) in vec3 position;
        void main() {
            gl_Position = P * V * pose_trans * pose_rot * vec4(position, 1);
        }
        """, GL.GL_VERTEX_SHADER)

        fragmentShader_depth = self.shaders.compileShader("""
        #version 460
        out vec4 outColor;
        void main()
        {
            outColor = vec4(1.0, 1.0, 1.0, 1.0);
        }
        """, GL.GL_FRAGMENT_SHADER)

        self.shaderProgram_depth = self.shaders.compileProgram(vertexShader_depth,
                                                               fragmentShader_depth)

        vertexShader_texture = self.shaders.compileShader("""
        #version 460
        uniform mat4 V;
        uniform mat4 P;
        uniform mat4 pose_rot;
        uniform mat4 pose_trans;
        uniform vec3 instance_color; 
                
        layout (location=0) in vec3 position;
        layout (location=1) in vec3 normal;
        layout (location=2) in vec2 texCoords;
        out vec2 theCoords;
        out vec3 Normal;
        out vec3 FragPos;
        out vec3 Normal_cam;
        out vec3 Instance_color;
        out vec3 Pos_cam;
        out vec3 Pos_obj;
        void main() {
            gl_Position = P * V * pose_trans * pose_rot * vec4(position, 1);
            vec4 world_position4 = pose_trans * pose_rot * vec4(position, 1);
            FragPos = vec3(world_position4.xyz / world_position4.w); // in world coordinate
            Normal = normalize(mat3(pose_rot) * normal); // in world coordinate
            Normal_cam = normalize(mat3(V) * mat3(pose_rot) * normal); // in camera coordinate
            vec4 pos_cam4 = V * pose_trans * pose_rot * vec4(position, 1);
            Pos_cam = pos_cam4.xyz / pos_cam4.w;
            Pos_obj = position;
            theCoords = texCoords;
            Instance_color = instance_color;
        }
        """, GL.GL_VERTEX_SHADER)

        fragmentShader_texture = self.shaders.compileShader("""
        #version 460
        uniform sampler2D texUnit;
        in vec2 theCoords;
        in vec3 Normal;
        in vec3 Normal_cam;
        in vec3 FragPos;
        in vec3 Instance_color;
        in vec3 Pos_cam;
        in vec3 Pos_obj;
        layout (location = 0) out vec4 outputColour;
        layout (location = 1) out vec4 NormalColour;
        layout (location = 2) out vec4 InstanceColour;
        layout (location = 3) out vec4 PCObject;
        layout (location = 4) out vec4 PCColour;
        uniform vec3 light_position;  // in world coordinate
        uniform vec3 light_color; // light color
        void main() {
            float ambientStrength = 0.2;
            vec3 ambient = ambientStrength * light_color;
            vec3 lightDir = normalize(light_position - FragPos);
            float diff = max(dot(Normal, lightDir), 0.0);
            vec3 diffuse = diff * light_color;
            outputColour =  texture(texUnit, theCoords) * vec4(diffuse + ambient, 1);
            NormalColour =  vec4((Normal_cam + 1) / 2,1);
            InstanceColour = vec4(Instance_color,1);
            PCObject = vec4(Pos_obj,1);
            PCColour = vec4(Pos_cam,1);
        }
        """, GL.GL_FRAGMENT_SHADER)

        self.shaderProgram_texture = self.shaders.compileProgram(vertexShader_texture,
                                                               fragmentShader_texture)
        self.texUnitUniform = GL.glGetUniformLocation(self.shaderProgram_texture, 'texUnit')


        self.fbo = GL.glGenFramebuffers(1)
        self.color_tex = GL.glGenTextures(1)
        self.depth_tex = GL.glGenTextures(1)

        GL.glBindTexture(GL.GL_TEXTURE_2D, self.color_tex)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA32F, self.width, self.height, 0,
                        GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.depth_tex)
        GL.glTexImage2D.wrappedOperation(
            GL.GL_TEXTURE_2D, 0, GL.GL_DEPTH24_STENCIL8, self.width, self.height, 0,
            GL.GL_DEPTH_STENCIL, GL.GL_UNSIGNED_INT_24_8, None)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, self.color_tex, 0)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_STENCIL_ATTACHMENT, GL.GL_TEXTURE_2D, self.depth_tex,
                                  0)
        GL.glViewport(0, 0, self.width, self.height)
        GL.glDrawBuffers(5, [GL.GL_COLOR_ATTACHMENT0])

        assert GL.glCheckFramebufferStatus(
            GL.GL_FRAMEBUFFER) == GL.GL_FRAMEBUFFER_COMPLETE

    def load_obj_parameters(self, obj_path):
        scene = load(obj_path)
        mesh = scene.meshes[0]
        release(scene)
        return mesh.vertices, mesh.normals, mesh.faces, mesh.texturecoords[0, :, :2]

    def load_object(self, vertices, normals, faces, texture_coords, texture_path):
        if self.render_type == "depth":
            vertexData = vertices.astype(np.float32)

            VAO = GL.glGenVertexArrays(1)
            GL.glBindVertexArray(VAO)

            # Need VBO for triangle vertices and texture UV coordinates
            VBO = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, VBO)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, vertexData.nbytes, vertexData, GL.GL_STATIC_DRAW)

            positionAttrib = GL.glGetAttribLocation(self.shaderProgram_depth, 'position')
            
            GL.glEnableVertexAttribArray(0)
            GL.glEnableVertexAttribArray(1)

            GL.glVertexAttribPointer(positionAttrib, 3, GL.GL_FLOAT, GL.GL_FALSE, 12, None)

            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
            GL.glBindVertexArray(0)
        elif self.render_type == "rgb":
            vertices = np.concatenate([vertices, normals], axis=-1)
            vertexData = vertices.astype(np.float32)

            VAO = GL.glGenVertexArrays(1)
            GL.glBindVertexArray(VAO)

            # Need VBO for triangle vertices and texture UV coordinates
            VBO = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, VBO)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, vertexData.nbytes, vertexData, GL.GL_STATIC_DRAW)


            positionAttrib = GL.glGetAttribLocation(self.shaderProgram_rgb, 'position')
            normalAttrib = GL.glGetAttribLocation(self.shaderProgram_rgb, 'normal')
    
            GL.glEnableVertexAttribArray(0)
            GL.glEnableVertexAttribArray(1)
            GL.glEnableVertexAttribArray(2)

            GL.glVertexAttribPointer(positionAttrib, 3, GL.GL_FLOAT, GL.GL_FALSE, 24, None)
            GL.glVertexAttribPointer(normalAttrib, 3, GL.GL_FLOAT, GL.GL_FALSE, 24, ctypes.c_void_p(12))

            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
            GL.glBindVertexArray(0)
        elif self.render_type == "texture":
            vertices = np.concatenate([vertices, normals, texture_coords], axis=-1)
            vertexData = vertices.astype(np.float32)
            texture = loadTexture(texture_path)
            self.textures.append(texture)

            VAO = GL.glGenVertexArrays(1)
            GL.glBindVertexArray(VAO)

            # Need VBO for triangle vertices and texture UV coordinates
            VBO = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, VBO)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, vertexData.nbytes, vertexData, GL.GL_STATIC_DRAW)

            positionAttrib = GL.glGetAttribLocation(self.shaderProgram_texture, 'position')
            normalAttrib = GL.glGetAttribLocation(self.shaderProgram_texture, 'normal')
            coordsAttrib = GL.glGetAttribLocation(self.shaderProgram_texture, 'texCoords')

            GL.glEnableVertexAttribArray(0)
            GL.glEnableVertexAttribArray(1)
            GL.glEnableVertexAttribArray(2)

            GL.glVertexAttribPointer(positionAttrib, 3, GL.GL_FLOAT, GL.GL_FALSE, 32, None)
            GL.glVertexAttribPointer(normalAttrib, 3, GL.GL_FLOAT, GL.GL_FALSE, 32, ctypes.c_void_p(12))
            GL.glVertexAttribPointer(coordsAttrib, 2, GL.GL_FLOAT, GL.GL_TRUE, 32, ctypes.c_void_p(24))

            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
            GL.glBindVertexArray(0)

        self.VAOs.append(VAO)
        self.VBOs.append(VBO)
        self.faces.append(faces)
        self.faces.append(faces)

    def render(self, cls_indexes, poses, colors=None):
        frame = 0
        GL.glClearColor(1.0, 1.0, 1.0, 1)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glEnable(GL.GL_DEPTH_TEST)

        for (i, index) in enumerate(cls_indexes):
            if self.render_type == "depth":
                GL.glUseProgram(self.shaderProgram_depth)
                GL.glUniformMatrix4fv(GL.glGetUniformLocation(self.shaderProgram_depth, 'V'), 1, GL.GL_TRUE,
                                      self.V)
                GL.glUniformMatrix4fv(GL.glGetUniformLocation(self.shaderProgram_depth, 'P'), 1, GL.GL_FALSE,
                                      self.P)
                GL.glUniformMatrix4fv(GL.glGetUniformLocation(self.shaderProgram_depth, 'pose_trans'), 1,
                                      GL.GL_FALSE, np.ascontiguousarray(xyz2mat(poses[i][:3])))
                GL.glUniformMatrix4fv(GL.glGetUniformLocation(self.shaderProgram_depth, 'pose_rot'), 1,
                                      GL.GL_TRUE, np.ascontiguousarray(quat2rotmat(poses[i][3:])))
            elif self.render_type == "rgb":            
                GL.glUseProgram(self.shaderProgram_rgb)
                GL.glUniformMatrix4fv(GL.glGetUniformLocation(self.shaderProgram_rgb, 'V'), 1, GL.GL_TRUE,
                                      self.V)
                GL.glUniformMatrix4fv(GL.glGetUniformLocation(self.shaderProgram_rgb, 'P'), 1, GL.GL_FALSE,
                                      self.P)
                GL.glUniformMatrix4fv(GL.glGetUniformLocation(self.shaderProgram_rgb, 'pose_trans'), 1,
                                      GL.GL_FALSE, np.ascontiguousarray(xyz2mat(poses[i][:3])))
                GL.glUniformMatrix4fv(GL.glGetUniformLocation(self.shaderProgram_rgb, 'pose_rot'), 1,
                                      GL.GL_TRUE, np.ascontiguousarray(quat2rotmat(poses[i][3:])))
                GL.glUniform3f(GL.glGetUniformLocation(self.shaderProgram_rgb, 'light_position'),
                               *self.lightpos)
                GL.glUniform3f(GL.glGetUniformLocation(self.shaderProgram_rgb, 'instance_color'),
                               *self.colors[index])
                GL.glUniform3f(GL.glGetUniformLocation(self.shaderProgram_rgb, 'light_color'), *self.lightcolor)
                GL.glUniform4f(GL.glGetUniformLocation(self.shaderProgram_rgb, 'color'), *colors[i])
            elif self.render_type == "texture":            
                GL.glUseProgram(self.shaderProgram_texture)
                GL.glUniformMatrix4fv(GL.glGetUniformLocation(self.shaderProgram_texture, 'V'), 1, GL.GL_TRUE,
                                      self.V)
                GL.glUniformMatrix4fv(GL.glGetUniformLocation(self.shaderProgram_texture, 'P'), 1, GL.GL_FALSE,
                                      self.P)
                GL.glUniformMatrix4fv(GL.glGetUniformLocation(self.shaderProgram_texture, 'pose_trans'), 1,
                                      GL.GL_FALSE, np.ascontiguousarray(xyz2mat(poses[i][:3])))
                GL.glUniformMatrix4fv(GL.glGetUniformLocation(self.shaderProgram_texture, 'pose_rot'), 1,
                                      GL.GL_TRUE, np.ascontiguousarray(quat2rotmat(poses[i][3:])))
                GL.glUniform3f(GL.glGetUniformLocation(self.shaderProgram_texture, 'light_position'),
                               *self.lightpos)
                GL.glUniform3f(GL.glGetUniformLocation(self.shaderProgram_texture, 'instance_color'),
                               *self.colors[index])
                GL.glUniform3f(GL.glGetUniformLocation(self.shaderProgram_texture, 'light_color'), *self.lightcolor)
            else:
                print("This is a problem!\n")

            try:
                if self.render_type == "texture": 
                    # Activate texture
                    GL.glActiveTexture(GL.GL_TEXTURE0)
                    GL.glBindTexture(GL.GL_TEXTURE_2D, self.textures[index])
                    GL.glUniform1i(self.texUnitUniform, 0)

                # Activate array
                GL.glBindVertexArray(self.VAOs[index])
                # draw triangles
                GL.glDrawElements(GL.GL_TRIANGLES, self.faces[index].size, GL.GL_UNSIGNED_INT,
                                  self.faces[index])
            finally:
                GL.glBindVertexArray(0)
                GL.glUseProgram(0)

        GL.glDisable(GL.GL_DEPTH_TEST)

        GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)        
        depth = GL.glReadPixels(0, 0, self.width, self.height, GL.GL_DEPTH_COMPONENT, GL.GL_FLOAT)
        depth = depth.reshape(self.height, self.width)[::-1, :]
        depth = (self.far * self.near) / (self.far - (self.far - self.near) * depth)

        if self.render_type == "depth":
            return depth

        GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)        
        rgb = GL.glReadPixels(0, 0, self.width, self.height, GL.GL_BGRA, GL.GL_FLOAT)
        rgb = np.frombuffer(rgb, dtype = np.float32).reshape(self.width, self.height, 4)
        rgb = rgb.reshape(self.height, self.width, 4)[::-1, :]
        
        return rgb, depth

    def set_camera(self, camera, target, up):
        self.camera = camera
        self.target = target
        self.up = up
        V = lookat(
            self.camera,
            self.target, up=self.up)

        self.V = np.ascontiguousarray(np.eye(4, dtype=np.float32), np.float32)

    def set_light_color(self, color):
        self.lightcolor = color
    def set_light_pos(self, light):
        self.lightpos = light

    def release(self):
        print(self.glstring)
        self.clean()
        self.r.release()

    def clean(self):
        GL.glDeleteTextures([self.color_tex, self.color_tex_2,
                             self.color_tex_3, self.color_tex_4, self.depth_tex])
        self.color_tex = None
        self.color_tex_2 = None
        self.color_tex_3 = None
        self.color_tex_4 = None

        self.depth_tex = None
        GL.glDeleteFramebuffers(1, [self.fbo])
        self.fbo = None
        GL.glDeleteBuffers(len(self.VAOs), self.VAOs)
        self.VAOs = []
        GL.glDeleteBuffers(len(self.VBOs), self.VBOs)
        self.VBOs = []
        GL.glDeleteTextures(self.textures)
        self.textures = []
        self.objects = []  # GC should free things here
        self.faces = []  # GC should free things here
        self.poses_trans = []  # GC should free things here
        self.poses_rot = []  # GC should free things here
