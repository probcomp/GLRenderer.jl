# -*- coding: utf-8 -*-
# +
import Revise
import GLRenderer
import PoseComposition
import Rotations
import Geometry
import Plots
import Images
import GenParticleFilters
PF = GenParticleFilters

I = Images
PL = Plots

R = Rotations
P = PoseComposition
GL = GLRenderer
# -

Revise.errors()
Revise.revise()

# +
cloud = rand(3,100) * 1.0
resolution = 0.1
v,n,f = GL.mesh_from_voxelized_cloud(GL.voxelize(cloud, resolution), resolution)

renderer = GL.setup_renderer(Geometry.CameraIntrinsics(
    640, 480,
    1000.0, 1000.0,
    320.0, 240.0,
    0.01, 20.0
), GL.RGBMode())
GL.load_object!(renderer, v,n,f)

renderer.gl_instance.lightpos = [0,0,0]
rgb_image, depth_image = GL.gl_render(renderer, 
    [1], [P.Pose([0.0, 0.0, 4.0], R.RotXYZ(0.0, 0.0, 0.0))], 
    [I.colorant"green"],
    P.IDENTITY_POSE)
I.colorview(I.RGBA, permutedims(rgb_image,(3,1,2)))

# +
cloud = rand(3,100) * 1.0
resolution = 0.05
v,n,f = GL.mesh_from_voxelized_cloud(GL.voxelize(cloud, resolution), resolution)

renderer = GL.setup_renderer(Geometry.CameraIntrinsics(
    640, 480,
    1000.0, 1000.0,
    320.0, 240.0,
    0.01, 20.0
), GL.RGBMode())
GL.load_object!(renderer, v,n,f)

renderer.gl_instance.lightpos = [0,0,0]
rgb_image, depth_image = GL.gl_render(renderer, 
    [1], [P.Pose([0.0, 0.0, 4.0], R.RotXYZ(0.0, 0.0, 0.0))], 
    [I.colorant"green"],
    P.IDENTITY_POSE)
I.colorview(I.RGBA, permutedims(rgb_image,(3,1,2)))

# +
obj_path = "035_power_drill/textured_simple.obj"
obj_vertices,_,_ = renderer.gl_instance.load_obj_parameters(obj_path)
obj_vertices = permutedims(obj_vertices)

resolution = 0.004
v,n,f = GL.mesh_from_voxelized_cloud(GL.voxelize(obj_vertices, resolution), resolution)
renderer = GL.setup_renderer(Geometry.CameraIntrinsics(
    640, 480,
    1000.0, 1000.0,
    320.0, 240.0,
    0.01, 20.0
), GL.RGBMode())
GL.load_object!(renderer, v,n,f)

renderer.gl_instance.lightpos = [0,0,0]
rgb_image, depth_image = GL.gl_render(renderer, 
    [1], [P.Pose([0.0, 0.0, 1.0], R.RotXYZ(pi, 0.3, 0.0))], 
    [I.colorant"green"],
    P.IDENTITY_POSE)
I.colorview(I.RGBA, permutedims(rgb_image,(3,1,2)))
