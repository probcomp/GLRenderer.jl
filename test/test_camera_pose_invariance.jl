# -*- coding: utf-8 -*-
import Revise
import GLRenderer
import PoseComposition
import Rotations
import FileIO

R = Rotations
P = PoseComposition
GL = GLRenderer

obj_path = joinpath(@__DIR__, "035_power_drill/textured_simple.obj")
texture_path = joinpath(@__DIR__, "035_power_drill/texture_map.png")

camera_intrinsics = GL.CameraIntrinsics(
    640
    
    , 480,
    1000.0, 1000.0,
    320.0, 240.0,
    0.01, 5.0
)
camera_intrinsics = GL.scale_down_camera(camera_intrinsics, 4)

renderer = GL.setup_renderer(camera_intrinsics, GL.DepthMode())

mesh_data = GL.get_mesh_data_from_obj_file(obj_path) 
GL.load_object!(renderer, mesh_data)

depth_image_base = GL.gl_render(renderer, [1], [P.Pose([0.0, 0.0, 1.0], R.RotXYZ(0.1, 0.4, 0.9))], P.IDENTITY_POSE)
GL.view_depth_image(depth_image_base)

for _ in 1:1000
    rand_pose = P.Pose(rand(3) * 10.0, R.RotXYZ(rand(3)*2*pi...))
    rand_pose
    depth_image = GL.gl_render(renderer, [1], [rand_pose * P.Pose([0.0, 0.0, 1.0], R.RotXYZ(0.1, 0.4, 0.9))], rand_pose)
    total_mismatches = sum(abs.(depth_image .- depth_image_base) .> 0.01)
    @assert total_mismatches < 3
end
