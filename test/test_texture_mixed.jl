# +
import Revise
import GLRenderer
import PoseComposition
import Rotations
import Images
I = Images

R = Rotations
P = PoseComposition
GL = GLRenderer
# -

obj_path = joinpath(@__DIR__, "035_power_drill/textured_simple.obj")
texture_path = joinpath(@__DIR__, "035_power_drill/texture_map.png")

# +
camera_intrinsics = GL.CameraIntrinsics(
    640, 480,
    1000.0, 1000.0,
    320.0, 240.0,
    0.01, 5.0
)

renderer = GL.setup_renderer(camera_intrinsics, GL.TextureMixedMode())


mesh_data = GL.get_mesh_data_from_obj_file(obj_path; tex_path=texture_path) 
GL.load_object!(renderer, mesh_data)

rgb_image, depth_image = GL.gl_render(
    renderer, [1], [P.Pose([0.0, 0.0, 1.0], R.RotXYZ(0.1, 0.4, 0.9))], 
    P.IDENTITY_POSE
)
img = I.colorview(I.RGBA, permutedims(rgb_image,(3,1,2)))


# +
camera_intrinsics = GL.CameraIntrinsics(
    640, 480,
    1000.0, 1000.0,
    320.0, 240.0,
    0.01, 30.0
)

renderer = GL.setup_renderer(camera_intrinsics, GL.TextureMixedMode())


mesh_data = GL.get_mesh_data_from_obj_file(obj_path; tex_path=texture_path) 
GL.load_object!(renderer, mesh_data)

box_mesh = GL.box_mesh_from_dims([10.0, 0.5, 5.0])
GL.load_object!(renderer, box_mesh)


rgb_image, depth_image = GL.gl_render(
    renderer, [1,2], [P.Pose([0.0, 0.0, 1.0], R.RotXYZ(0.1, 0.4, 0.9)), P.Pose([0.0, 2.0, 10.0],P.IDENTITY_ORN)], 
    P.IDENTITY_POSE; colors = [I.colorant"red" for _ in 1:2]
)

img = I.colorview(I.RGBA, permutedims(rgb_image,(3,1,2)))
I.save(joinpath(@__DIR__, "imgs/texture_mixed_rendering_rgb_image.png"), img)

img = I.colorview(I.Gray, depth_image ./ maximum(depth_image))
I.save(joinpath(@__DIR__, "imgs/texture_mixed_rendering_depth_image.png"), img)

