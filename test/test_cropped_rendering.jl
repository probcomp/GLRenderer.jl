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

renderer = GL.setup_renderer(camera_intrinsics, GL.TextureMode())

mesh_data = GL.get_mesh_data_from_obj_file(obj_path; tex_path=texture_path) 
GL.load_object!(renderer, mesh_data)

@time rgb_image, depth_image = GL.gl_render(
    renderer, [1], [P.Pose([0.0, 0.0, 1.0], R.RotXYZ(0.1, 0.4, 0.9))], 
    P.IDENTITY_POSE;
#     x_min=100, x_max=camera_intrinsics.width, y_min=100, y_max=camera_intrinsics.height
)
GL.view_depth_image(depth_image)
GL.view_rgb_image(rgb_image)

# +
camera_intrinsics = GL.CameraIntrinsics(
    200, 200,
    1000.0, 1000.0,
    100.0, 100.0,
    0.01, 5.0
)
renderer = GL.setup_renderer(camera_intrinsics, GL.TextureMode())

mesh_data = GL.get_mesh_data_from_obj_file(obj_path; tex_path=texture_path) 
GL.load_object!(renderer, mesh_data)

@time rgb_image, depth_image = GL.gl_render(
    renderer, [1], [P.Pose([0.0, 0.0, 1.0], R.RotXYZ(0.1, 0.4, 0.9))], 
    P.IDENTITY_POSE;
#     x_min=100, x_max=camera_intrinsics.width, y_min=100, y_max=camera_intrinsics.height
)
GL.view_depth_image(depth_image)
GL.view_rgb_image(rgb_image)

# +
camera_intrinsics = GL.CameraIntrinsics(
    640, 480,
    1000.0, 1000.0,
    320.0, 240.0,
    0.01, 5.0
)

renderer = GL.setup_renderer(camera_intrinsics, GL.TextureMode())

mesh_data = GL.get_mesh_data_from_obj_file(obj_path; tex_path=texture_path) 
GL.load_object!(renderer, mesh_data)

@time rgb_image, depth_image = GL.gl_render(
    renderer, [1], [P.Pose([0.0, 0.0, 1.0], R.RotXYZ(0.1, 0.4, 0.9))], 
    P.IDENTITY_POSE;
#     x_min=100, x_max=camera_intrinsics.width, y_min=100, y_max=camera_intrinsics.height
)
GL.view_depth_image(depth_image)
GL.view_rgb_image(rgb_image)

# +
camera_intrinsics = GL.CameraIntrinsics(
    640, 480,
    1000.0, 1000.0,
    320.0, 240.0,
    0.01, 5.0
)

GL.set_intrinsics!(renderer,camera_intrinsics)
@time rgb_image, depth_image = GL.gl_render(
    renderer, [1], [P.Pose([0.0, 0.0, 1.0], R.RotXYZ(0.1, 0.4, 0.9))], 
    P.IDENTITY_POSE;
#     x_min=100, x_max=camera_intrinsics.width, y_min=100, y_max=camera_intrinsics.height
)
GL.view_depth_image(depth_image)
GL.view_rgb_image(rgb_image)
# -

GL.view_rgb_image(rgb_image)[200:400,200:400]

# +
camera_intrinsics = GL.CameraIntrinsics(
    100, 100,
    1000.0, 1000.0,
    320.0 - 200.0, 240.0 - 200.0,
    0.01, 5.0
)

@time GL.set_intrinsics!(renderer,camera_intrinsics)
@time rgb_image2, depth_image2 = GL.gl_render(
    renderer, [1], [P.Pose([0.0, 0.0, 1.0], R.RotXYZ(0.1, 0.4, 0.9))], 
    P.IDENTITY_POSE;
#     x_min=100, x_max=camera_intrinsics.width, y_min=100, y_max=camera_intrinsics.height
)
GL.view_rgb_image(rgb_image2)
# -


