import Revise
import GLRenderer
import PoseComposition
import Rotations
import Images
I = Images

R = Rotations
P = PoseComposition
GL = GLRenderer

obj_path = joinpath(GL.get_glrenderer_module_path(), "test", "035_power_drill/textured_simple.obj")
texture_path = joinpath(GL.get_glrenderer_module_path(), "test", "035_power_drill/texture_map.png")

camera_intrinsics = GL.CameraIntrinsics(
    640, 480,
    1000.0, 1000.0,
    320.0, 240.0,
    0.01, 5.0
)


renderer = GL.setup_renderer(camera_intrinsics, GL.PointCloudMode())

mesh_data = GL.get_mesh_data_from_obj_file(obj_path) 
GL.load_object!(renderer, mesh_data)

point_cloud = GL.gl_render(renderer, [1], [P.Pose([0.0, 0.0, 1.0], R.RotXYZ(0.1, 0.4, 0.9))], P.IDENTITY_POSE)
x = GL.flatten_point_cloud(permutedims(point_cloud,(3,2,1))[:,:,1:3])
MV.viz(x;channel_name=:a, color=I.colorant"red")
MV.viz(mesh_data.vertices;channel_name=:b, color=I.colorant"blue")


img = GL.image_from_clouds_and_colors([x], [I.colorant"red"], camera_intrinsics)
I.save(joinpath(@__DIR__, "imgs/point_cloud_rendering.png"), img)

img


