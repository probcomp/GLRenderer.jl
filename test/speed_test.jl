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
    640, 480,
    1000.0, 1000.0,
    320.0, 240.0,
    0.01, 5.0
)
camera_intrinsics = GL.scale_down_camera(camera_intrinsics, 4)

renderer = GL.setup_renderer(camera_intrinsics, GL.DepthMode())

mesh_data = GL.get_mesh_data_from_obj_file(obj_path) 
GL.load_object!(renderer, mesh_data)

depth_image = GL.gl_render(renderer, [1], [P.Pose([0.0, 0.0, 1.0], R.RotXYZ(0.1, 0.4, 0.9))], P.IDENTITY_POSE)

times = [
    @elapsed GL.gl_render(renderer, [1], [P.Pose([0.0, 0.0, 1.0], R.RotXYZ(0.1, 0.4, 0.9))], P.IDENTITY_POSE)
    for _ in 1:1000
]
avg_time = sum(times)/length(times)
@show avg_time

FileIO.save(GL.view_depth_image(depth_image), "imgs/speed_test_img.png")
