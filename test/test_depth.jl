import Revise
import GLRenderer
import PoseComposition
import Rotations
import Geometry

R = Rotations
P = PoseComposition
GL = GLRenderer

camera_intrinsics = Geometry.CameraIntrinsics(
    640, 480,
    1000.0, 1000.0,
    320.0, 240.0,
    0.01, 5.0
)

renderer = GL.setup_renderer(camera_intrinsics, GL.DepthMode())

v,_,f,_ = renderer.gl_instance.load_obj_parameters(
    obj_path
) 
GL.load_object!(renderer, v, f)

depth_image = GL.gl_render(renderer, [1], [P.Pose([0.0, 0.0, 1.0], R.RotXYZ(0.1, 0.4, 0.9))], P.IDENTITY_POSE)

import Images
I = Images

img = I.colorview(I.Gray, depth_image ./ maximum(depth_image))
I.save(joinpath(@__DIR__, "imgs/depth_rendering_depth_image.png"), img)
