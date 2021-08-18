import Revise
import GLRenderer
import PoseComposition
import Rotations
import Geometry
import Images
I = Images

R = Rotations
P = PoseComposition
GL = GLRenderer

camera_intrinsics = Geometry.CameraIntrinsics(
    640, 480,
    1000.0, 1000.0,
    320.0, 240.0,
    0.01, 5.0
)

renderer = GL.setup_renderer(camera_intrinsics, GL.RGBMode())

v,n,f,_ = renderer.gl_instance.load_obj_parameters(
    obj_path
) 
GL.load_object!(renderer, v, n, f)

rgb_image, depth_image = GL.gl_render(
    renderer, [1], [P.Pose([0.0, 0.0, 1.0], R.RotXYZ(0.1, 0.4, 0.9))], 
    [I.colorant"red"],
    P.IDENTITY_POSE
)



img = I.colorview(I.RGBA, permutedims(rgb_image,(3,1,2)))
I.save(joinpath(@__DIR__, "imgs/rgb_rendering_rgb_image.png"), img)

img = I.colorview(I.Gray, depth_image ./ maximum(depth_image))
I.save(joinpath(@__DIR__, "imgs/rgb_rendering_depth_image.png"), img)
