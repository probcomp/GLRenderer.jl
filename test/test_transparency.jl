import GLRenderer as GL
import PoseComposition as PC
import Rotations as R

import Images as I
camera_intrinsics = GL.CameraIntrinsics(
    640, 480,
    1000.0, 1000.0,
    320.0, 240.0,
    0.01, 100.0
)

renderer = GL.setup_renderer(camera_intrinsics, GL.RGBMode();gl_version=(4,1))

box1 = GL.box_mesh_from_dims([1.0, 1.0, 1.0])
box2 = GL.box_mesh_from_dims([1.0, 1.5, 1.0])

GL.load_object!(renderer, box1)
GL.load_object!(renderer, box2)

poses = [PC.Pose([1.0, 0.0, 5.0], PC.IDENTITY_ORN), PC.Pose([-1.0, 0.0, 8.0], PC.IDENTITY_ORN)]

colors = [I.RGBA(0.9, 0.05, 0.05, 0.3),I.RGBA(0.1, 0.2, 0.9, 0.3)]
rgb_image, depth_image = GL.gl_render(renderer, [1,2], poses, PC.IDENTITY_POSE, colors=colors);
I.save("/tmp/img.png", GL.view_rgb_image(rgb_image))
