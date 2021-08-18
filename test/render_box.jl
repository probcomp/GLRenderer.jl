# +
import Revise
import GLRenderer
import PoseComposition
import Rotations
import Geometry
import Plots
import Images
I = Images
PL = Plots

R = Rotations
P = PoseComposition
GL = GLRenderer

camera_intrinsics = Geometry.CameraIntrinsics(
    640, 480,
    1000.0, 1000.0,
    320.0, 240.0,
    0.01, 20.0
)
# -

import Plots
PL = Plots

# +
v = [
-1.0 -1.0 1.0;
1.0 -1.0 1.0;
-1.0 1.0 1.0;
1.0 1.0 1.0;

1.0 -1.0 -1.0;
-1.0 -1.0 -1.0;
1.0 1.0 -1.0;
-1.0 1.0 -1.0;


1.0 -1.0 1.0;
1.0 -1.0 -1.0;
1.0 1.0 1.0;
1.0 1.0 -1.0;
        


-1.0 -1.0 -1.0;
-1.0 -1.0 1.0;
-1.0 1.0 -1.0;
-1.0 1.0 1.0;
        
-1.0 -1.0 -1.0;
1.0 -1.0 -1.0;
-1.0 -1.0 1.0;
1.0 -1.0 1.0;

-1.0 1.0 1.0;
1.0 1.0 1.0;
-1.0 1.0 -1.0;
1.0 1.0 -1.0;
]

f = vcat([
    [1 3 2;
    2 3 4;] .+ 4*(i-1) for i in 1:6
]...) .- 1

n = vcat(
[
    let
        vertices = v[4*(i-1)+1:4*(i-1)+4,:]
        s = sum(vertices, dims=1)
        s = s ./ sqrt(sum(s.^2))
        vcat([s for _ in 1:4]...)
    end
    for i in 1:6
]...)
f

# +
function voxelize(cloud, resolution)
    cloud = round.(cloud ./ resolution) * resolution
    idxs = unique(i -> cloud[:,i], 1:size(cloud)[2])
    cloud[:, idxs]
end

cloud = rand(3,100) * 1.0
resolution = 0.1
cloud = voxelize(cloud, resolution)

new_v = vcat([v .* resolution/2.0 .+ r' for r in eachcol(cloud)]...)
new_n = vcat([n for r in eachcol(cloud)]...)
new_f = vcat([f .+ 24*(i-1) for (i,r) in enumerate(eachcol(cloud))]...)
# -

renderer = GL.setup_renderer(camera_intrinsics, GL.RGBMode())
GL.load_object!(renderer, new_v, new_n, new_f)
renderer.gl_instance.lightpos = [0,0,0]
rgb_image, depth_image = GL.gl_render_rgb_depth_image(renderer, 
    [1], [P.Pose([0.0, 0.0, 4.0], R.RotXYZ(0.0, 0.6, 0.3))], 
    [I.colorant"red"],
    P.IDENTITY_POSE)
I.colorview(I.RGBA, permutedims(rgb_image,(3,1,2)))

camera_intrinsics = Geometry.CameraIntrinsics(
    10.0, 1,
    10.0, 1000.0,
    5.0, 0.5,
    0.01, 50.0
)
renderer = GL.setup_renderer(camera_intrinsics, GL.DepthMode())
GL.load_object!(renderer, v, f)
renderer.gl_instance.lightpos = [0,0,0]
@time depth_image = GL.gl_render(renderer, 
    [1], [P.Pose([0.0, 0.0, 10.0], R.RotXYZ(0.0, 0.0, 0.0))], 
    P.IDENTITY_POSE)
PL.heatmap(depth_image)

depth_image

camera_intrinsics.far


