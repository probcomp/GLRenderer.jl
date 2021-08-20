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

Revise.errors()
Revise.revise()

cloud = rand(3,100) * 1.0
resolution = 0.1
v,n,f = GL.mesh_from_voxelized_cloud(GL.voxelize(cloud, resolution), resolution)
renderer = GL.setup_renderer(camera_intrinsics, GL.RGBMode())
GL.load_object!(renderer, v,n,f)
renderer.gl_instance.lightpos = [0,0,0]
rgb_image, depth_image = GL.gl_render(renderer, 
    [1], [P.Pose([0.0, 0.0, 4.0], R.RotXYZ(0.0, 0.0, 0.0))], 
    [I.colorant"red"],
    P.IDENTITY_POSE)
I.colorview(I.RGBA, permutedims(rgb_image,(3,1,2)))

room_height_bounds = (-5.0, 5.0)
room_width_bounds = (-8.0, 8.0)
resolution = 0.1
room_cloud = []
for x in room_height_bounds[1]:resolution/2.0:room_height_bounds[2]
    push!(room_cloud, [x, 0.0, room_width_bounds[1]])
    push!(room_cloud, [x, 0.0, room_width_bounds[2]])
end
for z in room_width_bounds[1]:resolution/2.0:room_width_bounds[2]
    push!(room_cloud, [room_height_bounds[1], 0.0, z])
    push!(room_cloud, [room_height_bounds[2], 0.0, z])
end
room_cloud = hcat(room_cloud...)
v,n,f = GL.mesh_from_voxelized_cloud(GL.voxelize(room_cloud, resolution), resolution)

camera_intrinsics = Geometry.CameraIntrinsics(
    600, 600,
    300.0, 300.0,
    300.0,300.0,
    0.1, 50.0
)
renderer = GL.setup_renderer(camera_intrinsics, GL.DepthMode())
GL.load_object!(renderer, v, f)
renderer.gl_instance.lightpos = [0,0,0]
depth = GL.gl_render(renderer, 
    [1], [P.IDENTITY_POSE], 
    P.Pose([0.0, -10.0, 0.0], R.RotX(-pi/2)))
PL.heatmap(depth, aspect_ratio=:equal)

camera_intrinsics = Geometry.CameraIntrinsics(
    15, 1,
    10.0, 1.0,
    7.5, 0.5,
    0.1, 20.0
)
renderer = GL.setup_renderer(camera_intrinsics, GL.DepthMode())
GL.load_object!(renderer, v, f)
renderer.gl_instance.lightpos = [0,0,0]
cam_pose = P.Pose(zeros(3), R.RotY(0.0))
@time depth = GL.gl_render(renderer, 
    [1], [P.IDENTITY_POSE], cam_pose)
cloud = GL.flatten_point_cloud(GL.depth_image_to_point_cloud(depth, camera_intrinsics))
PL.heatmap(depth)

print(depth)

PL.scatter(cloud[1,:], cloud[3,:],xlim=(-10,10),ylim=(-10,10))

import Gen

import Distributions

# +
struct MultivariateUniform <: Gen.Distribution{Vector{Float64}} end
const mvuniform = MultivariateUniform()
function Gen.logpdf(::MultivariateUniform, x::AbstractArray{Float64,1}, b::AbstractArray{Float64,2})
    dist = Distributions.Product(Distributions.Uniform.(b[:,1], b[:,2]))
    Distributions.logpdf(dist, x)
end
function Gen.random(::MultivariateUniform, b::AbstractArray{Float64,2})
    dist = Distributions.Product(Distributions.Uniform.(b[:,1], b[:,2]))
    rand(dist)
end

struct MultivariateUniform <: Gen.Distribution{Vector{Float64}} end
const mvuniform = MultivariateUniform()
function Gen.logpdf(::MultivariateUniform, x::AbstractArray{Float64,1}, b::AbstractArray{Float64,2})
    dist = Distributions.Product(Distributions.Uniform.(b[:,1], b[:,2]))
    Distributions.logpdf(dist, x)
end
function Gen.random(::MultivariateUniform, b::AbstractArray{Float64,2})
    dist = Distributions.Product(Distributions.Uniform.(b[:,1], b[:,2]))
    rand(dist)
end
# -

room_bounds_uniform_params = [room_height_bounds[1] room_height_bounds[2];room_width_bounds[1] room_width_bounds[2]]
@show Gen.random(mvuniform,room_bounds_uniform_params)

I = Matrix{Float64}(LinearAlgebra.I, 15, 15) * 0.0001;
I

@Gen.gen function slam_single_timestep(room_bounds)
    pos ~ mvuniform(room_bounds)
    hd ~ Gen.uniform(0.0, 2*pi)
    
    cam_pose = P.Pose([pos[1], 0.0, pos[2]], R.RotY(hd))
    depth = GL.gl_render(renderer, 
        [1], [P.IDENTITY_POSE], cam_pose)    
    
    sense ~ Gen.mvnormal(depth[1,:], I)
    (pose=cam_pose, depth=depth)
end

# +
tr_gt, w = Gen.generate(slam_single_timestep, (room_bounds_uniform_params,));
@show tr_gt[:pos]
@show tr_gt[:hd]
@show tr_gt[:sense]
@show Gen.get_retval(tr_gt).depth
@show Gen.get_score(tr_gt)

observations = Gen.get_selected(Gen.get_choices(tr_gt), Gen.select(:sense))
observations

# +
function viz_trace(tr)
    PL.scatter(room_cloud[1,:], room_cloud[3,:], label=false)
    hd = tr[:hd]
    pos = tr[:pos]
    pose = Gen.get_retval(tr).pose

    PL.scatter!([pos[1]], [pos[2]])

    direction = pose.orientation * [0.0, 0.0, 1.0]
    PL.plot!([pos[1], pos[1]+direction[1]],[pos[2], pos[2]+direction[3]],arrow=true,color=:black,linewidth=2,label="")

    cloud = GL.flatten_point_cloud(GL.depth_image_to_point_cloud(Gen.get_retval(tr).depth, camera_intrinsics))
    cloud = GL.move_points_to_frame_b(cloud, pose)
    PL.scatter!(cloud[1,:], cloud[3,:], label=false)
end

viz_trace(tr_gt)
# -

@time traces, weights, lml_est = Gen.importance_sampling(slam_single_timestep,
    (room_bounds_uniform_params,), observations, 10000);

best_idx = argmax(weights)
best_tr = traces[best_idx]
@show best_tr[:pos]
@show best_tr[:hd]
viz_trace(best_tr)


