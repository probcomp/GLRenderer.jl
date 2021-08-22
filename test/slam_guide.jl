# -*- coding: utf-8 -*-
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
for z in room_height_bounds[1]:resolution/2.0:room_height_bounds[2]
    push!(room_cloud, [room_width_bounds[1], 0.0, z])
    push!(room_cloud, [room_width_bounds[2], 0.0, z])
end
for x in room_width_bounds[1]:resolution/2.0:room_width_bounds[2]
    push!(room_cloud, [x, 0.0, room_height_bounds[1]])
    push!(room_cloud, [x, 0.0, room_height_bounds[2]])
end
room_cloud = hcat(room_cloud...)
v,n,f = GL.mesh_from_voxelized_cloud(GL.voxelize(room_cloud, resolution), resolution)

renderer = GL.setup_renderer(Geometry.CameraIntrinsics(
    600, 600,
    300.0, 300.0,
    300.0,300.0,
    0.1, 50.0
), GL.DepthMode())
GL.load_object!(renderer, v, f)
renderer.gl_instance.lightpos = [0,0,0]
depth = GL.gl_render(renderer, 
    [1], [P.IDENTITY_POSE], 
    P.Pose([0.0, -10.0, 0.0], R.RotX(-pi/2)))
PL.heatmap(depth, aspect_ratio=:equal)

camera_intrinsics = Geometry.CameraIntrinsics(
    14, 1,
    10.0, 1.0,
    7.0, 0.5,
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

room_bounds_uniform_params = [room_width_bounds[1] room_width_bounds[2];room_height_bounds[1] room_height_bounds[2]]
@show Gen.random(mvuniform,room_bounds_uniform_params)

# +
@Gen.gen function slam_single_timestep(room_bounds, I)
    pos ~ mvuniform(room_bounds)
    hd ~ Gen.uniform(0.0, 2*pi)
    
    cam_pose = P.Pose([pos[1], 0.0, pos[2]], R.RotY(hd))
    depth = GL.gl_render(renderer, 
        [1], [P.IDENTITY_POSE], cam_pose)    
    
    sense ~ Gen.mvnormal(depth[1,:], I)
    (pose=cam_pose, depth=depth)
end

function viz_env()
    PL.scatter!(room_cloud[1,:], room_cloud[3,:], label=false)
end

function viz_pose(pose)
    pos = pose.pos
    PL.scatter!([pos[1]], [pos[3]],label=false)

    direction = pose.orientation * [0.0, 0.0, 1.0]
    PL.plot!([pos[1], pos[1]+direction[1]],[pos[3], pos[3]+direction[3]],arrow=true,color=:black,linewidth=2, label=false)
end

function viz_obs(tr)
    hd = tr[:hd]
    pos = tr[:pos]
    pose = Gen.get_retval(tr).pose
    cloud = GL.flatten_point_cloud(GL.depth_image_to_point_cloud(Gen.get_retval(tr).depth, camera_intrinsics))
    cloud = GL.move_points_to_frame_b(cloud, pose)
    PL.scatter!(cloud[1,:], cloud[3,:], aspect_ratio=:equal, label=false)
end
    
function viz_trace(tr)
    PL.plot()
    viz_env()
    viz_pose(Gen.get_retval(tr).pose)
    viz_obs(tr)
end

function viz_corner(corner_pose)
    PL.scatter!([corner_pose.pos[1]], [corner_pose.pos[3]], label=false)
    pos = corner_pose.pos
    direction = corner_pose.orientation * [1.0, 0.0, 0.0]
    PL.plot!([pos[1], pos[1]+direction[1]],[pos[3], pos[3]+direction[3]],arrow=true,color=:red,linewidth=2, label=false)
end

# +
# constraints = Gen.choicemap(:pos=>[0.0, 0.0], :hd=>pi/4)
import LinearAlgebra
I = Matrix{Float64}(LinearAlgebra.I, camera_intrinsics.width, camera_intrinsics.width) * 0.0001;
tr_gt, w = Gen.generate(slam_single_timestep, (room_bounds_uniform_params,I,));
observations = Gen.get_selected(Gen.get_choices(tr_gt), Gen.select(:sense))
observations

viz_trace(tr_gt)
# -

@time traces, weights, lml_est = Gen.importance_sampling(slam_single_timestep,
    Gen.get_args(tr_gt), observations, 100);
best_idx = argmax(weights)
best_tr = traces[best_idx]
viz_trace(best_tr)

# # Drift Moves

# +
@Gen.gen function position_drift_proposal(trace, cov)
   pos ~ Gen.mvnormal(trace[:pos], cov) 
end

@Gen.gen function head_direction_drift_proposal(trace, var)
   hd ~ Gen.normal(trace[:hd], var) 
end

@Gen.gen function joint_pose_drift_proposal(trace, cov, var)
   pos ~ Gen.mvnormal(trace[:pos], cov) 
   hd ~ Gen.normal(trace[:hd], var) 
end

function drift_inference_program(tr)
    cov = [1.0 0.0;0.0 1.0] * 0.5
    cov_x = [1.0 0.0;0.0 0.0001] * 0.5
    cov_y = [0.0001 0.0;0.0 1.0] * 0.5

    for _ in 1:500
       tr, _ = Gen.mh(tr, position_drift_proposal, (cov,)) 
       tr, _ = Gen.mh(tr, position_drift_proposal, (cov_x,)) 
       tr, _ = Gen.mh(tr, position_drift_proposal, (cov_y,)) 
       tr, _ = Gen.mh(tr, head_direction_drift_proposal, (deg2rad(10.0),)) 
       tr, _ = Gen.mh(tr, joint_pose_drift_proposal, (cov, deg2rad(10.0),)) 
    end
    tr
end

tr = drift_inference_program(best_tr)
viz_trace(tr)
# -

# # Corner Detection

# +
import GeometryBasics
GB = GeometryBasics

PL.plot()
viz_env()
corner_1 = P.Pose([room_width_bounds[1], 0.0, room_height_bounds[1]], R.RotY(-pi/4))
viz_corner(corner_1)
corner_2 = P.Pose([room_width_bounds[1], 0.0, room_height_bounds[2]], R.RotY(pi/4))
viz_corner(corner_2)
corner_3 = P.Pose([room_width_bounds[2], 0.0, room_height_bounds[1]], R.RotY(pi+pi/4))
viz_corner(corner_3)
corner_4 = P.Pose([room_width_bounds[2], 0.0, room_height_bounds[2]], R.RotY(pi-pi/4))
viz_corner(corner_4)

# +
# Corner detection
tr = tr_gt

cloud = GL.flatten_point_cloud(GL.depth_image_to_point_cloud(Gen.get_retval(tr).depth, camera_intrinsics))
cloud = vcat(cloud[1,:]', cloud[3,:]')
PL.scatter(cloud[1,:], cloud[2,:], label=false, aspect_ratio=:equal)
deltas = diff(cloud,dims=2)
PL.scatter(dirs[1,:], dirs[2,:], label=false)
dirs = map(x->R.RotMatrix{2,Float32}(atan(x[2],x[1])), eachcol(deltas))
angle_errors = [abs.(R.rotation_angle(inv(dirs[i])*dirs[i+1])) for i in 1:length(dirs)-1]
@show angle_errors
spikes = findall(angle_errors .> deg2rad(45.0))
@show spikes
spikes .+= 1

s = spikes[1]
i,j = s-2,s-1
k,l = s+2,s+1
a,b,c,d = cloud[:,i],cloud[:,j],cloud[:,k],cloud[:,l]
d1 =  a.-b
d2 =  c.-d
d1 = d1 / LinearAlgebra.norm(d1)
d2 = d2 / LinearAlgebra.norm(d2)
dir = (d1 .+ d2) ./ 2

function intersection(a,b,c,d)
    a1 = b[2] - a[2]
    b1 = a[1] - b[1]
    c1 = a1 * a[1] + b1 * a[2]
 
    a2 = d[2] - c[2]
    b2 = c[1] - d[1]
    c2 = a2 * c[1] + b2 * c[2]
 
    Δ = a1 * b2 - a2 * b1
    # If lines are parallel, intersection point will contain infinite values
    return (b2 * c1 - b1 * c2) / Δ, (a1 * c2 - a2 * c1) / Δ
end
corner = intersection(a,b,c,d)
corner_pose = P.Pose([corner[1], 0.0, corner[2]], R.RotY(-atan(dir[2],dir[1]))) 
# -
PL.scatter(cloud[1,:], cloud[2,:], aspect_ratio=:equal)
viz_corner(corner_pose)

Gen.get_retval(tr_gt).pose

corner_3 * inv(corner_pose)

inv

PL.plot()
viz_env()
corner_1 = P.Pose([room_width_bounds[1], 0.0, room_height_bounds[1]], R.RotY(-pi/4))
viz_corner(corner_1)
corner_2 = P.Pose([room_width_bounds[1], 0.0, room_height_bounds[2]], R.RotY(pi/4))
viz_corner(corner_2)
corner_3 = P.Pose([room_width_bounds[2], 0.0, room_height_bounds[1]], R.RotY(pi+pi/4))
viz_corner(corner_3)
corner_4 = P.Pose([room_width_bounds[2], 0.0, room_height_bounds[2]], R.RotY(pi-pi/4))
viz_corner(corner_4)

rad2deg(atan(dir[2],dir[1]))

@show q = [dir[1], 0.0, dir[2]]
q = q / LinearAlgebra.norm(q)
@show q
r = R.RotMatrix{2,Float32}(atan(dir[2],dir[1]))
@show r
r = R.RotY(-atan(dir[2],dir[1]))



viz_trace(tr_gt)
PL.plot!(rand(100),rand(100))

PL.plot(angle_errors)



