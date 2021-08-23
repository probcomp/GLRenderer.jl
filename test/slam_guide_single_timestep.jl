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
    5.0, 1.0,
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

import Gen
import Distributions

# +
struct PoseUniform <: Gen.Distribution{P.Pose} end
const pose_uniforms = PoseUniform()
function Gen.logpdf(::PoseUniform, pose::P.Pose, bounds_x, bounds_z)
    (
        Gen.logpdf(Gen.uniform, pose.pos[1], bounds_x...) +
        Gen.logpdf(Gen.uniform, pose.pos[3], bounds_z...) +
        Gen.logpdf(Gen.uniform, R.RotY(pose.orientation).theta, 0.0, 2*pi)
    )   
end
function Gen.random(::PoseUniform, bounds_x, bounds_z)
    x = rand(Distributions.Uniform(bounds_x...))
    z = rand(Distributions.Uniform(bounds_z...))
    hd = rand(Distributions.Uniform(0.0, 2*pi))
    P.Pose([x, 0.0, z], R.RotY(hd))
end

struct PoseGaussian <: Gen.Distribution{P.Pose} end
const pose_gaussian = PoseGaussian()
function Gen.logpdf(::PoseGaussian, pose::P.Pose, center_pose::P.Pose, cov, var)
    hd = R.RotY(pose.orientation).theta
    hd_center = R.RotY(center_pose.orientation).theta
    (
        Gen.logpdf(Gen.mvnormal, [pose.pos[1], pose.pos[3]], [center_pose.pos[1], center_pose.pos[3]], cov) +
        Gen.logpdf(Gen.normal, hd, hd_center, var)
    )   
end
function Gen.random(::PoseGaussian, center_pose::P.Pose, cov, var)
    hd_center = R.RotY(center_pose.orientation).theta
    pos = Gen.random(Gen.mvnormal, [center_pose.pos[1], center_pose.pos[3]], cov)
    hd = Gen.random(Gen.normal, hd_center, var)
    P.Pose([pos[1], 0.0, pos[2]], R.RotY(hd))
end
# -

room_bounds_uniform_params = [room_width_bounds[1] room_width_bounds[2];room_height_bounds[1] room_height_bounds[2]]

# +
@Gen.gen function slam_unfold_kernel(t, prev_data, room_bounds, I)
    if isnothing(prev_data)
        pose ~ pose_uniforms(room_bounds[1,:],room_bounds[2,:])
    else
        pose ~ pose_gaussian(prev_data.pose, [1.0 0.0;0.0 1.0] * 0.05, deg2rad(10.0))
    end
    depth = GL.gl_render(renderer, 
        [1], [P.IDENTITY_POSE], pose)    
    
    sense ~ Gen.mvnormal(depth[1,:], I)
    return (pose=pose, depth=depth, sense=sense)
end

sense_addr(t) = (:slam => t => :sense)
pose_addr(t) = (:slam => t => :pose)
get_pose(tr,t) = tr[pose_addr(t)]
get_depth(tr,t) = Gen.get_retval(tr)[t].depth
get_sense(tr,t) = tr[sense_addr(t)]

slam_unfolded = Gen.Unfold(slam_unfold_kernel)

@Gen.gen (static) function slam_multi_timestep(T, prev_data, room_bounds, I)
    slam ~ slam_unfolded(T, prev_data, room_bounds, I)
    return slam
end

Gen.@load_generated_functions

function viz_env()
    PL.scatter!(room_cloud[1,:], room_cloud[3,:], label=false)
end

function viz_pose(pose)
    pos = pose.pos
    PL.scatter!([pos[1]], [pos[3]],label=false)

    direction = pose.orientation * [0.0, 0.0, 1.0]
    PL.plot!([pos[1], pos[1]+direction[1]],
             [pos[3], pos[3]+direction[3]],
             arrow=true,color=:black,linewidth=2, label=false)
end

function viz_obs(tr,t)
    pose = get_pose(tr,t)
    depth = get_depth(tr,t)
    sense = get_sense(tr,t)
    
    cloud = GL.flatten_point_cloud(GL.depth_image_to_point_cloud(depth, camera_intrinsics))
    cloud = GL.move_points_to_frame_b(cloud, pose)
    PL.scatter!(cloud[1,:], cloud[3,:], aspect_ratio=:equal, label=false)
    
    cloud = GL.flatten_point_cloud(GL.depth_image_to_point_cloud(reshape(sense,(camera_intrinsics.height, camera_intrinsics.width)), camera_intrinsics))
    cloud = GL.move_points_to_frame_b(cloud, pose)
    PL.scatter!(cloud[1,:], cloud[3,:], aspect_ratio=:equal, label=false)
end
    
function viz_trace(tr, ts)
    PL.plot()
    T = Gen.get_args(tr)[1]
    viz_env()
    for t in ts
        viz_pose(get_pose(tr,t))
        viz_obs(tr, t)
    end
    PL.plot!()
end

function viz_corner(corner_pose)
    PL.scatter!([corner_pose.pos[1]], [corner_pose.pos[3]], label=false)
    pos = corner_pose.pos
    direction = corner_pose.orientation * [1.0, 0.0, 0.0]
    PL.plot!([pos[1], pos[1]+direction[1]],[pos[3], pos[3]+direction[3]],arrow=true,color=:red,linewidth=2, label=false)
end
# -

# constraints = Gen.choicemap(:pos=>[0.0, 0.0], :hd=>pi/4)
import LinearAlgebra
I = Matrix{Float64}(LinearAlgebra.I, camera_intrinsics.width, camera_intrinsics.width) * 0.01;
tr_gt, w = Gen.generate(slam_multi_timestep, (5, nothing, room_bounds_uniform_params,I,));
viz_trace(tr_gt,1:5)

# # Corner Detection

PL.plot()
viz_env()
corner_1 = P.Pose([room_width_bounds[1], 0.0, room_height_bounds[1]], R.RotY(-pi/4))
corner_2 = P.Pose([room_width_bounds[1], 0.0, room_height_bounds[2]], R.RotY(pi/4))
corner_3 = P.Pose([room_width_bounds[2], 0.0, room_height_bounds[1]], R.RotY(pi+pi/4))
corner_4 = P.Pose([room_width_bounds[2], 0.0, room_height_bounds[2]], R.RotY(pi-pi/4))
gt_corners = [corner_1,corner_2,corner_3,corner_4]
for c in gt_corners
    viz_corner(c)
end
PL.plot!()

function get_corners(sense)
    cloud = GL.flatten_point_cloud(GL.depth_image_to_point_cloud(reshape(sense,(camera_intrinsics.height, camera_intrinsics.width)), camera_intrinsics))
    cloud = vcat(cloud[1,:]', cloud[3,:]')
    deltas = diff(cloud,dims=2)
    dirs = map(x->R.RotMatrix{2,Float32}(atan(x[2],x[1])), eachcol(deltas))
    angle_errors = [abs.(R.rotation_angle(inv(dirs[i])*dirs[i+1])) for i in 1:length(dirs)-1]
    spikes = findall(angle_errors .> deg2rad(45.0))
    spikes .+= 1
    corners = []
    
    for s in spikes
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
        push!(corners, corner_pose)
    end
    corners
end

# +
mixture_of_pose_gaussians = Gen.HomogeneousMixture(pose_gaussian, [0, 2, 0])

@Gen.gen function pose_mixture_proposal(trace, poses, t, cov, var)
    n = length(poses)
    weights = ones(n) ./ n
    {pose_addr(t)} ~ mixture_of_pose_gaussians(weights, poses, cat([cov for _ in 1:n]..., dims=3), [var for _ in 1:n])
end
# -

# # Drift Moves

# +
@Gen.gen function position_drift_proposal(trace, t,cov)
   {pose_addr(t)} ~ pose_gaussian(trace[pose_addr(t)], cov, 0.0001) 
end

@Gen.gen function head_direction_drift_proposal(trace, t,var)
   {pose_addr(t)} ~ pose_gaussian(trace[pose_addr(t)], [1.0 0.0;0.0 1.0] * 0.00001, var) 
end

@Gen.gen function joint_pose_drift_proposal(trace, t, cov, var)
   {pose_addr(t)} ~ pose_gaussian(trace[pose_addr(t)], cov, var) 
end
# -

# # Inference

# +
T = 5
constraints = 
    Gen.choicemap(
        pose_addr(1)=>P.Pose([3.0, 0.0, 3.0], R.RotY(pi/4))
    )
for t in 2:T
    constraints[pose_addr(t)] = constraints[pose_addr(t-1)] * P.Pose(zeros(3), R.RotY(deg2rad(25.0)))
end

import LinearAlgebra
I = Matrix{Float64}(LinearAlgebra.I, camera_intrinsics.width, camera_intrinsics.width) * 0.1;
tr_gt, w = Gen.generate(slam_multi_timestep, (T, nothing, room_bounds_uniform_params,I,), constraints);
viz_trace(tr_gt, [1,2,3,4,5])
# -

import GenParticleFilters
PF = GenParticleFilters

# +
corners = get_corners(get_depth(tr_gt,1))
poses = []
for c in corners
    for c2 in gt_corners
        p = c2 * inv(c)
        push!(poses, p)
    end
end

t = 1
@time pf_state = PF.pf_initialize(slam_multi_timestep,
    (1,Gen.get_args(tr_gt)[2:end]...), Gen.choicemap(sense_addr(t) => get_depth(tr_gt,t)[:]), 
    pose_mixture_proposal, (nothing, poses, 1,[1.0 0.0;0.0 1.0] * 0.01, deg2rad(1.0)),
    10);

PF.pf_move_accept!(pf_state, Gen.metropolis_hastings, (position_drift_proposal, 
        (t,[1.0 0.0;0.0 1.0] * 0.5)), 50);
PF.pf_move_accept!(pf_state, Gen.metropolis_hastings, (head_direction_drift_proposal, 
        (t,deg2rad(5.0))), 50);
PF.pf_move_accept!(pf_state, Gen.metropolis_hastings, (joint_pose_drift_proposal, 
        (t,[1.0 0.0;0.0 1.0] * 0.5, deg2rad(5.0))), 100);
# -

best_idx = argmax(pf_state.log_weights)
best_tr = pf_state.traces[best_idx]
viz_trace(best_tr, [1])
# viz_pose(get_pose(best_tr,1))

for t in 3:3
    PF.pf_update!(pf_state,
                  (t, Gen.get_args(tr_gt)[2:end]...),
                  (Gen.UnknownChange(),[Gen.NoChange() for _ in 1:(length(Gen.get_args(tr_gt))-1)]...),
                   Gen.choicemap(sense_addr(t) => get_depth(tr_gt,t)[:]));

    corners = get_corners(get_depth(tr_gt,t))
    poses = []
    for c in corners
        for c2 in gt_corners
            p = c2 * inv(c)
            push!(poses, p)
        end
    end
    
    PF.pf_move_accept!(pf_state, Gen.metropolis_hastings, (pose_mixture_proposal, 
            (poses, t, [1.0 0.0;0.0 1.0] * 0.05, deg2rad(5.0))), 10);
    PF.pf_move_accept!(pf_state, Gen.metropolis_hastings, (position_drift_proposal, 
            (t,[1.0 0.0;0.0 1.0] * 0.5)), 50);
    PF.pf_move_accept!(pf_state, Gen.metropolis_hastings, (head_direction_drift_proposal, 
            (t,deg2rad(5.0))), 50);
    PF.pf_move_accept!(pf_state, Gen.metropolis_hastings, (joint_pose_drift_proposal, 
            (t,[1.0 0.0;0.0 1.0] * 0.5, deg2rad(5.0))), 100);
    PF.pf_move_accept!(pf_state, Gen.metropolis_hastings, (position_drift_proposal, 
            (t,[1.0 0.0;0.0 1.0] * 0.5)), 50);
    PF.pf_move_accept!(pf_state, Gen.metropolis_hastings, (head_direction_drift_proposal, 
            (t,deg2rad(5.0))), 50);
    PF.pf_move_accept!(pf_state, Gen.metropolis_hastings, (joint_pose_drift_proposal, 
            (t,[1.0 0.0;0.0 1.0] * 0.5, deg2rad(5.0))), 100);
end

order = sortperm(pf_state.log_weights,rev=true)
best_tr = pf_state.traces[order[1]]
viz_trace(best_tr, [1,2,3])
# viz_pose(get_pose(best_tr,1))



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

