# -*- coding: utf-8 -*-
# +
# ENV["PYTHON"] = "/home/falk/.julia/dev/GLRenderer.jl/venv/bin/python"

import Revise
import GLRenderer
import PoseComposition
import Rotations
import Geometry
import Plots
import Images

import Gen
import Distributions

import GenParticleFilters
PF = GenParticleFilters

I = Images
PL = Plots

R = Rotations
P = PoseComposition
GL = GLRenderer
# -

Revise.errors()
Revise.revise()

# +
cloud = rand(3,100) * 1.0
resolution = 0.1
v,n,f = GL.mesh_from_voxelized_cloud(GL.voxelize(cloud, resolution), resolution)

renderer = GL.setup_renderer(Geometry.CameraIntrinsics(
    640, 480,
    1000.0, 1000.0,
    320.0, 240.0,
    0.01, 20.0
), GL.RGBMode())
GL.load_object!(renderer, v,n,f)

renderer.gl_instance.lightpos = [0,0,0]
rgb_image, depth_image = GL.gl_render(renderer, 
    [1], [P.Pose([0.0, 0.0, 4.0], R.RotXYZ(0.0, 0.0, 0.0))], 
    [I.colorant"green"],
    P.IDENTITY_POSE)
I.colorview(I.RGBA, permutedims(rgb_image,(3,1,2)))

# +
room_height_bounds = (-5.0, 5.0)
room_width_bounds = (-8.0, 8.0)

resolution = 0.1
room_cloud_1 = []
room_cloud_2 = []
room_cloud_3 = []
room_cloud_4 = []

for z in room_height_bounds[1]:resolution/2.0:room_height_bounds[2]
    push!(room_cloud_1, [room_width_bounds[1], 0.0, z])
    push!(room_cloud_2, [room_width_bounds[2], 0.0, z])
end
for x in room_width_bounds[1]:resolution/2.0:room_width_bounds[2]
    push!(room_cloud_3, [x, 0.0, room_height_bounds[1]])
    push!(room_cloud_4, [x, 0.0, room_height_bounds[2]])
end
<<<<<<< HEAD
v1,n1,f1 = GL.mesh_from_voxelized_cloud(GL.voxelize(hcat(room_cloud_1...), resolution), resolution)
v2,n2,f2 = GL.mesh_from_voxelized_cloud(GL.voxelize(hcat(room_cloud_2...), resolution), resolution)
v3,n3,f3 = GL.mesh_from_voxelized_cloud(GL.voxelize(hcat(room_cloud_3...), resolution), resolution)
v4,n4,f4 = GL.mesh_from_voxelized_cloud(GL.voxelize(hcat(room_cloud_4...), resolution), resolution)

room_cloud = hcat(room_cloud_1...,room_cloud_2...,room_cloud_3...,room_cloud_4...)

PL.scatter(v1[:,1],v1[:,3],label="")
PL.scatter!(v2[:,1],v2[:,3],label="")
PL.scatter!(v3[:,1],v3[:,3],label="")
PL.scatter!(v4[:,1],v4[:,3],label="")
# -

=======
room_cloud = hcat(room_cloud...)
PL.scatter(room_cloud[1,:],room_cloud[3,:],label="")
# -

v,n,f = GL.mesh_from_voxelized_cloud(GL.voxelize(room_cloud, resolution), resolution)
>>>>>>> b9b7409... Potential mode collapse
renderer = GL.setup_renderer(Geometry.CameraIntrinsics(
    600, 600,
    300.0, 300.0,
    300.0,300.0,
    0.1, 60.0
), GL.RGBMode())
GL.load_object!(renderer, v3, n3, f3)
GL.load_object!(renderer, v4, n4, f4)
GL.load_object!(renderer, v2, n2, f2)
GL.load_object!(renderer, v1, n1, f1)
renderer.gl_instance.lightpos = [0,0,0]
rgb, depth = GL.gl_render(renderer, 
    [1,2,3,4], [P.IDENTITY_POSE, P.IDENTITY_POSE,P.IDENTITY_POSE,P.IDENTITY_POSE],
    [I.colorant"red",I.colorant"green",I.colorant"blue",I.colorant"yellow"],
    P.Pose([0.0, -10.0, 0.0], R.RotX(-pi/2)))
# PL.heatmap(depth, aspect_ratio=:equal)
I.colorview(I.RGBA,permutedims(rgb,(3,1,2)))

# +
camera_intrinsics = Geometry.CameraIntrinsics(
    14, 1,
    5.0, 1.0,
    7.0, 0.5,
    0.1, 20.0
)
renderer = GL.setup_renderer(camera_intrinsics, GL.RGBMode())
GL.load_object!(renderer, v3, n3, f3)
GL.load_object!(renderer, v4, n4, f4)
GL.load_object!(renderer, v2, n2, f2)
GL.load_object!(renderer, v1, n1, f1)
renderer.gl_instance.lightpos = [0,0,0]


cam_pose = P.Pose(zeros(3),R.RotY(-pi/4+ 1.0))
wall_colors = [I.colorant"red",I.colorant"green",I.colorant"blue",I.colorant"yellow"]
# wall_colors = [I.colorant"red",I.colorant"red",I.colorant"red",I.colorant"red"]
@time rgb, depth = GL.gl_render(renderer, 
     [1,2,3,4], [P.IDENTITY_POSE, P.IDENTITY_POSE,P.IDENTITY_POSE,P.IDENTITY_POSE],
    wall_colors, 
    cam_pose)
PL.heatmap(depth)

img = I.colorview(I.RGB,permutedims(rgb,(3,1,2))[1:3,:,:])
color = map(argmin, eachcol(vcat([I.colordiff.(img, c) for c in wall_colors]...)))
@show color 
img
# -

import Gen
import Distributions

# +
struct PoseUniform <: Gen.Distribution{P.Pose} end
const pose_uniform = PoseUniform()
function Gen.random(::PoseUniform, bounds_x, bounds_z)
    x = rand(Distributions.Uniform(bounds_x...))
    z = rand(Distributions.Uniform(bounds_z...))
    hd = rand(Distributions.Uniform(-2*pi, 2*pi))
    P.Pose([x, 0.0, z], R.RotY(hd))
end
function Gen.logpdf(::PoseUniform, pose::P.Pose, bounds_x, bounds_z)
    (
        Gen.logpdf(Gen.uniform, pose.pos[1], bounds_x...) +
        Gen.logpdf(Gen.uniform, pose.pos[3], bounds_z...) +
        Gen.logpdf(Gen.uniform, R.RotY(pose.orientation).theta, -2*pi, 2*pi)
    )   
end


struct PoseGaussian <: Gen.Distribution{P.Pose} end
const pose_gaussian = PoseGaussian()
function Gen.random(::PoseGaussian, center_pose::P.Pose, cov, var)
    pos = Gen.random(Gen.mvnormal, [center_pose.pos[1], center_pose.pos[3]], cov)

    hd_center = R.RotY(center_pose.orientation).theta
    hd = Gen.random(Gen.normal, hd_center, var)
    P.Pose([pos[1], 0.0, pos[2]], R.RotY(hd))
end
function Gen.logpdf(::PoseGaussian, pose::P.Pose, center_pose::P.Pose, cov, var)
    hd = R.RotY(pose.orientation).theta
    hd_center = R.RotY(center_pose.orientation).theta
    (
        Gen.logpdf(Gen.mvnormal, [pose.pos[1], pose.pos[3]], [center_pose.pos[1], center_pose.pos[3]], cov) +
        Gen.logpdf(Gen.normal, hd, hd_center, var)
    )   
end


struct ColorDistribution <: Gen.Distribution{Array{<:Real}} end
const color_distribution = ColorDistribution()
function Gen.random(::ColorDistribution, color, p)
    # This is incorrect
    color
end
function Gen.logpdf(::ColorDistribution, obs_color, color, p)
    n = length(wall_colors)
    img = Images.colorview(Images.RGB,permutedims(obs_color, (3,1,2))[1:3,:,:])
    obs = map(argmin, eachcol(vcat([I.colordiff.(img, c) for c in wall_colors]...)))
    
    img = Images.colorview(Images.RGB,permutedims(color, (3,1,2))[1:3,:,:])
    base = map(argmin, eachcol(vcat([I.colordiff.(img, c) for c in wall_colors]...)))
    
    disagree = sum(base .!= obs)
    agree = sum(base .== obs)
    
    agree * log(p) + disagree * log( (1-p)/(n-1))
end
# -

room_bounds_uniform_params = [room_width_bounds[1] room_width_bounds[2];room_height_bounds[1] room_height_bounds[2]]

# +
@Gen.gen function slam_unfold_kernel(t, prev_data, room_bounds, wall_colors, cov)
    if t==1
        pose ~ pose_uniform(room_bounds[1,:],room_bounds[2,:])
    else
        pose ~ pose_gaussian(prev_data.pose, [1.0 0.0;0.0 1.0] * 0.1, deg2rad(20.0))
    end
    rgb, depth = GL.gl_render(renderer, 
     [1,2,3,4], [P.IDENTITY_POSE, P.IDENTITY_POSE,P.IDENTITY_POSE,P.IDENTITY_POSE],
    wall_colors, 
    pose)
    
    sense_depth ~ Gen.mvnormal(depth[1,:], cov)
    sense_rgb ~ color_distribution(rgb, 0.999)
    return (pose=pose, rgb=rgb, depth=depth, sense_depth=sense_depth, sense_rgb=sense_rgb)
end

slam_unfolded = Gen.Unfold(slam_unfold_kernel)

@Gen.gen (static) function slam_multi_timestep(T, prev_data, room_bounds, wall_colors, cov)
    slam ~ slam_unfolded(T, prev_data, room_bounds, wall_colors, cov)
    return slam
end


sense_rgb_addr(t) = (:slam => t => :sense_rgb)
sense_depth_addr(t) = (:slam => t => :sense_depth)
pose_addr(t) = (:slam => t => :pose)
get_pose(tr,t) = tr[pose_addr(t)]
get_depth(tr,t) = Gen.get_retval(tr)[t].depth
get_rgb(tr,t) = Gen.get_retval(tr)[t].rgb
get_sense_rgb(tr,t) = tr[sense_rgb_addr(t)]
get_sense_depth(tr,t) = tr[sense_depth_addr(t)]

Gen.@load_generated_functions

function viz_env()
    PL.scatter(room_cloud[1,:], room_cloud[3,:], label=false)
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
    sense = get_sense_depth(tr,t)
    
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

# +
# constraints = Gen.choicemap(:pos=>[0.0, 0.0], :hd=>pi/4)

T_gen = 12

import LinearAlgebra
cov = Matrix{Float64}(LinearAlgebra.I, camera_intrinsics.width, camera_intrinsics.width) * 0.01;
tr_gt, w = Gen.generate(slam_multi_timestep, (5, nothing, room_bounds_uniform_params, wall_colors, cov,));
viz_trace(tr_gt,1)

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
# TODO: Write corner detection function that finds corners given an occupancy grid (aka point cloud) of the room.
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
        check_valid(idx) = (idx > 0 ) && (idx  <= size(cloud)[2])
        if !all(map(check_valid, [i,j,k,l]))
            continue
        end
            
        
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

sense = get_depth(tr_gt,1)
corners = get_corners(sense)
cloud = GL.flatten_point_cloud(GL.depth_image_to_point_cloud(reshape(sense,(camera_intrinsics.height, camera_intrinsics.width)), camera_intrinsics))
PL.scatter(cloud[1,:], cloud[3,:])
for c in corners
    viz_corner(c)
end
PL.plot!(xlim=(-10,10),ylim=(-10,10), aspect_ratio=:equal)

# +
sense = get_depth(tr_gt,1)

corners = get_corners(sense)
poses = []
for c in corners
    for c2 in gt_corners
        p = c2 * inv(c)
        push!(poses, p)
    end
end

PL.plot()
viz_env()
for p in poses
   viz_pose(p) 
end
for c in gt_corners
   viz_corner(c) 
end
PL.plot!()

# +
mixture_of_pose_gaussians = Gen.HomogeneousMixture(pose_gaussian, [0, 2, 0])

@Gen.gen function pose_mixture_proposal(trace, poses, t, cov, var)
    n = length(poses)
    weights = ones(n) ./ n
    {pose_addr(t)} ~ mixture_of_pose_gaussians(
        weights, poses, cat([cov for _ in 1:n]..., dims=3), [var for _ in 1:n]
    )
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
# T = T_gen
T = 12  # TODO 7 or above will cause error 3 cells down - why?
constraints = 
    Gen.choicemap(
        pose_addr(1)=>P.Pose([-3.0, 0.0, -1.0], R.RotY(pi+pi/4))
    )
for t in 2:T
    constraints[pose_addr(t)] = constraints[pose_addr(t-1)] * P.Pose(zeros(3), R.RotY(deg2rad(25.0)))
end

import LinearAlgebra
<<<<<<< HEAD
# wall_colors = [I.colorant"red",I.colorant"green",I.colorant"blue",I.colorant"yellow"]
wall_colors = [I.colorant"red",I.colorant"red",I.colorant"red",I.colorant"red"]
cov = Matrix{Float64}(LinearAlgebra.I, camera_intrinsics.width, camera_intrinsics.width) * 0.5;
tr_gt, w = Gen.generate(slam_multi_timestep, (T, nothing, room_bounds_uniform_params,wall_colors,cov,), constraints);
@show Gen.get_score(tr_gt)
viz_trace(tr_gt, [1])
=======
I = Matrix{Float64}(LinearAlgebra.I, camera_intrinsics.width, camera_intrinsics.width) * 0.1;
tr_gt, w = Gen.generate(slam_multi_timestep, (T, nothing, room_bounds_uniform_params,I,), constraints);
viz_trace(tr_gt, [2])
>>>>>>> b9b7409... Potential mode collapse

# +
function get_corners2(sense)
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
        if i > 0 && j > 0 && k > 0 && l > 0
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
    end
    corners
end

corners = get_corners2(get_depth(tr_gt,1))
poses = []
for c in corners
    for c2 in gt_corners
        p = c2 * inv(c)
        push!(poses, p)
    end
end

t = 1
@time pf_state = PF.pf_initialize(slam_multi_timestep,
    (1,Gen.get_args(tr_gt)[2:end]...),
    Gen.choicemap(sense_depth_addr(t) => get_depth(tr_gt,t)[:], sense_rgb_addr(t) => get_rgb(tr_gt,t)),
    pose_mixture_proposal, (nothing, poses, 1,[1.0 0.0;0.0 1.0] * 0.05, deg2rad(2.0)),
    100);

PF.pf_move_accept!(pf_state, Gen.metropolis_hastings, (position_drift_proposal, 
        (t,[1.0 0.0;0.0 1.0] * 0.1)), 10);
PF.pf_move_accept!(pf_state, Gen.metropolis_hastings, (head_direction_drift_proposal, 
        (t,deg2rad(1.0))), 10);
PF.pf_move_accept!(pf_state, Gen.metropolis_hastings, (joint_pose_drift_proposal, 
        (t,[1.0 0.0;0.0 1.0] * 0.1, deg2rad(1.0))), 10);
# -

order = sortperm(pf_state.log_weights,rev=true)
best_tr = pf_state.traces[order[3]]
@show Gen.get_score(best_tr)
@show Gen.project(best_tr, Gen.select(pose_addr(1)))
pose = best_tr[pose_addr(1)]
@show pose
viz_trace(best_tr, [1])
# viz_pose(get_pose(best_tr,1))

z = Gen.logsumexp(pf_state.log_weights)
log_weights = pf_state.log_weights .- z
weights = exp.(log_weights)
@show sum(weights)
weights

tr = best_tr
a=I.colorview(I.RGBA,permutedims(get_rgb(tr,1),(3,1,2)))
b = I.colorview(I.RGBA,permutedims(get_sense_rgb(tr,1),(3,1,2)))
vcat(a,b)

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
    (1,Gen.get_args(tr_gt)[2:end]...),
    Gen.choicemap(sense_depth_addr(t) => get_depth(tr_gt,t)[:], sense_rgb_addr(t) => get_rgb(tr_gt,t)),
    pose_mixture_proposal, (nothing, poses, 1,[1.0 0.0;0.0 1.0] * 0.05, deg2rad(2.0)),
    100);

PF.pf_move_accept!(pf_state, Gen.metropolis_hastings, (position_drift_proposal, 
        (t,[1.0 0.0;0.0 1.0] * 0.1)), 10);
PF.pf_move_accept!(pf_state, Gen.metropolis_hastings, (head_direction_drift_proposal, 
        (t,deg2rad(1.0))), 10);
PF.pf_move_accept!(pf_state, Gen.metropolis_hastings, (joint_pose_drift_proposal, 
        (t,[1.0 0.0;0.0 1.0] * 0.1, deg2rad(1.0))), 10);

# +
for t in 2:T
    PF.pf_update!(pf_state,
                  (t, Gen.get_args(tr_gt)[2:end]...),
                  (Gen.UnknownChange(),[Gen.NoChange() for _ in 1:(length(Gen.get_args(tr_gt))-1)]...),
    Gen.choicemap(sense_depth_addr(t) => get_depth(tr_gt,t)[:], sense_rgb_addr(t) => get_rgb(tr_gt,t)),
    );

    
    # Geometrically computed pose proposal
    corners = get_corners2(get_depth(tr_gt, t))  # get_corners(get_depth(tr_gt, t))
    poses = []
    for c in corners
        for c2 in gt_corners
            p = c2 * inv(c)
            push!(poses, p)
        end
    end
    PF.pf_move_accept!(pf_state, Gen.metropolis_hastings, (pose_mixture_proposal, 
            (poses, t, [1.0 0.0;0.0 1.0] * 0.05, deg2rad(2.0))), 10);

    # Drift Moves
    PF.pf_move_accept!(pf_state, Gen.metropolis_hastings, (position_drift_proposal, 
            (t,[1.0 0.0;0.0 1.0] * 0.5)), 10);
    PF.pf_move_accept!(pf_state, Gen.metropolis_hastings, (head_direction_drift_proposal, 
            (t,deg2rad(5.0))), 10);
    PF.pf_move_accept!(pf_state, Gen.metropolis_hastings, (joint_pose_drift_proposal, 
            (t,[1.0 0.0;0.0 1.0] * 0.5, deg2rad(5.0))), 10);
    
    PF.pf_move_accept!(pf_state, Gen.metropolis_hastings, (position_drift_proposal, 
            (t,[1.0 0.0;0.0 1.0] * 0.5)), 10);
    PF.pf_move_accept!(pf_state, Gen.metropolis_hastings, (head_direction_drift_proposal, 
            (t,deg2rad(5.0))), 10);
    PF.pf_move_accept!(pf_state, Gen.metropolis_hastings, (joint_pose_drift_proposal, 
            (t,[1.0 0.0;0.0 1.0] * 0.5, deg2rad(5.0))), 10);
end
# -

order = sortperm(pf_state.log_weights,rev=true)

best_tr = pf_state.traces[order[1]]
viz_trace(best_tr, 1:T)
# viz_pose(get_pose(best_tr,1))

viz_trace(tr_gt, 1:T)

z = Gen.logsumexp(pf_state.log_weights)
log_weights = pf_state.log_weights .- z
weights = exp.(log_weights)
@show sum(weights)
weights

# +
function viz_trace2(tr, ts)
#     PL.plot()
    T = Gen.get_args(tr)[1]
    viz_env()
    for t in ts
        viz_pose(get_pose(tr,t))
        viz_obs(tr, t)
    end
#     PL.plot!()
end

PL.plot()
for tr_idx in 1:length(pf_state.traces)
#     println(tr_idx)
    trac = pf_state.traces[tr_idx]
    #println(trac)
    viz_trace2(trac, 1:T)
end
PL.plot!()

# +
using Clustering  # import Pkg; Pkg.add("Clustering")

function collect_points(trs, t_max)
    positions = []
    for tr_idx in 1:length(trs)
        tr = trs[tr_idx]
        T = Gen.get_args(tr)[1]

        for t in 1:t_max
            push!(positions, get_pose(tr, t).pos)
        end
    end
    positions
end
    
pts = collect_points(pf_state.traces, T);
# -

pts_mat = vcat(hcat(pts...))
R = kmeans(pts_mat, 4; maxiter=200, display=:iter)

# +
import PyPlot
plt = PyPlot.plt

println(R.counts)
println(R.centers)
plt.figure(figsize=(4,4)); plt.gca().set_aspect(1.);

vals = R.counts/maximum(R.counts)

plt.scatter(R.centers[1, :], R.centers[3, :], c="red", marker="o", alpha=vals)
# -



order = sortperm(pf_state.log_weights,rev=true)
best_tr = pf_state.traces[order[1]]
viz_trace(best_tr, 1:max_t)





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

