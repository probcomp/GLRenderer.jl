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

# See [SLAM PR to PRAM](https://github.com/probcomp/GenPRAM.jl/blob/slam_build_map/agent_experiments/slam_poc/slam_3d_all_in_one.ipynb) for example how to do edge detection against MCS.

# +
import PyPlot
plt = PyPlot.plt

using PyCall
@pyimport machine_common_sense as mcs
StepMetadata=mcs.StepMetadata
ObjectMetadata=mcs.ObjectMetadata

global numpy = pyimport("numpy")
@pyimport numpy as np

# +
using Serialization
using StaticArrays
import GenGridSLAM: astar_search, backproject_grid_coord_to_world_coord, CameraIntrinsics, ccl, centroid_direction, centroid_direction_2, cloud_from_step_metadata,
    cloud_to_grid_coords, compute_frontier,
    convert_probabilistic_occupancy_grid_to_string_array, count_component_size,
    create_2d_occupancy_grid_with_segment_ids, expand_2d_occupancy_grid_dict_into_full_matrix, find_closest_unexplored_component,
    find_max_component, flip_array, generate_pddl, get_agent_grid_position_and_rotation, grid_coords_to_cloud, grid_xz_to_map_xz,
    in_grid_bounds, map_xz_to_grid_xz,
    OccupancyGrid, OccupancyGridConfig, pretty_print_occupancy_grid, pretty_print_occupancy_grid_string_array,
    print_occupancy_grid, project_3d_occupancy_grid_into_2d_dict, render_occupancy_grid, render_occupancy_grid_oriented,
    set_entry!, update_occupancy_grid!, project_3d_og_to_2d, generate_maps_2d_oriented, world_point_to_oriented_grid_coordinates

pram_path = ENV["PRAM_PATH"]
include("$(pram_path)/GenAgent/src/config/config.jl")
include("$(pram_path)/GenAgent/src/state/agent.jl")
include("$(pram_path)/GenAgent/src/submission.jl")

scene_path="/home/falk/.julia/dev/GenPRAM.jl/test/test_scenes/rectangular_colored_walls_empty.json"
mcs_executable_path="/home/falk/mitibm/AI2Thor_MCS/MCS-AI2-THOR-Unity-App-v0.4.3.x86_64"
mcs_config_path="/home/falk/PycharmProjects/MCS/scripts/config_oracle.ini"

config_data, status = mcs.load_scene_json_file(scene_path)
controller = mcs.create_controller(unity_app_file_path=mcs_executable_path,
                                   config_file_path=mcs_config_path)
step_metadata = controller.start_scene(config_data)

config = McsConfiguration()
information_level = infer_information_level(step_metadata)

intrinsics = CameraIntrinsics(step_metadata)
agent = Agent(intrinsics, [step_metadata], [],
              [Pose([0.0,0.0,0.0],  0.0, step_metadata.head_tilt)],
              [], nothing, information_level, config)

# Overwrite pram_mode so we don't need to deal with Redis etc.
ENV["PRAM_MODE"] = "production"
agent.config.compute_configuration.pram_mode = ENV["PRAM_MODE"]

# Load target model
trophy_model_path = joinpath(config.path_configuration.pram_path, "GenAgent/omg/models/soccer_ball.model")
println("Loading trophy model from $(trophy_model_path).")
trophy_model = deserialize(trophy_model_path)

# +
# Could use global information
# function find_scanline(valid_index_mask)
#     valid_rows = prod(valid_index_mask, dims=2)  # (400, 1) vector - pick index with value 1
#     for row_idx in 1:size(valid_rows)[1]
#         if valid_rows[row_idx]
#             return row_idx
#         end
#     end
#     return -1
# end

# """Beware: This function uses agent pose information that might not be available after kidnapping.
# Based on `detect_corner_from_depth(Agent)`."""
# function get_scanline(agent::Agent; verbose::Bool=false)
#     step_metadata = agent.step_metadatas[end]
#     pose = agent.poses[end]  # <-- Global pose information
#     cloud = get_cloud(agent, step_metadata, pose, agent.intrinsics; stride_x=1, stride_y=1)
    
#     height_offset = 0.1
#     min_height, max_height = minimum(cloud[2, :]) + height_offset, maximum(cloud[2, :]) - height_offset
#     if verbose println("Height range: $(min_height) to $(max_height)") end
    
#     # Boolean mask for all pixels that are definitely not ceiling or floor
#     valid_indices = (cloud[2,:] .> min_height) .& (cloud[2,:] .< max_height)
#     valid_indices_2d = reshape(valid_indices, size(agent.step_metadatas[end].depth_map_list[end]))
    
#     scanline_idx = find_scanline(valid_indices_2d)

#     # Reshape cloud to 8x400x600 for easier access
#     cloud_array_2d = reshape(cloud, (8, size(agent.step_metadatas[end].depth_map_list[end])...));

#     # Pick scanline (scanline_idx) and only look at 3D coordinates (1:3) => 3x600 matrix
#     return cloud_array_2d, scanline_idx
# end

# rows_of_interest = []

# for _ in 1:36
#     execute_command(controller, agent, "RotateRight")
#     cloud_array_2d, scanline_idx = get_scanline(agent)
#     roi = cloud_array_2d[1:3, scanline_idx, :]
#     if !isnothing(roi) push!(rows_of_interest, roi) end
# end

# +
"""
Same as `get_scanline()`, but works on camera frame point cloud
without depending on global coordinates. Based on `detect_corner_from_camera_frame(agent)`.
"""
function get_scanline_from_camera_frame(agent::Agent; verbose = false)
    
    step_metadata = agent.step_metadatas[end]
    
    cloud = cloud_from_depth_map(
                Matrix(numpy.array(last(step_metadata.depth_map_list))),
                intrinsics.cx, intrinsics.cy,
                intrinsics.fx, intrinsics.fy,
                intrinsics.width, intrinsics.height; stride_x=1, stride_y=1
            )
    cloud_array_2d = reshape(cloud, (3, size(agent.step_metadatas[end].depth_map_list[end])...))
    
    # We could get point cloud in world frame like this, but don't know our pose after kidnapping:
    # pose = agent.poses[end]
    # cloud = rotate_cloud(cloud; head_tilt_deg=pose.head_tilt, view_angle_deg=pose.rotation)
    # cloud = translate_cloud(cloud, pose.position)
    # cloud_array_2d_b = reshape(cloud, (3, size(agent.step_metadatas[end].depth_map_list[end])...));
    
    height_offset = 0.1
    min_height, max_height = minimum(cloud[2, :]) + height_offset, maximum(cloud[2, :]) - height_offset
    if verbose println("Height range: $(min_height) to $(max_height)") end

    # Boolean mask for all pixels that are definitely not ceiling or floor
    valid_indices = (cloud[2,:] .> min_height) .& (cloud[2,:] .< max_height)
    valid_indices_2d = reshape(valid_indices, size(agent.step_metadatas[end].depth_map_list[end]))

    scanline_idx = find_scanline(valid_indices_2d)
    
    return cloud_array_2d, scanline_idx
end

rows_of_interest_local = []

for _ in 1:36
    execute_command(controller, agent, "RotateRight")
    cloud_array_2d, scanline_idx = get_scanline_from_camera_frame(agent)
    roi = cloud_array_2d[1:3, scanline_idx, :]
    if !isnothing(roi) push!(rows_of_interest_local, roi) end
end
# -

# for row in rows_of_interest_local
#     println(row)
#     break
# end
row = rows_of_interest_local[1]
plt.scatter(row[1,:], row[3,:], c="lightgray", marker="o")



# +
cloud = rand(3,100) * 1.0
resolution = 0.1
v, n, f = GL.mesh_from_voxelized_cloud(GL.voxelize(cloud, resolution), resolution)

renderer = GL.setup_renderer(Geometry.CameraIntrinsics(
    640, 480,
    1000.0, 1000.0,
    320.0, 240.0,
    0.01, 20.0
), GL.RGBMode())
GL.load_object!(renderer, v, n, f)

renderer.gl_instance.lightpos = [0,0,0]
rgb_image, depth_image = GL.gl_render(renderer, 
    [1], [P.Pose([0.0, 0.0, 4.0], R.RotXYZ(0.0, 0.0, 0.0))], 
    [I.colorant"red"],
    P.IDENTITY_POSE)
I.colorview(I.RGBA, permutedims(rgb_image,(3,1,2)))

# +
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
PL.scatter(room_cloud[1,:], room_cloud[3,:], label="")
# -

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

# +
camera_intrinsics = Geometry.CameraIntrinsics(
    14, 1,
    5.0, 1.0,
    7.0, 0.5,
    0.1, 20.0
)
renderer = GL.setup_renderer(camera_intrinsics, GL.DepthMode())
GL.load_object!(renderer, v, f)
renderer.gl_instance.lightpos = [0,0,0]

cam_pose = P.IDENTITY_POSE
@time depth = GL.gl_render(renderer, 
    [1], [P.IDENTITY_POSE], cam_pose)
cloud = GL.flatten_point_cloud(GL.depth_image_to_point_cloud(depth, camera_intrinsics))
@show size(depth)
PL.heatmap(depth)
# +
struct PoseUniform <: Gen.Distribution{P.Pose} end
const pose_uniform = PoseUniform()
function Gen.random(::PoseUniform, bounds_x, bounds_z)
    x = rand(Distributions.Uniform(bounds_x...))
    z = rand(Distributions.Uniform(bounds_z...))
    hd = rand(Distributions.Uniform(0.0, 2*pi))
    P.Pose([x, 0.0, z], R.RotY(hd))
end
function Gen.logpdf(::PoseUniform, pose::P.Pose, bounds_x, bounds_z)
    (
        Gen.logpdf(Gen.uniform, pose.pos[1], bounds_x...) +
        Gen.logpdf(Gen.uniform, pose.pos[3], bounds_z...) +
        Gen.logpdf(Gen.uniform, R.RotY(pose.orientation).theta, 0.0, 2*pi)
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

# -

room_bounds_uniform_params = [room_width_bounds[1] room_width_bounds[2];room_height_bounds[1] room_height_bounds[2]]

# +
@Gen.gen function slam_unfold_kernel(t, prev_data, room_bounds, I)
    if t==1
        pose ~ pose_uniform(room_bounds[1,:],room_bounds[2,:])
    else
        # Note: deg2rad() here gives how far agent rotates
        pose ~ pose_gaussian(prev_data.pose, [1.0 0.0;0.0 1.0] * 0.1, deg2rad(10.0))
    end
    depth = GL.gl_render(renderer, [1], [P.IDENTITY_POSE], pose)    
    
    sense ~ Gen.mvnormal(depth[1,:], I)
    return (pose=pose, depth=depth, sense=sense)
end

slam_unfolded = Gen.Unfold(slam_unfold_kernel)

@Gen.gen (static) function slam_multi_timestep(T, prev_data, room_bounds, I)
    slam ~ slam_unfolded(T, prev_data, room_bounds, I)
    return slam
end


sense_addr(t) = (:slam => t => :sense)
pose_addr(t) = (:slam => t => :pose)
get_pose(tr,t) = tr[pose_addr(t)]
get_depth(tr,t) = Gen.get_retval(tr)[t].depth
get_sense(tr,t) = tr[sense_addr(t)]


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
    PL.plot!([pos[1], pos[1]+direction[1]],[pos[3], pos[3]+direction[3]], arrow=true, color=:red,linewidth=2, label=false)
end

# +
# constraints = Gen.choicemap(:pos=>[0.0, 0.0], :hd=>pi/4)

T_gen = 36

import LinearAlgebra
I = Matrix{Float64}(LinearAlgebra.I, camera_intrinsics.width, camera_intrinsics.width) * 0.01;
tr_gt, w = Gen.generate(slam_multi_timestep, (T_gen, nothing, room_bounds_uniform_params,I,));
viz_trace(tr_gt, 1)
# -

tr_gt[pose_addr(36)]

viz_trace(tr_gt, 36)

# # Corner Detection

PL.plot()
viz_env()
corner_1 = P.Pose([room_width_bounds[1], 0.0, room_height_bounds[1]], R.RotY(-pi/4))
corner_2 = P.Pose([room_width_bounds[1], 0.0, room_height_bounds[2]], R.RotY(pi/4))
corner_3 = P.Pose([room_width_bounds[2], 0.0, room_height_bounds[1]], R.RotY(pi+pi/4))
corner_4 = P.Pose([room_width_bounds[2], 0.0, room_height_bounds[2]], R.RotY(pi-pi/4))
gt_corners = [corner_1, corner_2, corner_3, corner_4]
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
        i,j = s-2, s-1
        k,l = s+2, s+1
        
        if i > 0 && i <= size(cloud)[2] && j > 0  && j <= size(cloud)[2] && k > 0 && k <= size(cloud)[2] && l > 0 && l <= size(cloud)[2]
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

sense = get_depth(tr_gt, 1)
corners = get_corners(sense)
cloud = GL.flatten_point_cloud(GL.depth_image_to_point_cloud(reshape(sense,(camera_intrinsics.height, camera_intrinsics.width)), camera_intrinsics))
PL.scatter(cloud[1, :], cloud[3, :])
for c in corners
    viz_corner(c)
end
PL.plot!(xlim=(-10,10), ylim=(-10,10), aspect_ratio=:equal)

# +
sense = get_depth(tr_gt, 1)

corners = get_corners(sense)
poses = []
for c in corners
    for c2 in gt_corners
        p = c2 * inv(c)
        push!(poses, p)
    end
end
poses_geometry = copy(poses)
# -

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
@Gen.gen function position_drift_proposal(trace, t, cov)
   {pose_addr(t)} ~ pose_gaussian(trace[pose_addr(t)], cov, 0.0001) 
end

@Gen.gen function head_direction_drift_proposal(trace, t, var)
   {pose_addr(t)} ~ pose_gaussian(trace[pose_addr(t)], [1.0 0.0;0.0 1.0] * 0.00001, var) 
end

@Gen.gen function joint_pose_drift_proposal(trace, t, cov, var)
   {pose_addr(t)} ~ pose_gaussian(trace[pose_addr(t)], cov, var) 
end
# -

# # Inference

# +
T = 36
constraints = 
    Gen.choicemap(
        pose_addr(1) => P.Pose([3.0, 0.0, 3.0], R.RotY(pi/4))
    )
for t in 2:T
    # deg2rad(25.0)
    constraints[pose_addr(t)] = constraints[pose_addr(t-1)] * P.Pose(zeros(3), R.RotY(deg2rad(10.0)))
end

import LinearAlgebra
I = Matrix{Float64}(LinearAlgebra.I, camera_intrinsics.width, camera_intrinsics.width) * 0.1;
tr_gt, w = Gen.generate(slam_multi_timestep, (T, nothing, room_bounds_uniform_params,I,), constraints);
viz_trace(tr_gt, [2])
# -

viz_trace(tr_gt, [25])

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
    10);  # <-- Number of particles

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

for t in 2:T
    
    # PF.pf_resample!(pf_state, :residual) would lead to mode collapse
    
    PF.pf_update!(pf_state,
                  (t, Gen.get_args(tr_gt)[2:end]...),
                  (Gen.UnknownChange(),[Gen.NoChange() for _ in 1:(length(Gen.get_args(tr_gt))-1)]...),
                   Gen.choicemap(sense_addr(t) => get_depth(tr_gt, t)[:]));

    
    # Geometrically computed pose proposal
    corners = get_corners(get_depth(tr_gt, t))
    poses = []
    for c in corners
        for c2 in gt_corners
            p = c2 * inv(c)
            push!(poses, p)
        end
    end
    if length(corners) > 0
        PF.pf_move_accept!(pf_state, Gen.metropolis_hastings, (pose_mixture_proposal, 
                (poses, t, [1.0 0.0; 0.0 1.0] * 0.05, deg2rad(5.0))), 10);
    end

    # Drift Moves
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
order = sortperm(pf_state.log_weights,rev=true)

for tr_idx in 1:8  # length(pf_state.traces)
    trac = pf_state.traces[order[tr_idx]]
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
cluster_r = kmeans(pts_mat, 4; maxiter=200, display=:iter)

# +
println(cluster_r.counts)
println(cluster_r.centers)
plt.figure(figsize=(4,4)); plt.gca().set_aspect(1.);

# Visualize inference cluster centers in red
vals = cluster_r.counts/maximum(cluster_r.counts)
# TODO alpha=vals does not always seem to work in next line - why?
plt.scatter(cluster_r.centers[1, :], cluster_r.centers[3, :], c="red", marker="o")

# Visualize geometry points and orientations for comparison in gray
for p in poses_geometry
    println(p)
    plt.scatter(p.pos[1], p.pos[3], c="lightgray", marker="o")
    
    # TODO @Nishad: Please double check this is what orientation means
    # I use orientation to rotate unit vector (0, 0, 1) and project result (x,y,z) to (x,z)
    # But I'm not sure about the scene geometry in GLRenderer
    base_vector = MVector{3,Float64}(0.0, 0.0, 1.0)
    ray = MVector{3,Float64}(p.orientation * base_vector)
    plt.quiver([p.pos[1]], [p.pos[3]], [ray[1]], [ray[3]],
                   color=["b"], angles="xy", scale_units="xy", scale=1.0, label="View")
end
# +
# vals
# size(cluster_r.centers)
# -


















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

