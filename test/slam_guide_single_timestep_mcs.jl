# -*- coding: utf-8 -*-
import Revise
Revise.errors()
Revise.revise()

# +
import GLRenderer
import PoseComposition
import Rotations
import Geometry
import LinearAlgebra
import Plots
import Images
import GenParticleFilters
PF = GenParticleFilters

I = Images
PL = Plots

R = Rotations
P = PoseComposition
GL = GLRenderer

import Gen
import Distributions

import PyPlot
plt = PyPlot.plt

using PyCall
@pyimport machine_common_sense as mcs
StepMetadata=mcs.StepMetadata
ObjectMetadata=mcs.ObjectMetadata

global numpy = pyimport("numpy")
@pyimport numpy as np

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
include("model_functions.jl")
include("mcs_common.jl")
# -

# ## Start MCS Environment

# +
scene_path=joinpath(pwd(), "data/rectangular_colored_walls_empty.json")
mcs_config_path=ENV["MCS_CONFIG_FILE_PATH"]
# mcs_executable_path=ENV["MCS_EXECUTABLE_PATH"]
mcs_executable_path = "/home/falk/mitibm/AI2Thor_MCS/MCS-AI2-THOR-Unity-App-v0.4.4-linux/MCS-AI2-THOR-Unity-App-v0.4.4.x86_64"

controller = mcs.create_controller(unity_app_file_path=mcs_executable_path,
                                   config_file_path=mcs_config_path)

# +
# config_data, status = mcs.load_scene_json_file(scene_path)
config_data = generate_scene(agent_pos_x = -2.0, agent_pos_z = 1.0)
step_metadata = controller.start_scene(config_data)

config = McsConfiguration()
information_level = infer_information_level(step_metadata)

intrinsics = CameraIntrinsics(step_metadata)
agent = Agent(intrinsics, [step_metadata], [],
              [Pose([0.0,0.0,0.0],  0.0, step_metadata.head_tilt)],
              [], nothing, information_level, config)

# Overwrite pram_mode so we don't need to deal with Redis etc.
agent.config.compute_configuration.pram_mode = "submission";  # usually ENV["PRAM_MODE"]

# +
# Could explore environment
# execute_command(controller, agent, "MoveAhead")  # MoveAhead, RotateRight, ...
# print_mcs_gt_pose(agent, -1)
# -

# ## Initialize GLRenderer Environment

# +
room_height_bounds = (-4.0, 4.0)
room_width_bounds = (-8.0, 8.0)

room_bounds_uniform_params = [room_width_bounds[1] room_width_bounds[2];room_height_bounds[1] room_height_bounds[2]]

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

room_cloud_1 = hcat(room_cloud_1...)
room_cloud_2 = hcat(room_cloud_2...)
room_cloud_3 = hcat(room_cloud_3...)
room_cloud_4 = hcat(room_cloud_4...)

# Need to mesh walls in order to render them
v1, n1, f1 = GL.mesh_from_voxelized_cloud(GL.voxelize(room_cloud_1, resolution), resolution)
v2, n2, f2 = GL.mesh_from_voxelized_cloud(GL.voxelize(room_cloud_2, resolution), resolution)
v3, n3, f3 = GL.mesh_from_voxelized_cloud(GL.voxelize(room_cloud_3, resolution), resolution)
v4, n4, f4 = GL.mesh_from_voxelized_cloud(GL.voxelize(room_cloud_4, resolution), resolution)

room_cloud = hcat(room_cloud_1, room_cloud_2, room_cloud_3, room_cloud_4)

PL.scatter(room_cloud_1[1,:], room_cloud_1[3,:], label="")
PL.scatter!(room_cloud_2[1,:], room_cloud_2[3,:], label="")
PL.scatter!(room_cloud_3[1,:], room_cloud_3[3,:], label="")
PL.scatter!(room_cloud_4[1,:], room_cloud_4[3,:], label="")

# PL.scatter!(v2[:,1], v2[:,3], label="")
# PL.scatter!(v3[:,1], v3[:,3], label="")
# PL.scatter!(v4[:,1], v4[:,3], label="")

# +
# MCS camera intrinsics
camera_intrinsics_mcs = Geometry.CameraIntrinsics(
    600, 400,
    514.2991467983065, 514.2991467983065,
    300.0, 200.0,
    0.1, 25.0
)

# Convert camera intrinsics to match downsampling
camera_intrinsics = convert_camera_intrinsics(camera_intrinsics_mcs, 40)

renderer = GL.setup_renderer(camera_intrinsics, GL.RGBBasicMode())

# Add Objects
GL.load_object!(renderer, v1, n1, f1)
GL.load_object!(renderer, v2, n2, f2)
GL.load_object!(renderer, v3, n3, f3)
GL.load_object!(renderer, v4, n4, f4)

# Add lighting
renderer.gl_instance.lightpos = [0, 0, 0]

# Camera Pose
cam_pose = P.Pose(zeros(3), R.RotY(-pi/4+ 0.0))

# Define Wall Colors
# wall_colors = [I.colorant"red",I.colorant"green",I.colorant"blue",I.colorant"yellow"]
wall_colors = [I.colorant"blue", I.colorant"blue", I.colorant"blue", I.colorant"blue"]
# -

# Note: `depth` until here has size `(600, 600)`, but in next cell will be `(1, 14)` - see first line of `camera_intrinsics` parameters.

# +
@time rgb, depth = GL.gl_render(
    renderer, 
    [1,2,3,4], [P.IDENTITY_POSE, P.IDENTITY_POSE, P.IDENTITY_POSE, P.IDENTITY_POSE],
    wall_colors, 
    cam_pose
)
PL.heatmap(depth)

img = I.colorview(I.RGB,permutedims(rgb,(3,1,2))[1:3,:,:])
color = map(argmin, eachcol(vcat([I.colordiff.(img, c) for c in wall_colors]...)))

@show color 
img
# -

# Rotate in MCS and collect rows.

# Rows is collection of one row per frame containing just walls in 360° rotation
# SIDE EFFECT: THIS TAKES STEPS IN THE ENVIRONMENT
rows = collect_rows(agent, intrinsics, 36);

# +
# Currently, we focus on a single time step for reorientation
time_step_to_analyze = 8

# Show range of rows - <rows, from, to> [if from==to, show single row]
print_rows(rows, time_step_to_analyze, time_step_to_analyze)
# -

# Compute depth sense and RGB sense, colors for plotting, corners and corner indices
sense, sense_rgb, color_tuple_vector, corners, corner_indices = sense_environment(rows, time_step_to_analyze)

# Compute cloud and plot sense with corners and averaged color segments
cloud = GL.flatten_point_cloud(
            GL.depth_image_to_point_cloud(
                reshape(sense, (camera_intrinsics.height, camera_intrinsics.width)
            ),
            camera_intrinsics)
        )
plt.scatter(cloud[1, :], cloud[3, :], c=color_tuple_vector, marker="o")

# We could generate ground truth and render it:
# ```julia
# cov = Matrix{Float64}(LinearAlgebra.I, camera_intrinsics.width, camera_intrinsics.width) * 0.01;
# tr_gt, w = Gen.generate(slam_multi_timestep, (5, nothing, room_bounds_uniform_params, wall_colors, cov,));
# viz_trace(tr_gt, 1, camera_intrinsics)
# ```

# # Corner Detection

# Define room corners.

corner_1 = P.Pose([room_width_bounds[1], 0.0, room_height_bounds[1]], R.RotY(-pi/4))
corner_2 = P.Pose([room_width_bounds[1], 0.0, room_height_bounds[2]], R.RotY(pi/4))
corner_3 = P.Pose([room_width_bounds[2], 0.0, room_height_bounds[1]], R.RotY(pi+pi/4))
corner_4 = P.Pose([room_width_bounds[2], 0.0, room_height_bounds[2]], R.RotY(pi-pi/4))
gt_corners = [corner_1, corner_2, corner_3, corner_4]

# Plot individual corner(s) in observation. Note: Corners one index away from extrema are ignored.

# +
cloud = GL.flatten_point_cloud(GL.depth_image_to_point_cloud(
                reshape(sense, (camera_intrinsics.height, camera_intrinsics.width)),
        camera_intrinsics))

PL.scatter(cloud[1, :], cloud[3, :], label="")
for c in corners
    viz_corner(c)
end
PL.plot!(xlim=(-10, 10), ylim=(-10, 10), aspect_ratio=:equal, label="")
# -

# Invert camera-coordinate-corner relative to ground truth corner.

# +
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
    # Could create pose like this: PoseComposition.Pose([0.0, 0.75, 0.0], R.RotXYZ(0.0, 0.0, 0.0))
    viz_pose(p)
end
for c in gt_corners
   viz_corner(c) 
end
PL.plot!()
# -

# # Inference

# - Both `sense_rgb` and `get_rgb(tr_gt, 1)` are 3-dim. float array (`Array{Float64, 3}`) of size `(1, 15, 4)`
# - Both `sense` and `get_depth(tr_gt, 1)` are `Matrix` (`Array{Float64, 2}`) of size `(1, 15)`

# +
# Blank out color (for now)
sense_rgb[:] .= 0.0

# TODO 1 is the number of time steps and becomes 36 or so when we rotate
cov = Matrix{Float64}(LinearAlgebra.I, camera_intrinsics.width, camera_intrinsics.width) * 0.01;

if length(poses) > 0
@time pf_state = PF.pf_initialize(slam_multi_timestep,
        (1, nothing, room_bounds_uniform_params, wall_colors, cov,),
    Gen.choicemap(sense_depth_addr(t) => sense[:], sense_rgb_addr(t) => sense_rgb),
    pose_mixture_proposal, (nothing, poses, 1, [1.0 0.0;0.0 1.0] * 0.05, deg2rad(2.0)),
    2000);
else
    pf_state = PF.pf_initialize(slam_multi_timestep,
        (1, nothing, room_bounds_uniform_params, wall_colors, cov,),
    Gen.choicemap(sense_depth_addr(t) => sense[:], sense_rgb_addr(t) => sense_rgb),
    4000);
end
# -

order = sortperm(pf_state.log_weights, rev=true)
best_tr = pf_state.traces[order[1]]
@show Gen.get_score(best_tr)
@show Gen.project(best_tr, Gen.select(pose_addr(1)))
pose = best_tr[pose_addr(1)]
@show pose
viz_trace(best_tr, [1], camera_intrinsics)
p0 = PL.plot!(ticks=nothing, border=nothing, xaxis=:off, yaxis=:off, xlim=(-10,10), ylim=(-6,6), title="Best Trace")

# +
PL.plot()
viz_env(;wall_colors=wall_colors)

z = Gen.logsumexp(pf_state.log_weights)
log_weights = pf_state.log_weights .- z
weights = exp.(log_weights)
for i in 1:length(pf_state.traces)
    tr = pf_state.traces[i]
    p = get_pose(tr, 1)
    lambda = 0.001
    viz_pose(p, alpha=lambda + (1.0-lambda) * weights[i])
end
# Inferred Pose Posterior
p1 = PL.plot!(ticks=nothing, border=nothing, xaxis=:off,yaxis=:off, xlim=(-9,9), ylim=(-6,6), title="Inferred")
p1
# +
agent_pose_gt_mcs_pos, agent_pose_gt_mcs_rot = print_mcs_gt_pose(agent, time_step_to_analyze)

PL.plot()
viz_env(;wall_colors=wall_colors)

pose2 = P.Pose([agent_pose_gt_mcs_pos["x"], agent_pose_gt_mcs_pos["y"], agent_pose_gt_mcs_pos["z"]],
    R.RotY(deg2rad(agent_pose_gt_mcs_rot)))
println("Extracted pose: $(pose)")

# Could render rays
cloud = GL.flatten_point_cloud(
            GL.depth_image_to_point_cloud(
                reshape(sense, (camera_intrinsics.height, camera_intrinsics.width)
            ),
            camera_intrinsics)
        )
cloud = GL.move_points_to_frame_b(cloud, pose2)
for i in 1:size(cloud)[2]
   PL.plot!([pose2.pos[1], cloud[1, i]], [pose2.pos[3], cloud[3,i]],
            color=I.colorant"grey90",
            linewidth=2,
            label=false) 
end

viz_pose(pose2)
orientation = agent_pose_gt_mcs_rot  # rad2deg(rotation_angle(pose.orientation))
title_str = "GT ($(round(pose.pos[1]; sigdigits=2)), $(round(pose.pos[3]; sigdigits=2))), $(round(orientation; sigdigits=2))°"
p2 = PL.plot!(ticks=nothing, border=nothing, xaxis=:off, yaxis=:off, xlim=(-9,9), ylim=(-6,6), title=title_str)
p2
# -

PL.plot(p1, p0, p2, layout=(3, 1), legend=false, aspect_ratio=:equal, size=(700, 700))

# # Nothing below this should matter for now

poses = []
for c in corners
    for c2 in gt_corners
        p = c2 * inv(c)
        push!(poses, p)
    end
end
@show length(poses)

# +
T = 5
constraints = 
    Gen.choicemap(
        # pose_addr(1)=>P.Pose([-2.0, 0.0, -3.0], R.RotY(pi))  # Look straight at wall
    pose_addr(1)=>P.Pose([-2.0, 0.0, -3.0], R.RotY(pi/2))  # Look at corner
    )
for t in 2:T
    constraints[pose_addr(t)] = constraints[pose_addr(t-1)] * P.Pose(zeros(3), R.RotY(deg2rad(25.0)))
end

wall_colors = [I.colorant"firebrick1",I.colorant"goldenrod1",I.colorant"green3",I.colorant"dodgerblue2"]
wall_colors = [I.colorant"firebrick1" for _ in 1:4]
cov = Matrix{Float64}(LinearAlgebra.I, camera_intrinsics.width, camera_intrinsics.width) * 0.3;
tr_gt, w = Gen.generate(slam_multi_timestep, (T, nothing, room_bounds_uniform_params,wall_colors,cov,), constraints);
@show Gen.get_score(tr_gt)
viz_trace(tr_gt, [1], camera_intrinsics)

# +
t=1  # Injected
corners = get_corners(get_depth(tr_gt, 1), false)  # Corners signature

poses = []
for c in corners
    for c2 in gt_corners
        p = c2 * inv(c)
        push!(poses, p)
    end
end
@show length(poses)

if length(poses) > 0
@time pf_state = PF.pf_initialize(slam_multi_timestep,
    (1,Gen.get_args(tr_gt)[2:end]...),
    Gen.choicemap(sense_depth_addr(t) => get_depth(tr_gt,t)[:], sense_rgb_addr(t) => get_rgb(tr_gt,t)),
    pose_mixture_proposal, (nothing, poses, 1,[1.0 0.0;0.0 1.0] * 0.05, deg2rad(2.0)),
    2000);

else
    pf_state = PF.pf_initialize(slam_multi_timestep,
        (1,Gen.get_args(tr_gt)[2:end]...),
    Gen.choicemap(sense_depth_addr(t) => get_depth(tr_gt,t)[:], sense_rgb_addr(t) => get_rgb(tr_gt,t)),
    4000);
end

# PF.pf_move_accept!(pf_state, Gen.metropolis_hastings, (position_drift_proposal, 
#         (t,[1.0 0.0;0.0 1.0] * 0.1)), 100);
# PF.pf_move_accept!(pf_state, Gen.metropolis_hastings, (head_direction_drift_proposal, 
#         (t,deg2rad(1.0))), 100);
# PF.pf_move_accept!(pf_state, Gen.metropolis_hastings, (joint_pose_drift_proposal, 
#         (t,[1.0 0.0;0.0 1.0] * 0.1, deg2rad(1.0))), 100);
# -

order = sortperm(pf_state.log_weights,rev=true)
best_tr = pf_state.traces[order[1]]
@show Gen.get_score(best_tr)
@show Gen.project(best_tr, Gen.select(pose_addr(1)))
pose = best_tr[pose_addr(1)]
@show pose
viz_trace(best_tr, [1], camera_intrinsics)
# viz_pose(get_pose(best_tr,1))

z = Gen.logsumexp(pf_state.log_weights)
log_weights = pf_state.log_weights .- z
weights = exp.(log_weights)
PL.plot(weights)

# +
PL.plot()
viz_env(;wall_colors=wall_colors)

z = Gen.logsumexp(pf_state.log_weights)
log_weights = pf_state.log_weights .- z
weights = exp.(log_weights)
for i in 1:length(pf_state.traces)
    tr = pf_state.traces[i]
    p = get_pose(tr, 1)
    lambda = 0.001
    viz_pose(p,alpha= lambda + (1.0-lambda) * weights[i])
end
p = PL.plot!(ticks=nothing, border=nothing, xaxis=:off,yaxis=:off, xlim=(-9,9), ylim=(-6,6),title="Inferred Pose Posterior")
p

# +
PL.plot()
viz_env(;wall_colors=wall_colors)

tr = tr_gt
pose = get_pose(tr,t)
depth = get_depth(tr,t)
sense = get_sense_depth(tr,t)

cloud = GL.flatten_point_cloud(GL.depth_image_to_point_cloud(depth, camera_intrinsics))
cloud = GL.move_points_to_frame_b(cloud, pose)
for i in 1:size(cloud)[2]
   PL.plot!([pose.pos[1], cloud[1,i]], [pose.pos[3], cloud[3,i]],
            color=I.colorant"grey90",
            linewidth=2,
            label=false) 
end


viz_pose(pose)
p2 = PL.plot!(ticks=nothing, border=nothing, xaxis=:off,yaxis=:off, xlim=(-9,9),ylim=(-6,6),title="Ground Truth")
p2
# -

PL.plot(p2, p, size=(1200,400))
PL.savefig("all.png")











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
# reducing over empty collection error happens in next line
if length(corners) != 0 && length(poses) != 0
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
end

for t in 2:5
    PF.pf_update!(pf_state,
                  (t, Gen.get_args(tr_gt)[2:end]...),
                  (Gen.UnknownChange(),[Gen.NoChange() for _ in 1:(length(Gen.get_args(tr_gt))-1)]...),
    Gen.choicemap(sense_depth_addr(t) => get_depth(tr_gt,t)[:], sense_rgb_addr(t) => get_rgb(tr_gt,t)),
    );

    
    # Geometrically computed pose proposal
    corners = get_corners(get_depth(tr_gt,t))
    poses = []
    for c in corners
        for c2 in gt_corners
            p = c2 * inv(c)
            push!(poses, p)
        end
    end
    if length(corners) == 0 || length(poses) == 0
        continue
    end
    println("Lenght corners: $(length(corners)) length poses: $(length(poses))")
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
best_tr = pf_state.traces[order[15]]
viz_trace(best_tr, [1,2,3,4,5], camera_intrinsics)
# viz_pose(get_pose(best_tr,1))

z = Gen.logsumexp(pf_state.log_weights)
log_weights = pf_state.log_weights .- z
weights = exp.(log_weights)
@show sum(weights)
weights

for i in 1:40:600
    println(i)
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

