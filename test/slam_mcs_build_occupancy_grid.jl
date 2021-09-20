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

using Colors

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
# scene_path = "/home/falk/mitibm/AI2Thor_MCS/dataset_eval4/rectangular_colored_walls_empty.json"
mcs_config_path="/home/falk/PycharmProjects/MCS/scripts/config_oracle.ini"  # ENV["MCS_CONFIG_FILE_PATH"]
# mcs_executable_path=ENV["MCS_EXECUTABLE_PATH"]
mcs_executable_path = "/home/falk/mitibm/AI2Thor_MCS/MCS-AI2-THOR-Unity-App-v0.4.4-linux/MCS-AI2-THOR-Unity-App-v0.4.4.x86_64"

controller = mcs.create_controller(unity_app_file_path=mcs_executable_path,
                                   config_file_path=mcs_config_path)

# +
config_data, status = mcs.load_scene_json_file(scene_path)
# Beware: If you do not define colors in generate_scene call when using it, walls will be monochrome
# config_data = generate_scene(agent_pos_x = -2.0, agent_pos_z = 1.0)

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
 using JSON
ENV["PRAM_MODE"] = "submission"

@show agent.information_level

# Get Bearings
rotate_360_simple(controller, agent)

# Adjust all poses to center and align the world frame
realign_world(agent)

# Add each observation thusfar to the occupancy grid
occupancy_grid_config = OccupancyGridConfig(SVector{3, Float64}([0.6, 0.6, 0.6]))
occupancy_grid = OccupancyGrid(occupancy_grid_config)

perception_updated_until = 0

objects_only_cloud, perception_updated_until = perception_initializiation(agent,
    occupancy_grid, perception_updated_until + 1, length(agent.poses))

# +
"""
Converts position `pos` in world coordinates into discrete occupancy grid coordinates.

Note that this returns `grid_x_col`, `grid_y_row`, but when indexing into the OG you might need to use
row/y index first, e.g. `map_occupied[grid_y_row, grid_x_col]`.

# Arguments
- `og` Occupancy grid
- `pos` Position in world coordinates
- `verbose` Optional, whether to print internal values to stdout for debugging.

# Example
```julia
pos = SVector{3, Float64}([-2.0, 0.0, 1.0])
# Could be MCS agent position:
# pos = SVector{3, Float64}([step_metadata.position["x"], step_metadata.position["y"], step_metadata.position["z"]])
grid_x_col, grid_y_row = convert_position_into_og_map(occupancy_grid, pos)
map_occupied[grid_y_row, grid_x_col] = 5000
```
"""
function convert_position_into_og_map(og::OccupancyGrid, pos::SVector{3, <:Real}; verbose::Bool = false)
    # Convert to grid coordinates, but these might be below 1 => shift into Matrix coords below
    agent_grid_coord = SVector{3,Int}(cloud_to_grid_coords(pos, og.config)...)
    
    # Preparation: determine Matrix shape as well as min and max values per axis
    min_x_col, _, min_y_row = og.min_grid
    max_x_col, _, max_y_row = og.max_grid
    shape = (max_y_row - min_y_row + 1, max_x_col - min_x_col + 1)  # No. rows, no. columns
    
    # Actual shift into Matrix coordinates
    grid_x_col = agent_grid_coord[1] - min_x_col + 1
    grid_y_row = shape[1] + 1 - (agent_grid_coord[3] - min_y_row + 1)
    
    if verbose
        println("Mins: $(min_x_col) $(min_y_row)")
        println("Maxs: $(max_x_col) $(max_y_row)")
        println("Shape: $(shape)")
        println("Agent grid coords: $(agent_grid_coord)")
        println("Final grid coords: $(grid_x_col) $(grid_y_row)")
    end
    
    return grid_x_col, grid_y_row
end

# # visualize_occupancy_map(occupancy_grid, pos)
# render_occupancy_grid(occupancy_grid)
free_dict_2d, occupied_dict_2d = project_3d_og_to_2d(occupancy_grid, occupancy_grid.config.height_slice, occupancy_grid.config.height_slice)
map_free, map_occupied, map_free_prob, map_occupied_prob, map_unobserved = generate_maps_2d_oriented(occupancy_grid, free_dict_2d, occupied_dict_2d);

# Use MCS position or...
pos = SVector{3, Float64}([step_metadata.position["x"], step_metadata.position["y"], step_metadata.position["z"]])
# ...manually defined coordinates:
pos = SVector{3, Float64}([-2.0, 0.0, 1.0])
println("Agent position: $(pos)")

grid_x_col, grid_y_row = convert_position_into_og_map(occupancy_grid, pos; verbose=true)

# Highlight agent position
map_occupied[grid_y_row, grid_x_col] = 5000

# display(map_occupied)
render_occupancy_grid_oriented(map_occupied)
# -

# Converting grid coordinates back to cloud coordinates (can lose information).

# grid_coords_to_cloud(grid_coords::Matrix, config::OccupancyGridConfig)
m = zeros(3, 1)
m[:,1] = [-3, 0, 2]  # [grid_x_col, 0, grid_y_row]
backproj = SVector{3, Float64}(grid_coords_to_cloud(m, occupancy_grid.config)...)
println("$(pos) => $(backproj)")

# Compute point cloud and occupancy grid.

# +
color_og = Dict()

# t = length(agent.step_metadatas)
for t in 1:length(agent.step_metadatas)
    step_metadata = agent.step_metadatas[t]
    pose = agent.poses[t]
    cloud = get_cloud(agent, step_metadata, pose, agent.intrinsics; t)
    grid_coords = cloud_to_grid_coords(cloud[1:3, :], occupancy_grid.config)
    # 3D points cloud[1:3, :]; RGB cloud[4:6, :], OG: grid_coords

    for i in 1:size(cloud)[2]
        if 2.0 < grid_coords[2, i] > 0.3
            x, z = grid_coords[1, i], grid_coords[3, i]
            rgb = cloud[4:6, i]
            if (x,z) ∉ keys(color_og)
                color_og[x,z] = rgb, 1
            else
                rgb_old, t = color_og[x,z]
                color_og[x,z] = (t-1)/t * rgb_old + 1/t * rgb, t+1
            end
        end
    end
end
# -

Plots.plot()
for (k, v) in color_og
    x, z = k
    c = RGB(round(v[1][1])/255, round(v[1][2])/255, round(v[1][3])/255)
    Plots.scatter!([x], [z], color=c)
end
Plots.plot!(xlim=(-15,15), ylim=(-10,10), legend=nothing)

color_og_keys_mat = [x[i] for x in [keys(color_og)...], i in 1:2]
xs = color_og_keys_mat[:, 1]  # All my xs
zs = color_og_keys_mat[:, 2]
min_x, max_x, min_z, max_z = min(xs...), max(xs...), min(zs...), max(zs...)
shape = (max_z - min_z + 1, max_x - min_x + 1)  # No. rows, no. columns
# Could show sorted color_og keys like this: print(sort!([keys(color_og)...]))

# Average color for four outer walls
# Beware: color_og is dictionary and hence index into with (x,y), not (y,x) as you would for array [row first]
top_row = [color_og[j, max_z][1] for j in min_x:max_x if (j, max_z) ∈ keys(color_og)]
bottom_row = [color_og[j, min_z][1] for j in min_x:max_x if (j, min_z) ∈ keys(color_og)]
left_row = [color_og[min_x,  j][1] for j in min_z:max_z if (min_x,  j) ∈ keys(color_og)]
right_row = [color_og[max_x,  j][1] for j in min_z:max_z if (max_x,  j) ∈ keys(color_og)]
top = Vector{Int64}(round.(sum(top_row)/length(top_row)))
bottom = Vector{Int64}(round.(sum(bottom_row)/length(bottom_row)))
left = Vector{Int64}(round.(sum(left_row)/length(left_row)))
right = Vector{Int64}(round.(sum(right_row)/length(right_row)))

RGB(bottom/255...)

# +
function record_equivalence!(dictionary, key, value)
    # Could inject lists if key not known, but here not necessary, since we instantiated list
    # if key ∉ keys(dictionary)
    #     dictionary[key] = [value]
    # push!(dictionary[key], value)
    for k in dictionary[key]
        push!(dictionary[k], value)
    end
end

function record_equivalences(equivalences, color_rgb_one, color_rgb_two,
        color_categorical_one, color_categorical_two; threshold = 10)
    if LinearAlgebra.norm(color_rgb_one - color_rgb_two) <= threshold
        record_equivalence!(equivalences, color_categorical_one, color_categorical_two)
        record_equivalence!(equivalences, color_categorical_two, color_categorical_one)
    end
end

# Pairwise compare outer wall colors and record equivalences
color_left, color_right, color_bottom, color_top = 1, 2, 3, 4  # categorical
equivalences = Dict(color_top => Set([color_top]), color_bottom => Set([color_bottom]),
    color_left => Set([color_left]), color_right => Set([color_right]))
record_equivalences(equivalences, top, left, color_top, color_left)
record_equivalences(equivalences, left, bottom, color_left, color_bottom)
record_equivalences(equivalences, bottom, right, color_bottom, color_right)
record_equivalences(equivalences, right, top, color_right, color_top)
record_equivalences(equivalences, top, bottom, color_top, color_bottom)
record_equivalences(equivalences, left, right, color_left, color_right)

# Color all walls with their lowest value color (unified coloring)
coloring = Dict()
for (k,v) in equivalences
    coloring[k] = minimum(v)
end
println("Equivalences")
println(equivalences)
println()
println("Coloring")
println(coloring)
# -

# Define wall_colors in order (left, right, bottom, top)
# all_colors = [I.colorant"red",I.colorant"green",I.colorant"blue",I.colorant"yellow"]
all_colors = [RGB(left/255...), RGB(right/255...), RGB(bottom/255...), RGB(top/255...)]
# println("left, right, bottom, top")
# println("$(left), $(right), $(bottom), $(top)")
wall_colors = []
for key ∈ 1:4  # Going in wall_colors order (left, right, bottom, top)
    all_colors_index = coloring[key]
    wall_color = all_colors[all_colors_index]
    println("$(key) -> $(all_colors_index) -> $(wall_color)")
    push!(wall_colors, wall_color)
end
# Convert any-typed vector to color vector so gl_render does not blow up
wall_colors = Vector{Color}(wall_colors)

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

# Define Wall Colors - done above instead for MCS
# wall_colors = [I.colorant"red",I.colorant"green",I.colorant"blue",I.colorant"yellow"]
# wall_colors = [I.colorant"blue", I.colorant"blue", I.colorant"blue", I.colorant"blue"]
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
t=1

# Blank out color (for now)
# sense_rgb[:] .= 0.0

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
