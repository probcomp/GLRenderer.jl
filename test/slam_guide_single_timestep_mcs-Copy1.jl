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
include("mcs_common.jl")

# +
room_height_bounds = (-4.0, 4.0)
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


room_cloud_1 = hcat(room_cloud_1...)
room_cloud_2 = hcat(room_cloud_2...)
room_cloud_3 = hcat(room_cloud_3...)
room_cloud_4 = hcat(room_cloud_4...)

v1,n1,f1 = GL.mesh_from_voxelized_cloud(GL.voxelize(room_cloud_1, resolution), resolution)
v2,n2,f2 = GL.mesh_from_voxelized_cloud(GL.voxelize(room_cloud_2, resolution), resolution)
v3,n3,f3 = GL.mesh_from_voxelized_cloud(GL.voxelize(room_cloud_3, resolution), resolution)
v4,n4,f4 = GL.mesh_from_voxelized_cloud(GL.voxelize(room_cloud_4, resolution), resolution)

room_cloud = hcat(room_cloud_1, room_cloud_2, room_cloud_3, room_cloud_4)

PL.scatter(room_cloud_1[1,:], room_cloud_1[3,:],label="")
PL.scatter!(room_cloud_2[1,:], room_cloud_2[3,:],label="")
PL.scatter!(room_cloud_3[1,:], room_cloud_3[3,:],label="")
PL.scatter!(room_cloud_4[1,:], room_cloud_4[3,:],label="")

# PL.scatter!(v2[:,1],v2[:,3],label="")
# PL.scatter!(v3[:,1],v3[:,3],label="")
# PL.scatter!(v4[:,1],v4[:,3],label="")

# +
camera_intrinsics_mcs = Geometry.CameraIntrinsics(
    600, 400,
    514.2991467983065, 514.2991467983065,
    300.0, 200.0,
    0.1, 25.0
)
camera_intrinsics = convert_camera_intrinsics(camera_intrinsics_mcs, 40)

# renderer = GL.setup_renderer(Geometry.CameraIntrinsics(
#     600, 600,
#     300.0, 300.0,
#     300.0, 300.0,
#     0.1, 60.0
# ), GL.RGBBasicMode())

renderer = GL.setup_renderer(camera_intrinsics, GL.RGBBasicMode())

GL.load_object!(renderer, v1, n1, f1)
GL.load_object!(renderer, v2, n2, f2)
GL.load_object!(renderer, v3, n3, f3)
GL.load_object!(renderer, v4, n4, f4)
renderer.gl_instance.lightpos = [0, 0, 0]

rgb, depth = GL.gl_render(
    renderer, 
    [1,2,3,4],
    [P.IDENTITY_POSE, P.IDENTITY_POSE, P.IDENTITY_POSE, P.IDENTITY_POSE],
    [I.colorant"red", I.colorant"green", I.colorant"blue", I.colorant"yellow"],
    P.Pose([0.0, -10.0, 0.0], R.RotX(-pi/2))
)
# PL.heatmap(depth, aspect_ratio=:equal)
I.colorview(I.RGBA, permutedims(rgb,(3,1,2)))
# -

camera_intrinsics

# Note: `depth` until here has size `(600, 600)`, but in next cell will be `(1, 14)` - see first line of `camera_intrinsics` parameters.

# +
# camera_intrinsics = Geometry.CameraIntrinsics(
#     14, 1,
#     5.0, 1.0,
#     7.0, 0.5,
#     0.1, 20.0
# )
# renderer = GL.setup_renderer(camera_intrinsics, GL.RGBBasicMode())

renderer = GL.setup_renderer(camera_intrinsics, GL.RGBBasicMode())
GL.load_object!(renderer, v1, n1, f1)
GL.load_object!(renderer, v2, n2, f2)
GL.load_object!(renderer, v3, n3, f3)
GL.load_object!(renderer, v4, n4, f4)
renderer.gl_instance.lightpos = [0, 0, 0]

cam_pose = P.Pose(zeros(3), R.RotY(-pi/4+ 0.0))
wall_colors = [I.colorant"red",I.colorant"green",I.colorant"blue",I.colorant"yellow"]
# wall_colors = [I.colorant"red",I.colorant"red",I.colorant"red",I.colorant"red"]
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

function viz_env(;wall_colors=nothing)
    width = 5
    if isnothing(wall_colors)
        c = I.colorant"blue"
        PL.plot!([room_width_bounds[1],room_width_bounds[1]],
                [room_height_bounds[1],room_height_bounds[2]], linewidth=width,
            color=c,label=false)
        PL.plot!([room_width_bounds[2],room_width_bounds[2]],
                [room_height_bounds[1],room_height_bounds[2]], linewidth=width,
            color=c,label=false)
        PL.plot!([room_width_bounds[1],room_width_bounds[2]],
                [room_height_bounds[1],room_height_bounds[1]], linewidth=width,
            color=c,label=false)
        PL.plot!([room_width_bounds[1],room_width_bounds[2]],
                [room_height_bounds[2],room_height_bounds[2]], linewidth=width,
            color=c,label=false)
    else
        PL.plot!([room_width_bounds[1],room_width_bounds[1]],
                [room_height_bounds[1],room_height_bounds[2]], linewidth=width,
            color=wall_colors[1],label=false)
        PL.plot!([room_width_bounds[2],room_width_bounds[2]],
                [room_height_bounds[1],room_height_bounds[2]], linewidth=width,
            color=wall_colors[2],label=false)
        PL.plot!([room_width_bounds[1],room_width_bounds[2]],
                [room_height_bounds[1],room_height_bounds[1]], linewidth=width,
            color=wall_colors[3],label=false)
        PL.plot!([room_width_bounds[1],room_width_bounds[2]],
                [room_height_bounds[2],room_height_bounds[2]], linewidth=width,
            color=wall_colors[4],label=false)
        
    end
end

function viz_pose(pose; alpha=1.0)
    pos = pose.pos
    PL.scatter!([pos[1]], [pos[3]],color=:red,alpha=alpha,label=false)

    direction = pose.orientation * [0.0, 0.0, 1.0]
    PL.plot!([pos[1], pos[1]+direction[1]],
             [pos[3], pos[3]+direction[3]],
             arrow=true,color=:black, linewidth=2, alpha=alpha,label=false)
end

function viz_obs(tr, t, camera_intrinsics)
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
    
function viz_trace(tr, ts, camera_intrinsics)
    PL.plot()
    T = Gen.get_args(tr)[1]
    viz_env()
    for t in ts
        viz_pose(get_pose(tr,t))
        viz_obs(tr, t, camera_intrinsics)
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

cov = Matrix{Float64}(LinearAlgebra.I, camera_intrinsics.width, camera_intrinsics.width) * 0.01;
tr_gt, w = Gen.generate(slam_multi_timestep, (5, nothing, room_bounds_uniform_params, wall_colors, cov,));
viz_trace(tr_gt, 1, camera_intrinsics)
# -

# # Corner Detection

# +
corner_1 = P.Pose([room_width_bounds[1], 0.0, room_height_bounds[1]], R.RotY(-pi/4))
corner_2 = P.Pose([room_width_bounds[1], 0.0, room_height_bounds[2]], R.RotY(pi/4))
corner_3 = P.Pose([room_width_bounds[2], 0.0, room_height_bounds[1]], R.RotY(pi+pi/4))
corner_4 = P.Pose([room_width_bounds[2], 0.0, room_height_bounds[2]], R.RotY(pi-pi/4))

# Plot corners
PL.plot()
viz_env()
gt_corners = [corner_1,corner_2,corner_3,corner_4]
for c in gt_corners
    viz_corner(c)
end
# TODO: Write corner detection function that finds corners given an occupancy grid (aka point cloud) of the room.
PL.plot!()

# +
sense = get_depth(tr_gt,1)
corners = get_corners(sense, false)
cloud = GL.flatten_point_cloud(GL.depth_image_to_point_cloud(
        reshape(sense,(camera_intrinsics.height, camera_intrinsics.width)), camera_intrinsics))

PL.scatter(cloud[1,:], cloud[3,:], label="")
for c in corners
    viz_corner(c)
end
PL.plot!(xlim=(-10,10),ylim=(-10,10), aspect_ratio=:equal, label="")

# +
sense = get_depth(tr_gt,1)

corners = get_corners(sense, false)
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

# Load MCS

# +
scene_path=joinpath(pwd(), "data/rectangular_colored_walls_empty.json")
mcs_executable_path=ENV["MCS_EXECUTABLE_PATH"]
mcs_config_path=ENV["MCS_CONFIG_FILE_PATH"]

# config_data, status = mcs.load_scene_json_file(scene_path)

config_data = Dict{Any, Any}(
    "name" => "template_individually_colored_walls",
    "version" => 2,
    "ceilingMaterial" => "AI2-THOR/Materials/Walls/Drywall",
    "floorMaterial" => "AI2-THOR/Materials/Fabrics/Carpet4",
    "roomMaterials" => Dict{Any, Any}(
        "left" => "AI2-THOR/Materials/Walls/DrywallGreen",
        "front" => "AI2-THOR/Materials/Walls/RedDrywall",
        "right" => "AI2-THOR/Materials/Walls/EggshellDrywall",
        "back" => "AI2-THOR/Materials/Walls/DrywallOrange"),
    "roomDimensions" => Dict{Any, Any}("x"=>16, "z"=>8, "y"=>4),
    "performerStart" => Dict{Any, Any}(
        "position" => Dict{Any, Any}("x" => 0, "z" => 0),
        "rotation" => Dict{Any, Any}("x" => 0, "y" => 0)),
    "objects" => []
)

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
agent.config.compute_configuration.pram_mode = "submission";  # usually ENV["PRAM_MODE"]
# -



# Now record `T` rows in total, each row valid (i.e. only contains wall segments) and closest to row 200 (expected to usually be exactly 200).
#
# Current issues:
# - Need to find RGB values along with 3D coordinates and bin them into up to 4 color bins. Should do corner detection first and then bin colors on each line segment (between corners) and unify them if they are close to each other.

# +
rois = collect_rows(agent, intrinsics, 36)

# for entry in rois
#     plt.scatter(entry[1][1,:], entry[1][3,:], c="gray", marker="o")  # rgb[:, 1:3]
# end
# -

print_rows(rois, 2, 8)

# +
# Marked for deletion - cell should be obsolete

# time_step = 6

# # Subsample Indices
# indices = collect(1:40:600)

# roi_subsampled = [rois[time_step][3][i] for i in indices]

# # println(size(roi_subsampled))

# # Could subsample color the same way:
# rgb = rois[time_step][2]'
# rgb_subsampled = [(rgb[1, i]/255, rgb[2, i]/255, rgb[3, i]/255) for i in 1:40:600]

# corners = get_corners(roi_subsampled)

# # Could plot depth snippet:
# # Plots.plot(roi_subsampled; aspect_ratio=:equal)
# -

# Side validation: Convert subsampled depth values to cloud and plot them out.

# +
# corners, corner_indices = get_corners(sense, true)


# -



# After this step
# - Both `sense_rgb` and `get_rgb(tr_gt, 1)` are 3-dim. float array (`Array{Float64, 3}`) of size `(1, 15, 4)`
# - Both `sense` and `get_depth(tr_gt, 1)` are `Matrix` (`Array{Float64, 2}`) of size `(1, 15)`

# +
t=1  # Injected

# FIXME If this value is too close (1 step?) to edge, the corner detector fails
sense, sense_rgb, _, _, _ = sense_environment(rois, 6)
corners = get_corners(sense[:], false)

cloud = GL.flatten_point_cloud(GL.depth_image_to_point_cloud(reshape(sense,
            (camera_intrinsics.height, camera_intrinsics.width)), camera_intrinsics))
# plt.scatter(cloud[1,:], cloud[3,:], c=color_tuple_vector, marker="o")
# # corners = get_corners(get_depth(tr_gt, 1), false)  # Corners signature

poses = []
for c in corners
    for c2 in gt_corners
        p = c2 * inv(c)
        push!(poses, p)
    end
end
@show length(poses) 


# poses = []
# for c in corners
#     for c2 in gt_corners
#         p = c2 * inv(c)
#         push!(poses, p)
#     end
# end

PL.plot()
viz_env()
for p in poses
   viz_pose(p) 
end
for c in gt_corners
   viz_corner(c) 
end
PL.plot!()
# -

sense_rgb[:] .= 0.0

# +
(5, nothing, room_bounds_uniform_params, wall_colors, cov,)
if length(poses) > 0
@time pf_state = PF.pf_initialize(slam_multi_timestep,
        # TODO: remove dependence on tr_gt. since this is no longer a thing
        # instead write out the paramters yourself, which will allow you to set the wall_colors explicitly
        # for the first round of deubgging to make sure the "kidnapped" thing works , set sense_rgb to the same value everywhere and the wall colors to the same value everywhere

    (1,Gen.get_args(tr_gt)[2:end]...),
    Gen.choicemap(sense_depth_addr(t) => sense[:], sense_rgb_addr(t) => sense_rgb),
    pose_mixture_proposal, (nothing, poses, 1, [1.0 0.0;0.0 1.0] * 0.05, deg2rad(2.0)),
    2000);

else
    pf_state = PF.pf_initialize(slam_multi_timestep,
        (1,Gen.get_args(tr_gt)[2:end]...),
    Gen.choicemap(sense_depth_addr(t) => sense[:], sense_rgb_addr(t) => sense_rgb),
    4000);
end
# -

println(Gen.get_args(tr_gt)[2:end])

tr_gt

order = sortperm(pf_state.log_weights,rev=true)
best_tr = pf_state.traces[order[1]]
@show Gen.get_score(best_tr)
@show Gen.project(best_tr, Gen.select(pose_addr(1)))
pose = best_tr[pose_addr(1)]
@show pose
viz_trace(best_tr, [1], camera_intrinsics)

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
# -
println(agent.step_metadatas[end].position)










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

import LinearAlgebra
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

PL.plot(p2,p, size=(1200,400))
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

