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

# Pose Proposal
mixture_of_pose_gaussians = Gen.HomogeneousMixture(pose_gaussian, [0, 2, 0])

@Gen.gen function pose_mixture_proposal(trace, poses, t, cov, var)
    n = length(poses)
    weights = ones(n) ./ n
    {pose_addr(t)} ~ mixture_of_pose_gaussians(
        weights, poses, cat([cov for _ in 1:n]..., dims=3), [var for _ in 1:n]
    )
end

# ----------

# Drift Moves

@Gen.gen function position_drift_proposal(trace, t,cov)
   {pose_addr(t)} ~ pose_gaussian(trace[pose_addr(t)], cov, 0.0001) 
end

@Gen.gen function head_direction_drift_proposal(trace, t,var)
   {pose_addr(t)} ~ pose_gaussian(trace[pose_addr(t)], [1.0 0.0;0.0 1.0] * 0.00001, var) 
end

@Gen.gen function joint_pose_drift_proposal(trace, t, cov, var)
   {pose_addr(t)} ~ pose_gaussian(trace[pose_addr(t)], cov, var) 
end

# ----------

@Gen.gen function slam_unfold_kernel(t, prev_data, room_bounds, wall_colors, cov)
    if t==1
        pose ~ pose_uniform(room_bounds[1,:], room_bounds[2,:])
    else
        pose ~ pose_gaussian(prev_data.pose, [1.0 0.0;0.0 1.0] * 0.1, deg2rad(20.0))
    end
    rgb, depth = GL.gl_render(renderer, 
     [1,2,3,4], [P.IDENTITY_POSE, P.IDENTITY_POSE,P.IDENTITY_POSE,P.IDENTITY_POSE],
    wall_colors, pose)
    
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

# Visualization Functions

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

# ----------
