"""
Converts to intrinsics downscaled by `factor` for slice of height 1.
"""
function convert_camera_intrinsics(ci::Geometry.CameraIntrinsics, factor)
    desired_width = ci.width / factor
    return Geometry.CameraIntrinsics(
    desired_width, 1,  # ci.height is 1 since intrinsincs for slice
    ci.fx / factor, ci.fy,
    desired_width / 2.0 .+ (1/2 - 1/factor * 1/2) , 0.5,  # ci.cy is 0.5 since height is 1
    ci.near, ci.far
)
end

function get_corners(sense, with_index::Bool)
    println("$(with_index)")
    c, cidx = get_corners(sense)
    if with_index
        return c, cidx
    else
        return c
    end
end

function get_corners(sense)
    cloud = GL.flatten_point_cloud(GL.depth_image_to_point_cloud(reshape(sense,(camera_intrinsics.height, camera_intrinsics.width)), camera_intrinsics))
    cloud = vcat(cloud[1,:]', cloud[3,:]')
    deltas = diff(cloud,dims=2)
    dirs = map(x->R.RotMatrix{2,Float32}(atan(x[2],x[1])), eachcol(deltas))
    angle_errors = [abs.(R.rotation_angle(inv(dirs[i])*dirs[i+1])) for i in 1:length(dirs)-1]
    println("Angle errors: $(angle_errors)")
    spikes = findall(angle_errors .> deg2rad(45.0))
    spikes .+= 1
    corners = []
    corners_idx = []
    
    println("Spikes: $(spikes)")
    
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
                push!(corners_idx, s)
                push!(corners, corner_pose)
        end
    end
    corners, corners_idx
end

function find_scanlines(valid_index_mask)
    valid_rows = prod(valid_index_mask, dims=2)  # (400, 1) vector - pick index with value 1
    valid_indices = []
    for row_idx in 1:size(valid_rows)[1]
        if valid_rows[row_idx]
            push!(valid_indices, row_idx)
        end
    end
    return valid_indices
end

"""
Same as `get_scanline()`, but works on camera frame point cloud
without depending on global coordinates. Based on `detect_corner_from_camera_frame(agent)`.
"""
function get_scanline_from_camera_frame(agent::Agent, intrinsics::CameraIntrinsics;
        verbose = false)
    
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

    scanline_idx = find_scanlines(valid_indices_2d)
    
    # size(depth_map) = (400, 600) => size(cloud_array_2d) = (3, 400, 600)
    return cloud_array_2d, scanline_idx
end

"""
Picks scanline closest to row index `target_row` from list of valid scanlines as produced by `find_scanlines()`
or `get_scanline_from_camera_frame()` and returns both its row index and its distance.
"""
function find_closest_row(scanline_indices, target_row::Int = 200)
    min_distance = 99999
    row = -1
    for scanline_idx in scanline_indices
        current_distance = abs(scanline_idx - target_row)
        if current_distance < min_distance
            min_distance = current_distance
            row = scanline_idx
        end
    end
    
    return row, min_distance
end

function collect_rows(agent, intrinsics, step_count::Int = 10)
    # Initial position already in agent.step_metadatas => Only need to rotate T-1 steps
    rois = []
    for i in 1:step_count
        cloud_array_2d, scanline_indices = get_scanline_from_camera_frame(agent, intrinsics)
        scanline_idx, _ = find_closest_row(scanline_indices)
        println("Closest row to target index 200 at time step $(i): $(scanline_idx).")
        roi = cloud_array_2d[1:3, scanline_idx, :]

        rgb = np.array(agent.step_metadatas[end].image_list[end])[scanline_idx, :, 1:3]

        depth = np.array(agent.step_metadatas[end].depth_map_list[end])[scanline_idx, :]
        println("RGB size $(size(rgb)), depth size $(size(depth))")

        push!(rois, (roi, rgb, depth, scanline_idx))
        
        if i<step_count  # Do not rotate in last step
            execute_command(controller, agent, "RotateRight")
        end
    end
    return rois
end

function print_rows(rois, step_count_start::Int = 1, step_count_end::Int = 10)
    for time_step in step_count_start:step_count_end
        threeD = rois[time_step][1]

        rgb = rois[time_step][2]'
        plot_colors = [(rgb[1, i]/255, rgb[2, i]/255, rgb[3, i]/255) for i in 1:600]  # 600 tuples

        # This would average all colors for a row, but we should only do that on line segments
    #     plot_color = map(round, sum(rgb, dims=2)/600)./255
    #     plot_colors = [tuple(plot_color...) for i ∈ 1:600]

        plt.scatter(threeD[1,:], threeD[3,:], c=plot_colors, marker="o")
    end
end

function sense_environment(rois, time_step::Int)
    indices = collect(1:40:600)
    # time_step = 8
    # (roi, rgb, depth, scanline_idx) -> value 2 is RGB, value 3 is depth

    # Depth
    depth_subsampled = [rois[time_step][3][i] for i in indices]
    sense = Matrix(depth_subsampled')

    corners, corner_indices = get_corners(sense, true)
#     cloud = GL.flatten_point_cloud(GL.depth_image_to_point_cloud(reshape(sense,
#                 (camera_intrinsics.height, camera_intrinsics.width)), camera_intrinsics))

    rgb = vcat(hcat([rois[time_step][2][i, :] for i ∈ 1:40:600]...))

    segment_start = 1
    segment_final = size(rgb)[2]
    rgb_segments = []
    sense_rgb_list = []

    for i ∈ corner_indices
        println("Segment from $(segment_start) to $(i)")
        segment_length = i - segment_start + 1

        r = sum(rgb[1:3, segment_start:i], dims=2) ./ segment_length ./ 255  # size (3, 1)
        println("r $(r) size $(size(r)) vcat $(vcat(r, 1.0))")
        # repeat([0.24593837535014007; 0.16302521008403362; 0.15294117647058825; 1.0], 1,2)
        push!(sense_rgb_list, repeat(vcat(r, 1.0), 1, segment_length))

        segment_start = i + 1

        # Push average color, but highlight corner itself; need vector of tuples for scatter plot
        # TODO Not clever - build r, then do tuple color conversion outside.
        push!(rgb_segments, [tuple(r...) for _ ∈ 1:segment_length-1])
        push!(rgb_segments, (1.0, 0.0, 0.0))

        println("---")
    end
    if segment_start <= segment_final
        println("Segment from $(segment_start) to $(segment_final)")
        segment_length = segment_final - segment_start + 1
        r = sum(rgb[1:3, segment_start:segment_final], dims=2) ./ segment_length ./ 255
        push!(sense_rgb_list, repeat(vcat(r, 1.0), 1, segment_length))

        push!(rgb_segments, [tuple(r...) for _ ∈ 1:segment_length])
        println("---")
    end
    sense_rgb = hcat(sense_rgb_list...)
    sense_rgb = reshape(sense_rgb, (1,size(sense_rgb')...))
    color_tuple_vector = vcat(rgb_segments...)

    return sense, sense_rgb, color_tuple_vector, corners, corner_indices
end

"""
Prints agents ground truth position and rotation.
"""
function print_mcs_gt_pose(agent::Agent, time_step)
    agent_pose_gt_mcs_pos = agent.step_metadatas[time_step].position
    agent_pose_gt_mcs_rot = agent.step_metadatas[time_step].rotation
    println("MCS Agent GT Position: $(agent_pose_gt_mcs_pos) Rotation: $(agent_pose_gt_mcs_rot)")
    return agent_pose_gt_mcs_pos, agent_pose_gt_mcs_rot
end

"""
Generates scene definition based on provided parameters.

# Example
`config_data = generate_scene()`
"""
function generate_scene(;agent_pos_x = 0, agent_pos_z = 0,
    agent_rot_x = 0, agent_rot_y = 0,
    room_dim_x = 16, room_dim_y = 8, room_dim_z = 4,
    color1 = "AI2-THOR/Materials/Walls/DrywallGreen",
    color2 = "AI2-THOR/Materials/Walls/RedDrywall",
    color3 = "AI2-THOR/Materials/Walls/EggshellDrywall",
    color4 = "AI2-THOR/Materials/Walls/DrywallOrange")

return Dict{Any, Any}(
    "name" => "template_individually_colored_walls",
    "version" => 2,
    "ceilingMaterial" => "AI2-THOR/Materials/Walls/Drywall",
    "floorMaterial" => "AI2-THOR/Materials/Fabrics/Carpet4",
    "roomMaterials" => Dict{Any, Any}(
        "left" => color1,
        "front" => color1,
        "right" => color1,
        "back" => color1
    ),
    "roomDimensions" => Dict{Any, Any}("x" => room_dim_x, "z" => room_dim_z, "y" => room_dim_y),
    "performerStart" => Dict{Any, Any}(
        "position" => Dict{Any, Any}("x" => agent_pos_x, "z" => agent_pos_z),
        "rotation" => Dict{Any, Any}("x" => agent_rot_x, "y" => agent_rot_y)
    ),
    "objects" => []
)
end
