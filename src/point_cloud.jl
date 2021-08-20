function depth_image_to_point_cloud(
        depth_map::Matrix, intrinsics::CameraIntrinsics;
        stride_x=1, stride_y=1)

    height = intrinsics.height
    width = intrinsics.width
    cx = intrinsics.cx
    cy = intrinsics.cy
    fx = intrinsics.fx
    fy = intrinsics.fy

    if size(depth_map) != (height, width)
        error("expected depth map that is (height x width) matrix")
    end

    x = depth_map[1:stride_y:end, 1:stride_x:end] .* (collect(1:stride_x:width) .- cx .- 0.5)' ./ fx
    y = depth_map[1:stride_y:end, 1:stride_x:end] .* (collect(1:stride_y:height) .- cy .- 0.5) ./ fy
    z = depth_map[1:stride_y:end, 1:stride_x:end]
    new_height, new_width = size(x)
    point_cloud = Array{Float64}(undef, new_height, new_width, 3)
    point_cloud[:,:,1] = x
    point_cloud[:,:,2] = y
    point_cloud[:,:,3] = z
    return point_cloud
end

function flatten_point_cloud(point_cloud::Array{Float64,3})
    (height, width, k) = size(point_cloud)
    if k != 3
        error("expected (height x width x 3) array")
    end
    flat_point_cloud = reshape(permutedims(point_cloud, (3, 1, 2)), (3, width * height))
    @assert size(flat_point_cloud) == (3, width * height)
    return flat_point_cloud
end

function get_points_in_frame_b(points_in_frame_a::Matrix{<:Real}, b_relative_to_a::Pose)
    if size(points_in_frame_a)[1] != 3
        error("expected an 3 x n matrix")
    end
    n = size(points_in_frame_a[2])
    return b_relative_to_a.orientation' * (points_in_frame_a .- b_relative_to_a.pos)
end

function move_points_to_frame_b(points_in_frame_a::Matrix{<:Real}, b_relative_to_a::Pose)
    if size(points_in_frame_a)[1] != 3
        error("expected an 3 x n matrix")
    end
    n = size(points_in_frame_a[2])
    return b_relative_to_a.orientation * points_in_frame_a .+ b_relative_to_a.pos
end

export flatten_point_cloud, depth_image_to_point_cloud, get_points_in_frame_b, move_points_to_frame_b
