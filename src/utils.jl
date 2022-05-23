function pose_to_matrix(pose::Pose)::Matrix{Float32}
    R = Matrix{Float32}(pose.orientation)
    mat = zeros(Float32,4,4)
    mat[end,end] = 1
    mat[1:3,1:3] = R
    mat[1:3,4] = pose.pos
    return mat
end


function I4(t)
    x = zeros(t, 4,4)
    x[1,1] = 1.0
    x[2,2] = 1.0
    x[3,3] = 1.0
    x[4,4] = 1.0
    x
end

function compute_projection_matrix(fx, fy, cx, cy, near, far, skew=0f0)
    proj = I4(Float32)
    proj[1, 1] = fx
    proj[2, 2] = fy
    proj[1, 2] = skew
    proj[1, 3] = -cx
    proj[2, 3] = -cy
    proj[3, 3] = near + far
    proj[3, 4] = near * far
    proj[4, 4] = 0.0f0
    proj[4, 3] = -1f0
    return proj
end



function compute_ortho_matrix(left, right, bottom, top, near, far)
    ortho = I4(Float32)
    ortho[1, 1] = 2f0 / (right-left)
    ortho[2, 2] = 2f0 / (top-bottom)
    ortho[3, 3] = - 2f0 / (far - near)
    ortho[1, 4] = - (right + left) / (right - left)
    ortho[2, 4] = - (top + bottom) / (top - bottom)
    ortho[3, 4] = - (far + near) / (far - near)
    return ortho
end

function get_perspective_matrix(width, height, fx, fy, cx, cy, near, far)
    # (height-cy) is used instead of cy because of the difference between
    # image coordinate systems between OpenCV and OpenGL. In the former,
    # the origin is at the top-left of the image while in the latter the
    # origin is at the bottom-left.
    proj_matrix = compute_projection_matrix(
            fx, fy, cx, (height-cy),
            near, far, 0.f0)
    ndc_matrix = compute_ortho_matrix(0, width, 0, height, near, far)
    ndc_matrix * proj_matrix
end


function CameraIntrinsics_from_fov_aspect(width, height, fov_y, aspect_ratio, near, far)
    # Camera principal point is the center of the image.
    cx, cy = width / 2.0, height / 2.0

    # Vertical field of view is given.
    fov_y = deg2rad(fov_y)
    # Convert field of view to distance to scale by aspect ratio and
    # convert back to radians to recover the horizontal field of view.
    fov_x = 2 * atan(aspect_ratio * tan(fov_y / 2.0))

    # Use the following relation to recover the focal length:
    #   FOV = 2 * atan( (0.5 * IMAGE_PLANE_SIZE) / FOCAL_LENGTH )
    fx = cx / tan(fov_x / 2.0)
    fy = cy / tan(fov_y / 2.0)


    CameraIntrinsics(width, height,
        fx, fy, cx, cy,
        near, far)
end


function scale_down_camera(camera, factor)
    camera_modified = CameraIntrinsics(
        width=round(Int,camera.width/factor), height=round(Int,camera.height/factor),
        fx=camera.fx/factor, fy=camera.fy/factor, cx=camera.cx/factor, cy=camera.cy/factor,
        near=camera.near, far=camera.far)
    camera_modified
end

