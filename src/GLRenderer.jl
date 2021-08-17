module GLRenderer

using PyCall
import PoseComposition: Pose
import Rotations
import Geometry: CameraIntrinsics
import Images: Color, RGBA
const glrenderer_python = PyNULL()

function __init__()
    copy!(glrenderer_python, pyimport("glrenderer"))
end


abstract type RenderMode end
struct DepthMode <: RenderMode end
struct RGBMode <: RenderMode end
struct TextureMode <: RenderMode end

mutable struct Renderer{T <: RenderMode}
    gl_instance
    camera_intrinsics::CameraIntrinsics
end

function setup_renderer(camera_intrinsics::CameraIntrinsics, mode::RenderMode)::Renderer
    if typeof(mode) == DepthMode
        mode_sting = "depth"
    elseif typeof(mode) == RGBMode
        mode_sting = "rgb"
    elseif typeof(mode) == TextureMode
        mode_sting = "texture"
    else
        error("unsupported mode $(mode)")
    end

    gl_instance = glrenderer_python.GLRenderer(
        camera_intrinsics.width,
        camera_intrinsics.height,
        camera_intrinsics.fx,
        camera_intrinsics.fy,
        camera_intrinsics.cx,
        camera_intrinsics.cy,
        camera_intrinsics.near,
        camera_intrinsics.far,
        mode_sting)
    Renderer{typeof(mode)}(gl_instance, camera_intrinsics)
end

function load_object!(renderer::Renderer{DepthMode}, vertices, faces)
    renderer.gl_instance.load_object(vertices, false, faces, false, false)
end
function load_object!(renderer::Renderer{RGBMode}, vertices, normals, faces)
    renderer.gl_instance.load_object(vertices, normals, faces, false, false)
end
function load_object!(renderer::Renderer{TextureMode}, vertices, normals, faces, texture_coords, texture_path)
    renderer.gl_instance.load_object(vertices, normals, faces, texture_coords, texture_path)
end

function pose_to_model_matrix(pose::Pose)::Matrix{Float32}
    R = Matrix{Float32}(pose.orientation)
    mat = zeros(Float32,4,4)
    mat[end,end] = 1
    mat[1:3,1:3] = R
    mat[1:3,4] = pose.pos
    return mat
end

function gl_render_depth_image(
        renderer::Renderer{DepthMode}, mesh_ids::Vector{Int},
        poses::Vector{Pose}, camera_pose::Pose
)
    renderer.gl_instance.V = pose_to_model_matrix(
        inv(Pose([camera_pose.pos...], camera_pose.orientation)))

    new_poses = [let
        q = Rotations.UnitQuaternion(p.orientation)
        [p.pos..., q.w, q.x, q.y, q.z]
        end 
    for p in poses
    ]
    depth = renderer.gl_instance.render([i-1 for i in mesh_ids], new_poses);
    depth
end

function gl_render_rgb_depth_image(
    renderer::Renderer{RGBMode}, mesh_ids::Vector{Int},
    poses::Vector{Pose}, colors::Vector{<:Color}, camera_pose::Pose
)
    renderer.gl_instance.V = pose_to_model_matrix(
        inv(Pose([camera_pose.pos...], camera_pose.orientation)))

    new_poses = [let
        q = Rotations.UnitQuaternion(p.orientation)
        [p.pos..., q.w, q.x, q.y, q.z]
        end 
    for p in poses
    ]
    rgb, depth = renderer.gl_instance.render([i-1 for i in mesh_ids],
        new_poses,
        [[c.r, c.g, c.b, c.alpha] for c in map(RGBA,colors)]
    );
    rgb = cat(rgb[:,:,3],rgb[:,:,2],rgb[:,:,1],rgb[:,:,4],dims=3)
    rgb = clamp.(rgb, 0.0, 1.0)
    rgb, depth
end

function gl_render_rgb_depth_image(
    renderer::Renderer{TextureMode}, mesh_ids::Vector{Int},
    poses::Vector{Pose}, camera_pose::Pose
)
    renderer.gl_instance.V = pose_to_model_matrix(
        inv(Pose([camera_pose.pos...], camera_pose.orientation)))

    new_poses = [let
        q = Rotations.UnitQuaternion(p.orientation)
        [p.pos..., q.w, q.x, q.y, q.z]
        end 
    for p in poses
    ]
    rgb, depth = renderer.gl_instance.render([i-1 for i in mesh_ids],
        new_poses,
    );
    rgb = cat(rgb[:,:,3],rgb[:,:,2],rgb[:,:,1],rgb[:,:,4],dims=3)
    rgb = clamp.(rgb, 0.0, 1.0)
    rgb, depth
end

export setup_renderer, load_object!, gl_render_depth_image

end
