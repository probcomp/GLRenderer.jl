module GLRenderer

import PoseComposition: Pose
import Rotations
import Images as I
import Images: Color, RGBA
import Parameters: @with_kw
import GLFW
import PyCall
import FileIO
import MeshIO
import GeometryBasics as GB
import ColorSchemes
using ModernGL



@with_kw struct CameraIntrinsics
    width::Int = 640
    height::Int = 480
    fx::Float64 = width
    fy::Float64 = width
    cx::Float64 = width/2
    cy::Float64 = height/2
    near::Float64 = 0.001
    far::Float64 = 100.0
end

@with_kw mutable struct Mesh
    vertices::Any = nothing
    indices::Any = nothing
    normals::Any = nothing
    tex_coords::Any = nothing
    tex_path::Any = nothing
end

copy_mesh(m::Mesh) = Mesh(
                        vertices=m.vertices,
                        indices=m.indices,
                        normals=m.normals,
                        tex_coords=m.tex_coords,
                        tex_path=m.tex_path,
                    )

include("shaders.jl")
include("utils.jl")
include("mesh.jl")
include("point_cloud.jl")


abstract type RenderMode end
struct DepthMode <: RenderMode end
struct RGBBasicMode <: RenderMode end
struct RGBMode <: RenderMode end
struct TextureMode <: RenderMode end
struct TextureMixedMode <: RenderMode end

mutable struct Renderer{T <: RenderMode}
    window::Any
    camera_intrinsics::CameraIntrinsics
    shader_program::Any
    mesh_pointers::Any
    mesh_sizes::Any
    textures::Any
    perspective_matrix::Matrix
end

function setup_renderer(camera_intrinsics::CameraIntrinsics, mode::RenderMode; name="GLRenderer", gl_version=nothing)::Renderer
    if isnothing(gl_version)
        window = GLFW.CreateWindow(1,1, "dummy")
        GLFW.MakeContextCurrent(window)
        version_string = unsafe_string(glGetString(GL_VERSION))
        gl_version = (i -> parse(Int,i)).(split(split(version_string," ")[1],"."))
        GLFW.DestroyWindow(window)
    end

    perspective_matrix = get_perspective_matrix(
        camera_intrinsics.width, camera_intrinsics.height,
        camera_intrinsics.fx,
        camera_intrinsics.fy,
        camera_intrinsics.cx,
        camera_intrinsics.cy,
        camera_intrinsics.near,
        camera_intrinsics.far,
    )

    window_hint = [
        # (GLFW.SAMPLES,      0),
        # (GLFW.DEPTH_BITS,   24),
        # (GLFW.ALPHA_BITS,   8),
        # (GLFW.RED_BITS,     8),
        # (GLFW.GREEN_BITS,   8),
        # (GLFW.BLUE_BITS,    8),
        # (GLFW.STENCIL_BITS, 0),
        # (GLFW.AUX_BUFFERS,  0),
        (GLFW.CONTEXT_VERSION_MAJOR, gl_version[1]),
        (GLFW.CONTEXT_VERSION_MINOR, gl_version[2]),
        (GLFW.OPENGL_PROFILE, GLFW.OPENGL_CORE_PROFILE),
        (GLFW.OPENGL_FORWARD_COMPAT, GL_TRUE),
        (GLFW.VISIBLE, GL_FALSE)
    ]
    for (key, value) in window_hint
        GLFW.WindowHint(key, value)
    end

    window = GLFW.CreateWindow(camera_intrinsics.width, camera_intrinsics.height, "GLRenderer")
    GLFW.MakeContextCurrent(window)
    glEnable(GL_DEPTH_TEST)
    glClear(GL_DEPTH_BUFFER_BIT)

    gl_version_for_shaders = "$(gl_version[1])$(gl_version[2])0"
    @show gl_version_for_shaders
    if typeof(mode) == DepthMode
        vertex_shader = createShader(vertex_source_depth(gl_version_for_shaders), GL_VERTEX_SHADER)
        fragment_shader = createShader(fragment_source_depth(gl_version_for_shaders), GL_FRAGMENT_SHADER)
    elseif typeof(mode) == RGBBasicMode
        vertex_shader = createShader(vertexShader_rgb_basic(gl_version_for_shaders), GL_VERTEX_SHADER)
        fragment_shader = createShader(fragmentShader_rgb_basic(gl_version_for_shaders), GL_FRAGMENT_SHADER)
    elseif typeof(mode) == RGBMode
        vertex_shader = createShader(vertexShader_rgb(gl_version_for_shaders), GL_VERTEX_SHADER)
        fragment_shader = createShader(fragmentShader_rgb(gl_version_for_shaders), GL_FRAGMENT_SHADER)
    elseif typeof(mode) == TextureMode
        vertex_shader = createShader(vertexShader_texture(gl_version_for_shaders), GL_VERTEX_SHADER)
        fragment_shader = createShader(fragmentShader_texture(gl_version_for_shaders), GL_FRAGMENT_SHADER)
    elseif typeof(mode) == TextureMixedMode
        vertex_shader = createShader(vertexShader_texture_mixed(gl_version_for_shaders), GL_VERTEX_SHADER)
        fragment_shader = createShader(fragmentShader_texture_mixed(gl_version_for_shaders), GL_FRAGMENT_SHADER)
    else
        error("unsupported mode $(mode)")
    end

    shader_program = glCreateProgram()
    glAttachShader(shader_program, vertex_shader)
    glAttachShader(shader_program, fragment_shader)
    glLinkProgram(shader_program)

    fbo = Ref(GLuint(0))
    glGenFramebuffers(1, fbo)

    depth_tex = Ref(GLuint(0))
    glGenTextures(1, depth_tex)
    color_tex = Ref(GLuint(0))
    glGenTextures(1, color_tex)

    glBindFramebuffer(GL_FRAMEBUFFER, fbo[])

    glBindTexture(GL_TEXTURE_2D, depth_tex[])
    glTexImage2D(
      GL_TEXTURE_2D, 0, GL_DEPTH24_STENCIL8, camera_intrinsics.width, camera_intrinsics.height, 0, 
      GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8, C_NULL
    );
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D, depth_tex[], 0);  

    glBindTexture(GL_TEXTURE_2D, color_tex[])
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, camera_intrinsics.width, camera_intrinsics.height, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, C_NULL)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color_tex[], 0)
    

    glViewport(0, 0, camera_intrinsics.width, camera_intrinsics.height)
    glDrawBuffers(2, [GL_DEPTH_STENCIL_ATTACHMENT, GL_COLOR_ATTACHMENT0])
    println(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE)

    Renderer{typeof(mode)}(window, camera_intrinsics, shader_program, [], [], [], perspective_matrix)
end

function activate_renderer(renderer::Renderer)
    GLFW.MakeContextCurrent(renderer.window)
end


function set_intrinsics!(renderer::Renderer, camera_intrinsics::CameraIntrinsics)
    renderer.camera_intrinsics = camera_intrinsics
    renderer.perspective_matrix = get_perspective_matrix(
        camera_intrinsics.width, camera_intrinsics.height,
        camera_intrinsics.fx,
        camera_intrinsics.fy,
        camera_intrinsics.cx,
        camera_intrinsics.cy,
        camera_intrinsics.near,
        camera_intrinsics.far,
    )

    fbo = Ref(GLuint(0))
    glGenFramebuffers(1, fbo)

    depth_tex = Ref(GLuint(0))
    glGenTextures(1, depth_tex)
    color_tex = Ref(GLuint(0))
    glGenTextures(1, color_tex)

    glBindFramebuffer(GL_FRAMEBUFFER, fbo[])

    glBindTexture(GL_TEXTURE_2D, depth_tex[])
    glTexImage2D(
      GL_TEXTURE_2D, 0, GL_DEPTH24_STENCIL8, camera_intrinsics.width, camera_intrinsics.height, 0, 
      GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8, C_NULL
    );
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D, depth_tex[], 0);

    glBindTexture(GL_TEXTURE_2D, color_tex[])
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, camera_intrinsics.width, camera_intrinsics.height, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, C_NULL)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color_tex[], 0)

    glViewport(0, 0, camera_intrinsics.width, camera_intrinsics.height)
    glDrawBuffers(2, [GL_DEPTH_STENCIL_ATTACHMENT, GL_COLOR_ATTACHMENT0])
    println(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE)
end

function load_object!(renderer::Renderer{T}, mesh) where T <: RenderMode
    vao = Ref(GLuint(0))
    glGenVertexArrays(1, vao)
    glBindVertexArray(vao[])

    if T == DepthMode || T == RGBBasicMode
        @assert size(mesh.vertices)[1] == 3
        vertex_data = mesh.vertices
    elseif T == RGBMode
        @assert size(mesh.vertices)[1] == 3
        @assert size(mesh.normals)[1] == 3
        @assert size(mesh.vertices)[2] == size(mesh.normals)[2]
        vertex_data = vcat(mesh.vertices, mesh.normals)
    else
        @assert size(mesh.vertices)[1] == 3
        @assert size(mesh.normals)[1] == 3
        if T == TextureMixedMode && isnothing(mesh.tex_coords)
            tex_coords = zeros(2, size(mesh.vertices)[2])
        else
            @assert size(mesh.tex_coords)[1] == 2
            tex_coords = mesh.tex_coords
        end
        @assert size(mesh.vertices)[2] == size(mesh.normals)[2] == size(tex_coords)[2]
        vertex_data = vcat(mesh.vertices, mesh.normals, tex_coords)
    end

    vertex_data = Matrix{Float32}(vertex_data)
    indices = Matrix{UInt32}(mesh.indices)

    # copy vertex data into an OpenGL buffer
    vbo = Ref(GLuint(0))
    glGenBuffers(1, vbo)
    glBindBuffer(GL_ARRAY_BUFFER, vbo[])
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertex_data), Ref(vertex_data, 1), GL_STATIC_DRAW)

    # element buffer object for indices
    ebo = Ref(GLuint(0))
    glGenBuffers(1, ebo)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo[])
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), Ref(indices, 1), GL_STATIC_DRAW)


    if T == DepthMode || T == RGBBasicMode
        glEnableVertexAttribArray(glGetAttribLocation(renderer.shader_program, "position"))
        
        # set vertex attribute pointers
        glVertexAttribPointer(glGetAttribLocation(renderer.shader_program, "position"),
            3, GL_FLOAT, GL_FALSE, 3 * sizeof(Float32), C_NULL)
        glEnableVertexAttribArray(glGetAttribLocation(renderer.shader_program, "position"))
    elseif T == RGBMode
        # set vertex attribute pointers
        glEnableVertexAttribArray(glGetAttribLocation(renderer.shader_program, "position"))
        glEnableVertexAttribArray(glGetAttribLocation(renderer.shader_program, "normal"))

        glVertexAttribPointer(
            glGetAttribLocation(renderer.shader_program, "position"),
            3, GL_FLOAT, GL_FALSE, 3 * sizeof(Float32) * 2, C_NULL)
        glVertexAttribPointer(
            glGetAttribLocation(renderer.shader_program, "normal"),
            3, GL_FLOAT, GL_FALSE, 3 * sizeof(Float32) * 2, C_NULL + (3 * sizeof(Float32)))
    else
        # set vertex attribute pointers
        glEnableVertexAttribArray(glGetAttribLocation(renderer.shader_program, "position"))
        glEnableVertexAttribArray(glGetAttribLocation(renderer.shader_program, "normal"))
        glEnableVertexAttribArray(glGetAttribLocation(renderer.shader_program, "vertTexCoord"))

        glVertexAttribPointer(
            glGetAttribLocation(renderer.shader_program, "position"),
            3, GL_FLOAT, GL_FALSE, 32, C_NULL)
        glVertexAttribPointer(
            glGetAttribLocation(renderer.shader_program, "normal"),
            3, GL_FLOAT, GL_FALSE, 32, C_NULL + 12)
        glVertexAttribPointer(
            glGetAttribLocation(renderer.shader_program, "vertTexCoord"),
            2, GL_FLOAT, GL_FALSE, 32, C_NULL + 24)

        PyCall.py"""
        from PIL import Image
        import numpy as np
        def load_texture_bytes(path):
            img = Image.open(path).transpose(Image.FLIP_TOP_BOTTOM)
            img_data = np.fromstring(img.tobytes(), np.uint8)
            return img_data, img.width, img.height
        """
        if isnothing(mesh.tex_path)
            push!(renderer.textures, nothing)
        else
            img_data, width, height = PyCall.py"load_texture_bytes"(mesh.tex_path);

            texture = Ref(GLuint(0))
            glGenTextures(1, texture)
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
            glBindTexture(GL_TEXTURE_2D, texture[])
            glTexParameterf(
                GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameterf(
                GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
            glTexParameterf(
                GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameterf(
                GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0,
                            GL_RGB, GL_UNSIGNED_BYTE, img_data)
            glGenerateMipmap(GL_TEXTURE_2D)
        
            push!(renderer.textures, texture)
        end

    end

    # unbind it
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)

    push!(renderer.mesh_pointers, vao[])
    push!(renderer.mesh_sizes, size(mesh.indices)[2] * 3)
    return true
end


function gl_render(
        renderer::Union{Renderer{T}}, mesh_ids::Vector{Int},
        poses::Vector{Pose}, camera_pose::Pose; colors=nothing
) where T <: RenderMode
    
    if isnothing(colors)
        colors = fill(I.colorant"black", length(poses))
    end
    colors = map(x -> convert(I.RGBA,x), colors)

    @assert length(mesh_ids) == length(poses)

    glClearColor(1.0, 1.0, 1.0, 1)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glEnable(GL_DEPTH_TEST)
    
    camera_pose_mat = pose_to_matrix(inv(camera_pose * Pose(zeros(3),Rotations.RotX(pi))))

    for (id, p, c) in zip(mesh_ids, poses, colors)
        if id < 1
            continue
        end
        vao = renderer.mesh_pointers[id]
        
        glUseProgram(renderer.shader_program)
        
        glUniformMatrix4fv(glGetUniformLocation(
                renderer.shader_program,"P"), 1, GL_FALSE, Ref(renderer.perspective_matrix, 1))
        glUniformMatrix4fv(glGetUniformLocation(
                renderer.shader_program,"V"), 1, GL_FALSE, Ref(camera_pose_mat, 1))
        glUniformMatrix4fv(glGetUniformLocation(
                renderer.shader_program,"pose_mat"), 1, GL_FALSE, Ref(pose_to_matrix(p), 1))

        if T == RGBBasicMode || T == RGBMode
            glUniform4fv(glGetUniformLocation(
                    renderer.shader_program,"color"), 1, Float32[c.r, c.g, c.b, c.alpha])
        elseif T == TextureMode || T == TextureMixedMode
            glUniformMatrix4fv(glGetUniformLocation(
                    renderer.shader_program,"pose_rot"), 1, GL_FALSE, Ref(pose_to_matrix(Pose(zeros(3),p.orientation)), 1))        

            if isnothing(renderer.textures[id])
                glUniform1f(glGetUniformLocation(
                    renderer.shader_program,"textured"), 0.0)
                glUniform4fv(glGetUniformLocation(
                        renderer.shader_program,"color"), 1, Float32[c.r, c.g, c.b, c.alpha])
            else
                glUniform1f(glGetUniformLocation(
                    renderer.shader_program,"textured"), 1.0)
                glActiveTexture(GL_TEXTURE0)
                glBindTexture(GL_TEXTURE_2D, renderer.textures[id][])
                glUniform1i(glGetUniformLocation(
                        renderer.shader_program,"tex"), 0)
            end
        end

        glBindVertexArray(vao)
        glDrawElements(GL_TRIANGLES, renderer.mesh_sizes[id], GL_UNSIGNED_INT, C_NULL)
        glBindVertexArray(0)
    end
    
    cam = renderer.camera_intrinsics
    data = Matrix{Float32}(undef, cam.width, cam.height)
    glReadPixels(0, 0, cam.width, cam.height, GL_DEPTH_COMPONENT, GL_FLOAT, Ref(data, 1))
    near,far = cam.near, cam.far
    depth_image = far .* near ./ (far .- (far .- near) .* data)
    depth_image = permutedims(depth_image[:,end:-1:1])

    if T == DepthMode
        return depth_image
    end

    glReadBuffer(GL_COLOR_ATTACHMENT0)        
    data1 = zeros(Float32, 4,cam.width, cam.height)
    glReadPixels(0, 0, cam.width, cam.height, GL_BGRA, GL_FLOAT, Ref(data1, 1))
    rgb = permutedims(data1,(3,2,1))[end:-1:1,:,:]
    rgb = cat(rgb[:,:,3],rgb[:,:,2],rgb[:,:,1],rgb[:,:,4],dims=3)
    return rgb, depth_image
end

# Viewing images
function view_depth_image(depth_image; scheme=nothing)
    max_val = maximum(depth_image)
    min_val = minimum(depth_image)
    vals = depth_image[(depth_image .> (min_val + 1e-3)) .&  (depth_image .< (max_val - 1e-3))]
    max_val = maximum(vals)
    min_val = minimum(vals)
    d = clamp.(depth_image, min_val, max_val)
    img = get(ColorSchemes.blackbody, d, (min_val, max_val))	
    I.convert.(I.RGBA, img)
end

function view_rgb_image(rgb_image; in_255=false)
    if in_255
        rgb_image = Float64.(rgb_image)./255.0
    end
    if size(rgb_image)[3] == 3
        img = I.colorview(I.RGB, permutedims(rgb_image,(3,1,2)))
        img = I.convert.(I.RGBA, img)
    else
        img = I.colorview(I.RGBA, permutedims(rgb_image,(3,1,2)))
    end
    img
end

end