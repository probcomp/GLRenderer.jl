import Printf

function voxelize(cloud, resolution)
    cloud_xyz = round.(cloud[1:min(size(cloud,1), 3),:] ./ resolution) * resolution
    idxs = unique(i -> cloud_xyz[:,i], 1:size(cloud_xyz)[2])
    if size(cloud,1) > 3
        vcat(cloud_xyz[:, idxs], cloud[4:end, idxs])
    else
        cloud_xyz[:, idxs]
    end
end

const cube_vertices = [
 1.         -1.          1. ;
-1.         -1.          1. ;
-1.         -1.         -1. ;
-1.          1.         -1. ;
-1.          1.          1. ;
 1.          1.          1. ;
 1.          1.         -1. ;
 1.          1.          1. ;
 1.         -1.          1. ;
 1.          1.          1. ;
-1.          1.          1. ;
-1.         -1.          1. ;
-1.         -1.          1. ;
-1.          1.          1. ;
-1.          1.         -1. ;
 1.         -1.         -1. ;
-1.         -1.         -1. ;
-1.          1.         -1. ;
 1.         -1.         -1. ;
 1.          1.         -1. ;
 1.         -1.         -1. ;
 1.         -1.          1. ;
-1.         -1.         -1. ;
 1.          1.         -1.
 ]

const cube_normals = [
 0. -1.  0.;
 0. -1.  0.;
 0. -1.  0.;
 0.  1.  0.;
 0.  1.  0.;
 0.  1.  0.;
 1.  0.  0.;
 1.  0.  0.;
 1.  0.  0.;
-0.  0.  1.;
-0.  0.  1.;
-0.  0.  1.;
-1. -0. -0.;
-1. -0. -0.;
-1. -0. -0.;
 0.  0. -1.;
 0.  0. -1.;
 0.  0. -1.;
 0. -1.  0.;
 0.  1.  0.;
 1.  0.  0.;
-0.  0.  1.;
-1. -0. -0.;
 0.  0. -1.;
]

const cube_faces = [
 0  1  2;
 3  4  5;
 6  7  8;
 9 10 11;
12 13 14;
15 16 17;
18  0  2;
19  3  5;
20  6  8;
21  9 11;
22 12 14;
23 15 17;
]

function mesh_from_voxelized_cloud(cloud, resolution)
    new_v = vcat([cube_vertices .* resolution/2.0 .+ r' for r in eachcol(cloud)]...)
    new_n = vcat([cube_normals for _ in 1:size(cloud)[2]]...)
    new_f = vcat([cube_faces .+ 24*(i-1) for i in 1:size(cloud)[2]]...)
    Mesh(vertices=permutedims(new_v),
         indices=permutedims(new_f), normals=permutedims(new_n))
end

function box_mesh_from_dims(dims)
    Mesh(
        vertices=permutedims(cube_vertices .* (dims ./ 2.0)'),
        indices=permutedims(cube_faces),
        normals=permutedims(cube_normals)
    )
end

function box_container_mesh_from_dims(dims)
    cube_faces_minus_top = cube_faces[1:end .∉ [[1, 7]],:]
    Mesh(
        vertices=permutedims(cube_vertices .* (dims ./ 2.0)'),
        indices=permutedims(cube_faces_minus_top),
        normals=permutedims(cube_normals)
    )
end

function box_wireframe_mesh_from_dims(dims, w)
    x,y,z = dims
    wireframe_mesh = +([
        let
            mesh_x = box_mesh_from_dims([x + w, w, w])
            mesh_x = Pose([0.0, s_x * y/2.0, s_y * z/2.0]) * mesh_x
            mesh_y = box_mesh_from_dims([w, y + w, w])
            mesh_y = Pose([s_x * x/2.0, 0.0, s_y * z/2.0]) * mesh_y
            mesh_z = box_mesh_from_dims([w, w, z + w])
            mesh_z = Pose([s_x * x/2.0, s_y * y/2.0, 0.0]) * mesh_z
            mesh_x + mesh_y + mesh_z
        end
        for (s_x, s_y) in [(-1,1),(-1,-1),(1,-1),(1,1)]
    ]...)
    wireframe_mesh
end


function Base.:(+)(a::Mesh, b::Mesh)::Mesh
    Mesh(
        vertices=hcat(a.vertices, b.vertices),
        indices=hcat(a.indices, size(a.vertices)[2] .+ b.indices),
        normals=hcat(a.normals, b.normals),
    )    
end

function Base.:(*)(p::Pose, m::Mesh)::Mesh
    mesh_copy = copy_mesh(m)
    mesh_copy.vertices = p * mesh_copy.vertices
    mesh_copy
end

function get_mesh_data_from_obj_file(obj_file_path; tex_path=nothing, scaling_factor=1.0)
    mesh = FileIO.load(obj_file_path);    
    vertices = hcat([[x...] for x in mesh.position]...) .* scaling_factor
    indices = hcat([[map(GB.value,a)...] for a in GB.faces(mesh)]...) .- 1

    if hasproperty(mesh, :normals)
        normals = hcat([[x...] for x in mesh.normals]...)
        normals = Matrix{Float64}(normals)
    else
        normals = nothing
    end

    if hasproperty(mesh, :uv)
        tex_coords = hcat([[x...] for x in mesh.uv]...)
        tex_coords = Matrix{Float64}(tex_coords)
    else
        tex_coords = nothing
    end

    vertices = Matrix{Float64}(vertices)
    indices = Matrix{UInt32}(indices)

    Mesh(vertices=vertices, indices=indices, normals=normals,
         tex_coords=tex_coords, tex_path=tex_path)
end


function write_mesh_to_obj_file(mesh::Mesh, filepath::String)
    open(filepath, "w") do file
        for ver in eachcol(mesh.vertices)
            Printf.@printf(file, "v %.4f %.4f %.4f\n", ver[1],ver[2],ver[3])
        end
        for ver in eachcol(mesh.indices)
            Printf.@printf(file, "f %d %d %d\n", (ver .+ 1)...)
        end
    end
end

export voxelize, mesh_from_voxelized_cloud
