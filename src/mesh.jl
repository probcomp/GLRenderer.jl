import Printf

function voxelize(cloud, resolution)
    cloud = round.(cloud ./ resolution) * resolution
    idxs = unique(i -> cloud[:,i], 1:size(cloud)[2])
    cloud[:, idxs]
end

const cube_vertices = [
-1.0 -1.0 1.0;
1.0 -1.0 1.0;
-1.0 1.0 1.0;
1.0 1.0 1.0;

1.0 -1.0 -1.0;
-1.0 -1.0 -1.0;
1.0 1.0 -1.0;
-1.0 1.0 -1.0;

1.0 -1.0 1.0;
1.0 -1.0 -1.0;
1.0 1.0 1.0;
1.0 1.0 -1.0;
        
-1.0 -1.0 -1.0;
-1.0 -1.0 1.0;
-1.0 1.0 -1.0;
-1.0 1.0 1.0;
        
-1.0 -1.0 -1.0;
1.0 -1.0 -1.0;
-1.0 -1.0 1.0;
1.0 -1.0 1.0;

-1.0 1.0 1.0;
1.0 1.0 1.0;
-1.0 1.0 -1.0;
1.0 1.0 -1.0;
]

const cube_normals = [0.0 0.0 1.0; 0.0 0.0 1.0; 0.0 0.0 1.0; 0.0 0.0 1.0; 0.0 0.0 -1.0; 0.0 0.0 -1.0; 0.0 0.0 -1.0; 0.0 0.0 -1.0; 1.0 0.0 0.0; 1.0 0.0 0.0; 1.0 0.0 0.0; 1.0 0.0 0.0; -1.0 0.0 0.0; -1.0 0.0 0.0; -1.0 0.0 0.0; -1.0 0.0 0.0; 0.0 -1.0 0.0; 0.0 -1.0 0.0; 0.0 -1.0 0.0; 0.0 -1.0 0.0; 0.0 1.0 0.0; 0.0 1.0 0.0; 0.0 1.0 0.0; 0.0 1.0 0.0]
const cube_faces = [0 2 1; 1 2 3; 4 6 5; 5 6 7; 8 10 9; 9 10 11; 12 14 13; 13 14 15; 16 18 17; 17 18 19; 20 22 21; 21 22 23]


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
    normals = hcat([[x...] for x in mesh.normals]...)
    tex_coords = hcat([[x...] for x in mesh.uv]...)
    
    vertices = Matrix{Float32}(vertices)
    indices = Matrix{UInt32}(indices)
    normals = Matrix{Float32}(normals)
    tex_coords = Matrix{Float32}(tex_coords)

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
