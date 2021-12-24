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
    new_v, new_n, new_f
end

function box_mesh_from_dims(dims)
    (cube_vertices .* (dims ./ 2.0)'), cube_normals, cube_faces
end

export voxelize, mesh_from_voxelized_cloud
