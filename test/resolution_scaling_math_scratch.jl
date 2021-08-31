# -*- coding: utf-8 -*-
# +
import Revise
import GLRenderer
import PoseComposition
import Rotations
import Geometry
import Plots
import Images
import GenParticleFilters
PF = GenParticleFilters

I = Images
PL = Plots

R = Rotations
P = PoseComposition
GL = GLRenderer
# -

Revise.errors()
Revise.revise()

# +
room_height_bounds = (-5.0, 5.0)
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

v1,n1,f1 = GL.mesh_from_voxelized_cloud(GL.voxelize(room_cloud_1, resolution), resolution * 1.05)
v2,n2,f2 = GL.mesh_from_voxelized_cloud(GL.voxelize(room_cloud_2, resolution), resolution * 1.05)
v3,n3,f3 = GL.mesh_from_voxelized_cloud(GL.voxelize(room_cloud_3, resolution), resolution * 1.05)
v4,n4,f4 = GL.mesh_from_voxelized_cloud(GL.voxelize(room_cloud_4, resolution), resolution * 1.05)

room_cloud = hcat(room_cloud_1, room_cloud_2, room_cloud_3, room_cloud_4)

PL.scatter(room_cloud_1[1,:], room_cloud_1[3,:],label="")
PL.scatter!(room_cloud_2[1,:], room_cloud_2[3,:],label="")
PL.scatter!(room_cloud_3[1,:], room_cloud_3[3,:],label="")
PL.scatter!(room_cloud_4[1,:], room_cloud_4[3,:],label="")

# PL.scatter!(v2[:,1],v2[:,3],label="")
# PL.scatter!(v3[:,1],v3[:,3],label="")
# PL.scatter!(v4[:,1],v4[:,3],label="")
# -

# # Subsampling Camera Intrinsics Transfomation
#
# The code below shows how we take a `camera_intrinsics_1` and scale it down by a factor of `FACTOR` to get new intrinsics parameters `camera_intrinsics_2`. When we render out depth images from both of these cameras, we see that the depth image from `camera_intrinsics_2` is just a subsampled version of the depth image from `camera_intrinsics_1`.

# +
W = 320
camera_intrinsics_1 = Geometry.CameraIntrinsics(
    W, 1,
    20.0, 1.0,
    W/2.0, 0.5,
    0.1, 20.0
)
renderer = GL.setup_renderer(camera_intrinsics_1, GL.DepthMode())
GL.load_object!(renderer, v1, f1)
GL.load_object!(renderer, v2, f2)
GL.load_object!(renderer, v3, f3)
GL.load_object!(renderer, v4, f4)
renderer.gl_instance.lightpos = [0,0,0]

cam_pose = P.Pose([1.0, 0.0, 3.0],R.RotY(-pi/4+ 0.0))
wall_colors = [I.colorant"red",I.colorant"green",I.colorant"blue",I.colorant"yellow"]
# wall_colors = [I.colorant"red",I.colorant"red",I.colorant"red",I.colorant"red"]
@time depth_1 = GL.gl_render(
    renderer, 
    [1,2,3,4], [P.IDENTITY_POSE, P.IDENTITY_POSE,P.IDENTITY_POSE,P.IDENTITY_POSE],
    cam_pose
);

FACTOR = 4
DESIRED_WIDTH = W / FACTOR

camera_intrinsics_2 = Geometry.CameraIntrinsics(
    DESIRED_WIDTH, 1,
    camera_intrinsics_1.fx / FACTOR, 1.0,
    DESIRED_WIDTH / 2.0 .+ (1/2 - 1/FACTOR *1/2) , 0.5,
    0.1, 20.0
)
renderer = GL.setup_renderer(camera_intrinsics_2, GL.DepthMode())
GL.load_object!(renderer, v1, f1)
GL.load_object!(renderer, v2, f2)
GL.load_object!(renderer, v3, f3)
GL.load_object!(renderer, v4, f4)
renderer.gl_instance.lightpos = [0,0,0]

wall_colors = [I.colorant"red",I.colorant"green",I.colorant"blue",I.colorant"yellow"]
# wall_colors = [I.colorant"red",I.colorant"red",I.colorant"red",I.colorant"red"]
@time depth_2 = GL.gl_render(
    renderer, 
    [1,2,3,4], [P.IDENTITY_POSE, P.IDENTITY_POSE,P.IDENTITY_POSE,P.IDENTITY_POSE],
    cam_pose
);

camera_intrinsics_2

# +
PL.plot()

cloud_1 = GL.flatten_point_cloud(GL.depth_image_to_point_cloud(depth_1, camera_intrinsics_1))
cloud_1 = GL.move_points_to_frame_b(cloud_1, cam_pose)

cloud_2 = GL.flatten_point_cloud(GL.depth_image_to_point_cloud(depth_2, camera_intrinsics_2))
cloud_2 = GL.move_points_to_frame_b(cloud_2, cam_pose)


pose= cam_pose
for i in 1:size(cloud_1)[2]
   PL.plot!([pose.pos[1], cloud_1[1,i]], [pose.pos[3], cloud_1[3,i]],
            color=I.colorant"blue",
            alpha=0.5,
            linewidth=2,
            label=false) 
end

for i in 1:size(cloud_2)[2]
   PL.plot!([pose.pos[1], cloud_2[1,i]], [pose.pos[3], cloud_2[3,i]],
            color=I.colorant"orange",
            alpha=0.5,
            linewidth=2,
            label=false) 
end


PL.plot!(xlim=(-10,10),ylim=(-10,10))
# -

depth_1

depth_2

# # Validation
#
# Below, I validate that depth_2 is a subsmapled version of depth_1. @Falk, this means that to pass the depth "slices" from MCS as input to this system, you will have to first choose a `FACTOR` to resize by and then ensure you use the correct transformed camera model for the renderer using in our SLAM code. (Refer to the code above to see how we get `camera_intrinsics_2` from `camera_intrinsics_1`. This is the correct transformation.)

all(abs.(depth_1[1:FACTOR:end] .- depth_2[:]) .< 0.005)
