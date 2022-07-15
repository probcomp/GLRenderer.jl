function image_from_clouds_and_colors(clouds, colors, intrinsics)
    img = fill(I.colorant"white", intrinsics.height, intrinsics.width)
    for (c, col) in zip(clouds, colors)
		if isempty(c)
			continue
		end
        pixel_coordinates = point_cloud_to_pixel_coordinates(c, intrinsics)
        mask = (
            (1 .<= pixel_coordinates[1,:] .<= intrinsics.width) .&
            (1 .<= pixel_coordinates[2,:] .<= intrinsics.height)
        )
        pixel_coordinates = pixel_coordinates[:, mask]
        unraveled_index = (pixel_coordinates[1, :] .- 1) .* intrinsics.height .+ pixel_coordinates[2,:] 
        img[unraveled_index] .= col
    end
	return img
end