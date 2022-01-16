obj_path = joinpath(@__DIR__, "035_power_drill/textured_simple.obj")
texture_path = joinpath(@__DIR__, "035_power_drill/texture_map.png")

include("test_depth.jl")
include("test_rgb_basic.jl")
include("test_rgb.jl")
include("test_texture.jl")
