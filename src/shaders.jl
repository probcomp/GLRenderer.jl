################
# shader utils #
################

# from https://github.com/JuliaGL/ModernGL.jl/blob/d56e4ad51f4459c97deeea7666361600a1e6065e/test/util.jl

function validateShader(shader)
	success = GLint[0]
	glGetShaderiv(shader, GL_COMPILE_STATUS, success)
	success[] == GL_TRUE
end

function glErrorMessage()
# Return a string representing the current OpenGL error flag, or the empty string if there's no error.
	err = glGetError()
	err == GL_NO_ERROR ? "" :
	err == GL_INVALID_ENUM ? "GL_INVALID_ENUM: An unacceptable value is specified for an enumerated argument. The offending command is ignored and has no other side effect than to set the error flag." :
	err == GL_INVALID_VALUE ? "GL_INVALID_VALUE: A numeric argument is out of range. The offending command is ignored and has no other side effect than to set the error flag." :
	err == GL_INVALID_OPERATION ? "GL_INVALID_OPERATION: The specified operation is not allowed in the current state. The offending command is ignored and has no other side effect than to set the error flag." :
	err == GL_INVALID_FRAMEBUFFER_OPERATION ? "GL_INVALID_FRAMEBUFFER_OPERATION: The framebuffer object is not complete. The offending command is ignored and has no other side effect than to set the error flag." :
	err == GL_OUT_OF_MEMORY ? "GL_OUT_OF_MEMORY: There is not enough memory left to execute the command. The state of the GL is undefined, except for the state of the error flags, after this error is recorded." : "Unknown OpenGL error with error code $err."
end

function getInfoLog(obj::GLuint)
	# Return the info log for obj, whether it be a shader or a program.
	isShader = glIsShader(obj)
	getiv = isShader == GL_TRUE ? glGetShaderiv : glGetProgramiv
	getInfo = isShader == GL_TRUE ? glGetShaderInfoLog : glGetProgramInfoLog
	# Get the maximum possible length for the descriptive error message
	len = GLint[0]
	getiv(obj, GL_INFO_LOG_LENGTH, len)
	maxlength = len[]
	# TODO: Create a macro that turns the following into the above:
	# maxlength = @glPointer getiv(obj, GL_INFO_LOG_LENGTH, GLint)
	# Return the text of the message if there is any
	if maxlength > 0
		buffer = zeros(GLchar, maxlength)
		sizei = GLsizei[0]
		getInfo(obj, maxlength, sizei, buffer)
		len = sizei[]
		unsafe_string(pointer(buffer), len)
	else
		""
	end
end

function createShader(source, typ)

    # Create the shader
	shader = glCreateShader(typ)::GLuint
	if shader == 0
		error("Error creating shader: ", glErrorMessage())
	end

	# Compile the shader
	glShaderSource(
        shader, 1, convert(Ptr{UInt8},
        pointer([convert(Ptr{GLchar}, pointer(source))])), C_NULL)
	glCompileShader(shader)

	# Check for errors
	!validateShader(shader) && error("Shader creation error: ", getInfoLog(shader))
	shader
end

##################
# OpenGL shaders #
##################

# vertex shader for computing depth image
const vertex_source = """
#version 410 core
uniform mat4 P;
uniform mat4 V;
uniform mat4 pose_mat;
in vec3 position;
void main() {
    gl_Position = P * V * pose_mat * vec4(position, 1);
}
"""

# fragment shader for sillhouette
const fragment_source = """
#version 410 core
out vec4 outColor;
void main()
{
    outColor = vec4(1.0, 1.0, 1.0, 1.0);
}
"""

const vertexShader_rgb_basic = """
#version 410 core
uniform mat4 P;
uniform mat4 V;
uniform mat4 pose_mat;
uniform vec4 color; 
in vec3 position;
out vec4 instance_color;
void main() {
    gl_Position = P * V * pose_mat * vec4(position, 1);
    instance_color = color;
}
"""

const fragmentShader_rgb_basic = """
#version 410 core
in vec4 instance_color;
layout(location = 0) out vec4 color;
void main() {
    color = instance_color;
}
"""



const vertexShader_rgb = """
#version 410 core
uniform mat4 P;
uniform mat4 V;
uniform mat4 pose_mat;
uniform mat4 pose_rot;
uniform vec4 color; 

in vec3 position;
in vec3 normal;

out vec4 instance_color;
out vec3 normal_out;
out mat4 V_out;
out mat4 pose_mat_out;
out mat4 pose_rot_out;
out vec3 fragVert;

void main() {
    gl_Position = P * V * pose_mat * vec4(position, 1);
    instance_color = color;
    normal_out = normal;
    V_out = V;
    pose_mat_out = pose_mat;
    pose_rot_out = pose_rot;
    fragVert = position;
}
"""

const fragmentShader_rgb = """
#version 410 core
in vec4 instance_color;
in vec3 normal_out;
in mat4 V_out;
in mat4 pose_mat_out;
in mat4 pose_rot_out;
in vec3 fragVert;

layout(location = 0) out vec4 color;
void main() {
    mat3 normalMatrix = transpose(inverse(mat3(pose_mat_out)));
    vec3 normal = normalize(normalMatrix * normal_out);
    
    //calculate the location of this fragment (pixel) in world coordinates
    vec3 fragPosition = vec3(pose_mat_out * vec4(fragVert, 1));
    
    //calculate the vector from this pixels surface to the light source
    vec3 surfaceToLight = vec3(0.0, 0.0, 0.0) - fragPosition;

    //calculate the cosine of the angle of incidence
    float brightness = dot(normal, surfaceToLight) / (length(surfaceToLight) * length(normal));
    brightness = clamp(brightness, 0, 1);

    //calculate final color of the pixel, based on:
    // 1. The angle of incidence: brightness
    // 2. The color/intensities of the light: light.intensities
    // 3. The texture and texture coord: texture(tex, fragTexCoord)
    vec4 surfaceColor = instance_color;
    color = vec4(brightness * vec3(1.0, 1.0, 1.0) * surfaceColor.rgb, surfaceColor.a);
}
"""


const vertexShader_texture = """
#version 410 core
uniform mat4 P;
uniform mat4 V;
uniform mat4 pose_mat;
uniform mat4 pose_rot;

in vec3 position;
in vec3 normal;
in vec2 vertTexCoord;

out vec3 normal_out;
out mat4 V_out;
out mat4 pose_mat_out;
out mat4 pose_rot_out;
out vec3 fragVert;
out vec2 fragTexCoord;

void main() {
    gl_Position = P * V * pose_mat * vec4(position, 1);
    normal_out = normal;
    V_out = V;
    pose_mat_out = pose_mat;
    pose_rot_out = pose_rot;
    fragVert = position;
    fragTexCoord = vertTexCoord;
}
"""

const fragmentShader_texture = """
#version 410 core

uniform sampler2D tex;
in vec3 normal_out;
in mat4 V_out;
in mat4 pose_mat_out;
in mat4 pose_rot_out;
in vec3 fragVert;
in vec2 fragTexCoord;

layout(location = 0) out vec4 color;
void main() {
    mat3 normalMatrix = transpose(inverse(mat3(pose_mat_out)));
    vec3 normal = normalize(normalMatrix * normal_out);
    
    //calculate the location of this fragment (pixel) in world coordinates
    vec3 fragPosition = vec3(pose_mat_out * vec4(fragVert, 1));
    
    //calculate the vector from this pixels surface to the light source
    vec3 surfaceToLight = vec3(0.0, 0.0, 0.0) - fragPosition;

    //calculate the cosine of the angle of incidence
    float brightness = dot(normal, surfaceToLight) / (length(surfaceToLight) * length(normal));
    brightness = clamp(brightness, 0, 1);

    //calculate final color of the pixel, based on:
    // 1. The angle of incidence: brightness
    // 2. The color/intensities of the light: light.intensities
    // 3. The texture and texture coord: texture(tex, fragTexCoord)
	vec4 surfaceColor = texture(tex, fragTexCoord);
    color = vec4(brightness * vec3(1.0, 1.0, 1.0) * surfaceColor.rgb, surfaceColor.a);
}
"""
