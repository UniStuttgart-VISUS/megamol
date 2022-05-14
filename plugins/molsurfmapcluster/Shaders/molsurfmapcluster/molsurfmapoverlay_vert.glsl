layout(location = 0) in vec3 in_position;
layout(location = 1) in vec2 in_tc;

uniform vec2 lowerleft = vec2(0.0, 0.0);
uniform vec2 upperright = vec2(1.0, 1.0);

uniform mat4 mvp = mat4(1.0);

out vec2 texCoord;

void main()
{
    texCoord = in_tc;
    gl_Position = mvp * vec4(mix(lowerleft, upperright, in_position.xy), in_position.z, 1.0);
}