layout(location = 0) in vec3 in_position;
layout(location = 1) in vec2 in_tc;

uniform vec2 lowerleft = vec2(0.0, 0.0);
uniform vec2 upperright = vec2(1.0, 1.0);

out vec2 texCoord;

void main()
{
    texCoord = in_tc;
    gl_Position = vec4(in_position, 1.0);
}