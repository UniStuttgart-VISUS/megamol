uniform mat4 modelView = mat4(1.0);
uniform mat4 projection = mat4(1.0);

layout(location = 0) in vec3 in_position;

void main()
{
    gl_Position = projection * modelView * vec4(in_position, 1.0);
}
