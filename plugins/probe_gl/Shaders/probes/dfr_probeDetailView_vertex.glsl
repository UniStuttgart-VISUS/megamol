layout(location = 0) in vec3 v_position;

layout(location = 0) out vec3 world_pos;

void main()
{
    world_pos = v_position;
    gl_Position =  vec4(v_position,1.0);
}