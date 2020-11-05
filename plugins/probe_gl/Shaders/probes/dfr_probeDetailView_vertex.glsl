layout(location = 0) in vec4 v_position;

layout(location = 0) out vec3 world_pos;
layout(location = 1) out float sample_value;
layout(location = 2) flat out int draw_id;

void main()
{
    draw_id = gl_DrawIDARB;
    world_pos = v_position.xyz;
    sample_value = v_position.w;
    gl_Position =  vec4(v_position.xyz,1.0);
}
