uniform mat4 view_mx;
uniform mat4 proj_mx;

layout(location = 0) in vec3 v_position;

void main()
{
    gl_Position =  proj_mx * view_mx * vec4(v_position,1.0);
}
