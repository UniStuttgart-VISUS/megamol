
layout(std430, binding = 0) readonly buffer MeshShaderParamsBuffer { MeshShaderParams[] mesh_shader_params; };

uniform mat4 view_mx;
uniform mat4 proj_mx;

layout(location = 0) flat out int draw_id;
layout(location = 1) out vec2 uv_coords;

//http://www.neilmendoza.com/glsl-rotation-about-an-arbitrary-axis/
mat4 rotationMatrix(vec3 axis, float angle)
{
    axis = normalize(axis);
    float s = sin(angle);
    float c = cos(angle);
    float oc = 1.0 - c;
    
    return mat4(oc * axis.x * axis.x + c,           oc * axis.x * axis.y - axis.z * s,  oc * axis.z * axis.x + axis.y * s,  0.0,
                oc * axis.x * axis.y + axis.z * s,  oc * axis.y * axis.y + c,           oc * axis.y * axis.z - axis.x * s,  0.0,
                oc * axis.z * axis.x - axis.y * s,  oc * axis.y * axis.z + axis.x * s,  oc * axis.z * axis.z + c,           0.0,
                0.0,                                0.0,                                0.0,                                1.0);
}

void main()
{
    const vec4 vertices[6] = vec4[6]( vec4( -1.0,-1.0,0.0,0.0 ),
									  vec4( 1.0,1.0,1.0,1.0 ),
									  vec4( -1.0,1.0,0.0,1.0 ),
									  vec4( 1.0,1.0,1.0,1.0 ),
									  vec4( -1.0,-1.0,0.0,0.0 ),
                                	  vec4( 1.0,-1.0,1.0,0.0 ) );

    draw_id = gl_DrawIDARB;
    uv_coords = vertices[gl_VertexID].zw;

    vec4 glyph_pos = vec4(mesh_shader_params[gl_DrawIDARB].glpyh_position.xyz, 1.0);
    vec4 clip_pos = (proj_mx * view_mx * glyph_pos);  
    float aspect = proj_mx[1][1] / proj_mx[0][0];
    vec2  bboard_vertex = vertices[gl_VertexID].xy;
    
    gl_Position = clip_pos + vec4(bboard_vertex.x * mesh_shader_params[gl_DrawIDARB].scale,
                                 bboard_vertex.y * mesh_shader_params[gl_DrawIDARB].scale * aspect, 0.0, 0.0);
}