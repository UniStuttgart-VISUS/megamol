#extension GL_ARB_shader_draw_parameters : require

struct MeshShaderParams
{
    vec4 glpyh_position;
};

layout(std430, binding = 0) readonly buffer MeshShaderParamsBuffer { MeshShaderParams[] mesh_shader_params; };

uniform mat4 view_mx;
uniform mat4 proj_mx;

void main()
{
    vec4 glyph_pos = mesh_shader_params[gl_DrawIDARB].glpyh_position;

    const vec4 vertices[6] = vec4[6]( vec4( -1.0,-1.0,0.0,0.0 ),
									  vec4( 1.0,1.0,1.0,1.0 ),
									  vec4( -1.0,1.0,0.0,1.0 ),
									  vec4( 1.0,1.0,1.0,1.0 ),
									  vec4( -1.0,-1.0,0.0,0.0 ),
                                	  vec4( 1.0,-1.0,1.0,0.0 ) );

    vec4 clip_pos = (proj_mx * view_mx * glyph_pos);  

    float aspect = proj_mx[1][1] / proj_mx[0][0];
    vec2  bboard_vertex = vertices[gl_VertexID].xy;

    gl_Position =  clip_pos + vec4(bboard_vertex.x * 0.2,bboard_vertex.y * 0.2 * aspect,0.0,0.0);;
}