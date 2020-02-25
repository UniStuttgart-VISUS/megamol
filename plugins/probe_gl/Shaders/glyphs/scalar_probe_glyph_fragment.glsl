
layout(std430, binding = 0) readonly buffer MeshShaderParamsBuffer { MeshShaderParams[] mesh_shader_params; };

layout(location = 0) flat in int draw_id;
layout(location = 1) in vec2 uv_coords;

layout(location = 0) out vec4 albedo_out;
layout(location = 1) out vec3 normal_out;
layout(location = 2) out float depth_out;

void main() {
    albedo_out = vec4(uv_coords,0.0,1.0);
    normal_out = vec3(0.0,0.0,1.0);
    depth_out = gl_FragCoord.z;
}