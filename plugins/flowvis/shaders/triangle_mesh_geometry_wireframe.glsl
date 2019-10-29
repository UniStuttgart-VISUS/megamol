layout(triangles) in;
layout(line_strip, max_vertices = 4) out;

uniform mat4 view_mx;
uniform mat4 proj_mx;

in vec4 colors[];
out vec4 color;

void main() {
    const vec3 normal = normalize(cross(gl_in[0].gl_Position.xyz - gl_in[2].gl_Position.xyz, gl_in[0].gl_Position.xyz - gl_in[1].gl_Position.xyz));
    const vec3 light_dir_1 = normalize(vec3(1.0f, 1.0f, 1.0f));
    const vec3 light_dir_2 = normalize(vec3(-1.0f, 1.0f, -1.0f));

    const float illum_coeff = 1.0f; //max(0.1f, 0.5f * (dot(normal, light_dir_1) + dot(normal, light_dir_2)));

    gl_Position = proj_mx * view_mx * gl_in[0].gl_Position;
    color = illum_coeff * colors[0];
    EmitVertex();

    gl_Position = proj_mx * view_mx * gl_in[1].gl_Position;
    color = illum_coeff * colors[1];
    EmitVertex();

    gl_Position = proj_mx * view_mx * gl_in[2].gl_Position;
    color = illum_coeff * colors[2];
    EmitVertex();

    gl_Position = proj_mx * view_mx * gl_in[0].gl_Position;
    color = illum_coeff * colors[0];
    EmitVertex();

    EndPrimitive();
}
