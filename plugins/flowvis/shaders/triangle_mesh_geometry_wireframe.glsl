layout(triangles) in;
layout(line_strip, max_vertices = 4) out;

uniform mat4 view_mx;
uniform mat4 proj_mx;

in vec4 colors[];

out vec4 color;
out vec3 normal;

void main() {
    const vec3 calculated_normal =
        calculate_normal(view_mx, gl_in[0].gl_Position, gl_in[1].gl_Position, gl_in[2].gl_Position);

    gl_Position = proj_mx * view_mx * gl_in[0].gl_Position;
    color = colors[0];
    normal = calculated_normal;
    EmitVertex();

    gl_Position = proj_mx * view_mx * gl_in[1].gl_Position;
    color = colors[1];
    normal = calculated_normal;
    EmitVertex();

    gl_Position = proj_mx * view_mx * gl_in[2].gl_Position;
    color = colors[2];
    normal = calculated_normal;
    EmitVertex();

    gl_Position = proj_mx * view_mx * gl_in[0].gl_Position;
    color = colors[0];
    normal = calculated_normal;
    EmitVertex();

    EndPrimitive();
}
