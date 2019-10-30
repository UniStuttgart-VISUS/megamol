layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

uniform mat4 view_mx;
uniform mat4 proj_mx;

in vec4 colors[];
out vec4 color;

void main() {
    const float illum_coeff =
        calculate_illumination(view_mx, gl_in[0].gl_Position, gl_in[1].gl_Position, gl_in[2].gl_Position);

    gl_Position = proj_mx * view_mx * gl_in[0].gl_Position;
    color = illum_coeff * colors[0];
    EmitVertex();

    gl_Position = proj_mx * view_mx * gl_in[1].gl_Position;
    color = illum_coeff * colors[1];
    EmitVertex();

    gl_Position = proj_mx * view_mx * gl_in[2].gl_Position;
    color = illum_coeff * colors[2];
    EmitVertex();

    EndPrimitive();
}
