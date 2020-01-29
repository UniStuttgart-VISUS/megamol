layout(triangles) in;
layout(line_strip, max_vertices = 4) out;

uniform mat4 view_mx;
uniform mat4 proj_mx;

in vec4 colors[];
in vec3 normals[];

out vec4 color;
out vec3 normal;

void main() {
    gl_Position = proj_mx * view_mx * gl_in[0].gl_Position;
    color = colors[0];
    normal = normals[0];
    EmitVertex();

    gl_Position = proj_mx * view_mx * gl_in[1].gl_Position;
    color = colors[1];
    normal = normals[1];
    EmitVertex();

    gl_Position = proj_mx * view_mx * gl_in[2].gl_Position;
    color = colors[2];
    normal = normals[2];
    EmitVertex();

    gl_Position = proj_mx * view_mx * gl_in[0].gl_Position;
    color = colors[0];
    normal = normals[0];
    EmitVertex();

    EndPrimitive();
}
