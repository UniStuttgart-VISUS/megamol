#version 430

layout(points) in;
layout(line_strip, max_vertices = 6) out;

in vec4 vs_orient[];

uniform mat4 mvp;
uniform float direction_len;

out vec4 vertColor;

vec3 qtransform(vec4 q, vec3 v) {
    return v + 2.0 * cross(cross(v, q.xyz) + q.w * v, q.xyz);
}

void main(void) {
    vec4 quat = normalize(vs_orient[0]);
    quat = vec4(-quat.x, -quat.y, -quat.z, quat.w);

    vec3 up = qtransform(quat, vec3(0.0, 1.0, 0.0));
    vec3 view = qtransform(quat, vec3(0.0, 0.0, -1.0));
    //vec3 right = qtransform(quat, vec3(1.0, 0.0, 0.0));
    // this is cheaper
    vec3 right = cross(view, up);

    vec3 pos = gl_in[0].gl_Position.xyz;

    // right
    vertColor = vec4(1.0, 0.4, 0.4, 1.0);
    gl_Position = mvp * vec4(pos, 1.0);
    EmitVertex();
    gl_Position = mvp * vec4(pos + right * direction_len, 1.0);
    EmitVertex();
    EndPrimitive();

    // in
    vertColor = vec4(0.4, 0.4, 1.0, 1.0);
    gl_Position = mvp * vec4(pos, 1.0);
    EmitVertex();
    gl_Position = mvp * vec4(pos + view * direction_len, 1.0);
    EmitVertex();
    EndPrimitive();

    // up
    vertColor = vec4(0.4, 1.0, 0.4, 1.0);
    gl_Position = mvp * vec4(pos, 1.0);
    EmitVertex();
    gl_Position = mvp * vec4(pos + up * direction_len, 1.0);
    EmitVertex();
    EndPrimitive();
}
