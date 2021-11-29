#version 400

layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

in vec4 color[];
flat out vec4 pos;
out vec4 myColor;
out vec2 texCoords;

in float myRadius[];
out float radius;

uniform mat4 mvp;
uniform mat4 view;
uniform mat4 proj;
uniform mat4 model = mat4(1.0);
uniform mat4 invModel = mat4(1.0);
uniform vec3 camPos;
uniform vec3 camRight;
uniform vec3 camUp;
uniform vec3 camDir;

void main () {
    myColor = color[0];
    radius = myRadius[0];
    pos = gl_in[0].gl_Position;

    // Right and up vector of the billboard plane
    // NOT THE RIGHT AND UP VECTORS OF THE CAMERA!!!
    vec3 right;
    vec3 up;

    vec3 cPos = vec3(invModel * vec4(camPos, 1.0));

    // Normal vector of the billboard plane
    vec3 n = normalize(cPos - pos.xyz);

    // Project right and up vector onto the billboard plane
    right = camRight - dot(camRight, n) * n;
    up = camUp - dot(camUp, n) * n;

    right = normalize(right);
    up = normalize(up);

    // compute all 4 vertex positions
    vec3 vb = pos.xyz - (right - up) * radius;
    texCoords = vec2(0, 1);
    gl_Position = mvp * vec4(vb, 1.0);
    EmitVertex();

    vec3 va = pos.xyz - (right + up) * radius;
    texCoords = vec2(0, 0);
    gl_Position = mvp * vec4(va, 1.0);
    EmitVertex();

    vec3 vc = pos.xyz + (right + up) * radius;
    texCoords = vec2(1, 1);
    gl_Position = mvp * vec4(vc, 1.0);
    EmitVertex();

    vec3 vd = pos.xyz + (right - up) * radius;
    texCoords = vec2(1, 0);
    gl_Position = mvp * vec4(vd, 1.0);
    EmitVertex();

    EndPrimitive();
}
