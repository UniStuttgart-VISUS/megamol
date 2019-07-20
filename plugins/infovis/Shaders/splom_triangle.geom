layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

in float vsValue[];

flat out vec3 gsFlatValues;
out vec3 gsBarycentricPosition;

void main() {
    // Add barycentric coordinates and face values.
    vec3 faceValues = vec3(vsValue[0], vsValue[1], vsValue[2]);
    for (int i = 0; i < 3; ++i) {
        gsFlatValues = faceValues;
        gsBarycentricPosition = vec3(0.0);
        gsBarycentricPosition[i] = 1.0;
        gl_Position = gl_in[i].gl_Position;
        EmitVertex();
    }
}
