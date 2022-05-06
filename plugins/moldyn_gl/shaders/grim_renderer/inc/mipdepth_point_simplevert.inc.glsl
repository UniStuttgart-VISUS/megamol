
uniform mat4 mvp;

void main(void) {
    vec4 p = mvp * gl_Vertex; // was: ftransform
    gl_Position = p;
}
