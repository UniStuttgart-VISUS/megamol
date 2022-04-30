varying vec3 interpolRay;
const mat4 mvp = mat4(1.0);

void main(void) {
  gl_Position = mvp * gl_Vertex; // was: ftransform, should be screenspace with identity matrices
  interpolRay = gl_Normal.xyz;
}
