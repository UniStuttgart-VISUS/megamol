uniform vec3 camX;
uniform vec3 camY;
uniform vec3 camZ;

void main() {
  vec4 pos = gl_Vertex;
  pos = vec4(dot(camX, pos.xyz), dot(camY, pos.xyz), 0.0, 1.0);
  pos = gl_ModelViewProjectionMatrix * pos;
  gl_FrontColor = vec4(1.0, 0.0, 0.0, 0.0);
  gl_Position = pos;
  gl_PointSize = 4.0;
}
