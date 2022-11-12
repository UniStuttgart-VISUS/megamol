uniform vec4 clipplane; // a,b,c,d
uniform mat4 ModelViewProjectionMatrix;
layout(location = 0) in vec4 vertex;

void main() {
  // OLD vec4 pos = vec4(gl_Vertex.xyz, 1.0);
  vec4 pos = vec4(vertex.xyz, 1.0);
  gl_FrontColor = vec4(1.0, 0.0, 0.0, 1.0);
  if (clipplane != vec4(0.0)) {
    float d = dot(pos.xyz, clipplane.xyz);
    if (d > -clipplane.w) {
      gl_FrontColor = vec4(0.0, 0.0, 1.0, 1.0);
      pos.w = 0.0;
    }
  }
  pos = ModelViewProjectionMatrix * pos;
  gl_Position = pos / pos.w;
  gl_PointSize = 4.0;
}
