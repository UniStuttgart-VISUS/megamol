uniform vec4 viewAttr;
uniform vec3 camX;
uniform vec3 camY;
uniform vec3 camZ;
uniform vec3 bboxmin;
uniform vec3 bboxmax;
uniform vec3 posoffset;

varying vec3 objPos;
varying vec4 quat;
varying float rad;

void main() {
  vec4 inquat = gl_MultiTexCoord0 * vec4(-1.0, -1.0, -1.0, 1.0); // inverted/conjugated quaternion

  quat = inquat;
  objPos = gl_Vertex.xyz + posoffset;
  rad = gl_Vertex.w;

  vec4 pos = vec4(dot(camX, objPos), dot(camY, objPos), 0.0, 1.0);
  vec4 ppos = gl_ModelViewProjectionMatrix * pos;
  gl_Position = ppos / ppos.w;

  pos.x += OUTERRAD * gl_Vertex.w;
  vec4 ppos2 = gl_ModelViewProjectionMatrix * pos;
  gl_PointSize = 2.0 * abs(ppos2.x - ppos.x) / viewAttr.z;

  gl_FrontColor = gl_Color;
}
