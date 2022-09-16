varying vec4 x;
varying vec4 y;
varying vec4 z;
varying vec4 quat;

void main() {
  quat = gl_MultiTexCoord0;
  vec4 origin = gl_ModelViewProjectionMatrix * vec4(gl_Vertex.xyz, 1.0);
  vec3 vec = vec3(1.0, 0.0, 0.0);
  vec = ((2.0 * ((dot(quat.xyz, vec) * quat.xyz) + (quat.w * cross(quat.xyz, vec)))) + (((quat.w * quat.w) - dot(quat.xyz, quat.xyz)) * vec));
  vec4 xaxis = gl_ModelViewProjectionMatrix * vec4(gl_Vertex.xyz + vec, 1.0);
  vec = vec3(0.0, 1.0, 0.0);
  vec = ((2.0 * ((dot(quat.xyz, vec) * quat.xyz) + (quat.w * cross(quat.xyz, vec)))) + (((quat.w * quat.w) - dot(quat.xyz, quat.xyz)) * vec));
  vec4 yaxis = gl_ModelViewProjectionMatrix * vec4(gl_Vertex.xyz + vec, 1.0);
  vec = vec3(0.0, 0.0, 1.0);
  vec = ((2.0 * ((dot(quat.xyz, vec) * quat.xyz) + (quat.w * cross(quat.xyz, vec)))) + (((quat.w * quat.w) - dot(quat.xyz, quat.xyz)) * vec));
  vec4 zaxis = gl_ModelViewProjectionMatrix * vec4(gl_Vertex.xyz + vec, 1.0);
  gl_Position = origin;
  x = (xaxis - origin) * gl_Vertex.w;
  y = (yaxis - origin) * gl_Vertex.w;
  z = (zaxis - origin) * gl_Vertex.w;
  gl_FrontColor = gl_Color;
}
