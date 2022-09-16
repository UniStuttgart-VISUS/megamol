void main() {
  vec4 quat = gl_MultiTexCoord1;
  vec3 normal = gl_Normal;
  normal = ((2.0 * ((dot(quat.xyz, normal) * quat.xyz) + (quat.w * cross(quat.xyz, normal)))) + (((quat.w * quat.w) - dot(quat.xyz, quat.xyz)) * normal));
  gl_FrontColor = DirectLight(normal.xyz, gl_Color);
  vec3 vec = gl_Vertex.xyz * gl_MultiTexCoord0.w;
  vec = ((2.0 * ((dot(quat.xyz, vec) * quat.xyz) + (quat.w * cross(quat.xyz, vec)))) + (((quat.w * quat.w) - dot(quat.xyz, quat.xyz)) * vec));
  vec += gl_MultiTexCoord0.xyz;
  vec4 pos = gl_ModelViewProjectionMatrix * vec4(vec, 1.0);
  gl_Position = pos / pos.w;
}
