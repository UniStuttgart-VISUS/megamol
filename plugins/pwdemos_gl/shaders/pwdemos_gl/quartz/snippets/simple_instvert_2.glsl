uniform sampler2D partTex;
uniform int partTexWidth;
uniform vec4 bboxMin;
uniform vec4 bboxSize;

void main() {
  int id = 2 * gl_InstanceID;
  ivec2 texC = ivec2(mod(id, partTexWidth), floor(id / partTexWidth));
  vec4 posR = texelFetch2D(partTex, texC, 0);
  posR *= bboxSize;
  posR += bboxMin;
  vec4 quat = texelFetch2D(partTex, texC + ivec2(1, 0), 0);
  quat *= bboxSize;
  quat += bboxMin;

  vec3 normal = gl_Normal;
  normal = ((2.0 * ((dot(quat.xyz, normal) * quat.xyz) + (quat.w * cross(quat.xyz, normal)))) + (((quat.w * quat.w) - dot(quat.xyz, quat.xyz)) * normal));

  gl_FrontColor = DirectLight(normal.xyz, gl_Color);

  vec3 vec = gl_Vertex.xyz * posR.w;
  vec = ((2.0 * ((dot(quat.xyz, vec) * quat.xyz) + (quat.w * cross(quat.xyz, vec)))) + (((quat.w * quat.w) - dot(quat.xyz, quat.xyz)) * vec));
  vec += posR.xyz;
  vec4 pos = gl_ModelViewProjectionMatrix * vec4(vec, 1.0);

  gl_Position = pos / pos.w;
}
