// @param faceNormal The face normal vector in glyph space
// @param faceDist Face distance from origin in glyph space
// @param ray The ray vector in glyph space
// @uses camPos The camera position in glyph space
vec2 planeCast(vec3 faceNormal, float faceDist, vec3 ray) {
  float hitdir = dot(ray, faceNormal);
  float lambda = (faceDist - dot(camPos.xyz, faceNormal)) / hitdir;
  return vec2(lambda, hitdir);
}
