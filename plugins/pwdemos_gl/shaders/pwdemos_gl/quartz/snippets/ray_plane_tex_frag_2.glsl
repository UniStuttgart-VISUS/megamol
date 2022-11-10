uniform vec4 viewAttr;
uniform vec3 camX;
uniform vec3 camY;
uniform vec3 camZ;
uniform float planeZ;
uniform vec3 posoffset;
uniform vec3 bboxmin;
uniform vec3 bboxmax;
uniform sampler2D typeData;
uniform ivec2 typeInfo; // type-row, num-faces

in vec3 objPos;
in vec4 quat;
in float rad;

void main() {
  // transform fragment coordinates from window coordinates to view coordinates.
  vec4 col = gl_Color;
  vec4 coord = gl_FragCoord * vec4(viewAttr.z, viewAttr.w, 2.0, 0.0) 
      + vec4(-1.0, -1.0, -1.0, 1.0);
  coord = gl_ModelViewProjectionMatrixInverse * coord;
  coord /= coord.w;
  coord.xyz = (camX * coord.x) + (camY * coord.y) - (camZ * planeZ);

  if (bboxmin != bboxmax) {
    if (any(lessThan(coord.xyz, bboxmin))
        || any(greaterThan(coord.xyz, bboxmax))) {
      //col = vec4(1.0, 0.0, 0.0, 1.0);
      discard;
    }
  }

  coord.xyz -= objPos;
  coord.xyz = ((2.0 * ((dot(quat.xyz, coord.xyz) * quat.xyz) + (quat.w * cross(quat.xyz, coord.xyz)))) + (((quat.w * quat.w) - dot(quat.xyz, quat.xyz)) * coord.xyz));
  coord.xyz /= rad;
