  // bboxclipping
  bool wsNormalSet = false;
  if (bboxmin != bboxmax) {
    vec3 rayWS = normalize(fragCoordWS - camPosWS);
    vec3 posFWS = camPosWS + (rayWS * lams.x);
    vec3 posBWS = camPosWS + (rayWS * lams.y);

    vec2 blams = vec2(0.0, 1000000.0);
    vec3 bn;
    float bl;
    if (rayWS.x != 0.0) {
      bl = (bboxmax.x - camPosWS.x) / rayWS.x;
      if (camPosWS.x > bboxmax.x) { // maxX hit from outside
        if (bl > blams.x) {
          blams.x = bl;
          bn = vec3(1.0, 0.0, 0.0);
        }
      } else if (bl > 0.0) {
        blams.y = min(blams.y, bl);
      }
      bl = (bboxmin.x - camPosWS.x) / rayWS.x;
      if (camPosWS.x < bboxmin.x) { // minX hit from outside
        if (bl > blams.x) {
          blams.x = bl;
          bn = vec3(-1.0, 0.0, 0.0);
        }
      } else if (bl > 0.0) {
        blams.y = min(blams.y, bl);
      }
    }
    if (rayWS.y != 0.0) {
      bl = (bboxmax.y - camPosWS.y) / rayWS.y;
      if (camPosWS.y > bboxmax.y) { // maxY hit from outside
        if (bl > blams.x) {
          blams.x = bl;
          bn = vec3(0.0, 1.0, 0.0);
        }
      } else if (bl > 0.0) {
        blams.y = min(blams.y, bl);
      }
      bl = (bboxmin.y - camPosWS.y) / rayWS.y;
      if (camPosWS.y < bboxmin.y) { // minY hit from outside
        if (bl > blams.x) {
          blams.x = bl;
          bn = vec3(0.0, -1.0, 0.0);
        }
      } else if (bl > 0.0) {
        blams.y = min(blams.y, bl);
      }
    }
    if (rayWS.z != 0.0) {
      bl = (bboxmax.z - camPosWS.z) / rayWS.z;
      if (camPosWS.z > bboxmax.z) { // maxZ hit from outside
        if (bl > blams.x) {
          blams.x = bl;
          bn = vec3(0.0, 0.0, 1.0);
        }
      } else if (bl > 0.0) {
        blams.y = min(blams.y, bl);
      }
      bl = (bboxmin.z - camPosWS.z) / rayWS.z;
      if (camPosWS.z < bboxmin.z) { // minZ hit from outside
        if (bl > blams.x) {
          blams.x = bl;
          bn = vec3(0.0, 0.0, -1.0);
        }
      } else if (bl > 0.0) {
        blams.y = min(blams.y, bl);
      }
    }
    if (blams.y < blams.x) discard;

    vec3 faceposWS = camPosWS + (rayWS * blams.x);
    const float EPSILON = 0.001;
    if (any(greaterThan(faceposWS, bboxmax + vec3(EPSILON))) || any(lessThan(faceposWS, bboxmin - vec3(EPSILON)))) discard;
    if (blams.y < lambda) discard;
    if (blams.x > lambda) {
      // OLD col = gl_Color; // avoid error close to the clipping plane
      col = color; // avoid error close to the clipping plane
      normal = bn;
      wsNormalSet = true;
      lambda = blams.x;
      if (lambda > lams.y) discard;
    }
  }
