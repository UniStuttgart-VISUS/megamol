    vec3 aoFac = vec3(
      (normal.x > 0.0) ? aoPos.x : aoNeg.x,
      (normal.y > 0.0) ? aoPos.y : aoNeg.y,
      (normal.z > 0.0) ? aoPos.z : aoNeg.z);
    aoFac.x *= (normal.x * normal.x);
    aoFac.y *= (normal.y * normal.y);
    aoFac.z *= (normal.z * normal.z);
    float aoFactor = aoFac.x + aoFac.y + aoFac.z;
