  vec2 ll;
  vec2 lams = vec2(-1000000.0, 1000000.0);
  for (int i = 0; i < typeInfo.y; i++) {
    vec4 face = texelFetch2D(typeData, ivec2(i, typeInfo.x), 0);

    ll = planeCast(face.xyz, face.w * rad, ray);

    if (ll.y < 0.0) {
      // hit from front
      if (lams.x < ll.x) {
        lams.x = ll.x;
        normal = face.xyz;
      }
    }
    if (ll.y > 0.0) {
      // hit from behind
      if (lams.y > ll.x) {
        lams.y = ll.x;
      }
    }
  }

  if (lams.y < lams.x) discard;
  lambda = lams.x;
