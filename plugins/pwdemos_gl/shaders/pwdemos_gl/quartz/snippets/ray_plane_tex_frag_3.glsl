
  float d = 0.0;
  for (int i = 0; i < typeInfo.y; i++) {
    vec4 f = texelFetch2D(typeData, ivec2(i, typeInfo.x), 0);
    d = max(d, dot(f.xyz, coord.xyz) / f.w);
  }
  if (d > 1.0) discard;
