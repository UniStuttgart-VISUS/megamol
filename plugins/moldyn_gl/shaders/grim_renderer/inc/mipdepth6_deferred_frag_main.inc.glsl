void main(void) {
  ivec2 fc = ivec2(gl_FragCoord.xy);

  // depths (obj-spc-positions) of neighbor fragments
  vec4 c22 = texelFetch2D(pos, fc + ivec2(1, 1), 0);
  vec4 c02 = texelFetch2D(pos, fc + ivec2(-1, 1), 0);
  vec4 c00 = texelFetch2D(pos, fc + ivec2(-1, -1), 0);
  vec4 c20 = texelFetch2D(pos, fc + ivec2(1, -1), 0);

  vec4 v12 = texelFetch2D(pos, fc + ivec2(0, 1), 0);
  vec4 v01 = texelFetch2D(pos, fc + ivec2(-1, 0), 0);
  vec4 v10 = texelFetch2D(pos, fc + ivec2(0, -1), 0);
  vec4 v21 = texelFetch2D(pos, fc + ivec2(1, 0), 0);

  vec4 v11 = texelFetch2D(pos, fc, 0);

  // glyph normal and colour
  vec4 objn = texelFetch2D(normal, fc, 0);
  vec4 col = texelFetch2D(colour, fc, 0);

  if (col.a < 0.01) {
    discard;
  }

  // background handling
  if (c00.w < 0.5) { c00 = v11; }
  if (c02.w < 0.5) { c02 = v11; }
  if (c20.w < 0.5) { c20 = v11; }
  if (c22.w < 0.5) { c22 = v11; }
  if (v01.w < 0.5) { v01 = v11; }
  if (v10.w < 0.5) { v10 = v11; }
  if (v12.w < 0.5) { v12 = v11; }
  if (v21.w < 0.5) { v21 = v11; }

  /*// approach 1: central differences
  vec3 c12 = 2.0 * v12 - 0.5 * (c02 + c22);
  vec3 c01 = 2.0 * v01 - 0.5 * (c02 + c00);
  vec3 c10 = 2.0 * v10 - 0.5 * (c20 + c00);
  vec3 c21 = 2.0 * v21 - 0.5 * (c20 + c22);

  vec3 c11 = 4.0 * v11 - (v10 + v12) - 0.5 * (c01 + c21);

  vec3 xu = 0.5 * c11 + 0.25 * (c01 + c21);
  vec3 xv = 0.5 * c11 + 0.25 * (c10 + c12);

  vec3 dy = 0.5 * (v10 + xu) - 0.5 * (v12 + xu);
  vec3 dx = 0.5 * (v01 + xv) - 0.5 * (v21 + xv);
  */

  // approach 2: shifted patch
  vec3 c12 = v12.xyz;
  vec3 c01 = v01.xyz;
  vec3 c10 = v10.xyz;
  vec3 c21 = v21.xyz;
  vec3 c11 = v11.xyz;

  // evaluate at (0.5, 0.5)
  vec3 b0 = (c00.xyz + 2.0 * c01 + c02.xyz) * 0.25;
  vec3 b1 = (c10 + 2.0 * c11 + c12) * 0.25;
  vec3 b2 = (c20.xyz + 2.0 * c21 + c22.xyz) * 0.25;
  vec3 dx = 0.5 * (b0 + b1) - 0.5 * (b1 + b2);
  b0 = (c00.xyz + 2.0 * c10 + c20.xyz) * 0.25;
  b1 = (c01 + 2.0 * c11 + c21) * 0.25;
  b2 = (c02.xyz + 2.0 * c12 + c22.xyz) * 0.25;
  vec3 dy = 0.5 * (b0 + b1) - 0.5 * (b1 + b2);

  // normal evaluation (approach 1 + 2)
  vec3 n = normalize(cross(dx, dy));

  // blend between glyph and calculated normal
  //col = vec3(0.0, 0.0, 1.0);
  if (objn.w > 0.99) {
    n = objn.xyz;
    //col = vec3(1.0, 0.0, 0.0);
  } else if (objn.w > 0.01) {
    n = n * (1.0 - objn.w) + objn.xyz * objn.w;
    //col = vec3(objn.w, 0.0, 1.0 - objn.w);
  }

  vec3 ray2 = normalize(interpolRay);

  // lighting (approach 1 + 2)
  if (v11.w < 0.5) {
    gl_FragColor = vec4(1.0f);
  } else {
    gl_FragColor = vec4(
      (v11.w > 0.5) ? LocalLighting(ray2, n, lightDir.xyz, col.rgb) : col.rgb,
      1.0);
  }
}
