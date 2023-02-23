    // clipping plane action
    if (clipping != vec4(0.0)) {
      ll = planeCast(clipping.xyz, clipping.w, ray);
      if (ll.y < 0.0) {
        // hit from front
        if (lams.x < ll.x) {
          lams.x = ll.x;
          normal = clipping.xyz;
          col = vec4(clipcol, 1.0);
// DEBUG: positions for debugging in clipplane-space
//col.xyz = camPos.xyz + ray * lams.x; // glyph space coordinates
//vec4 iquat = quat * vec4(-1.0, -1.0, -1.0, 1.0);
//col.xyz = ((2.0 * ((dot(iquat.xyz, col.xyz) * iquat.xyz) + (iquat.w * cross(iquat.xyz, col.xyz)))) + (((iquat.w * iquat.w) - dot(iquat.xyz, iquat.xyz)) * col.xyz));
//col.xyz += objPos.xyz;  // world space
//col.xyz /= 255.0;
        }
      }
      if (ll.y > 0.0) {
        // hit from behind
        if (lams.y > ll.x) {
          lams.y = ll.x;
        }
      }
      if (lams.y < lams.x) discard;
      lambda = lams.x;
    }
