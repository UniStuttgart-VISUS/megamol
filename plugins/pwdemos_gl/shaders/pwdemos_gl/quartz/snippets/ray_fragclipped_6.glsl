    vec4 iquat = quat * vec4(-1.0, -1.0, -1.0, 1.0);

    // transform normal into object space to work with the correct half-way-vector (ugly)
    if (!wsNormalSet) {
      normal.xyz = ((2.0 * ((dot(iquat.xyz, normal.xyz) * iquat.xyz) + (iquat.w * cross(iquat.xyz, normal.xyz)))) + (((iquat.w * iquat.w) - dot(iquat.xyz, iquat.xyz)) * normal.xyz));
    }

    outColor += DirectLight(normal, col);

    // fragment position
    // ... glyph-space
    coord = camPos;
    coord.xyz += ray * lambda;

    // ... object-space
    coord.xyz = ((2.0 * ((dot(iquat.xyz, coord.xyz) * iquat.xyz) + (iquat.w * cross(iquat.xyz, coord.xyz)))) + (((iquat.w * iquat.w) - dot(iquat.xyz, iquat.xyz)) * coord.xyz));
    coord += objPos;

    // ... image-space
    coord.w = 1.0;
    vec2 depth;
    // OLD depth.x = dot(gl_ModelViewProjectionMatrixTranspose[2], coord);
    depth.x = dot(ModelViewProjectionMatrixTranspose[2], coord);
    // OLD depth.y = dot(gl_ModelViewProjectionMatrixTranspose[3], coord);
    depth.y = dot(ModelViewProjectionMatrixTranspose[3], coord);
    gl_FragDepth = ((depth.x / depth.y) + 1.0) * 0.5;
}
