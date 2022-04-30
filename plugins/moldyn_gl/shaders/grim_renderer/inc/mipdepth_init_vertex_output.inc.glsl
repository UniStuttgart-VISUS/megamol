    vec4 depthPos; // ass of sphere in object space

    vec3 v = objPos.xyz - cam_pos.xyz;
    float l = length(v);
    v *= (l + rad) / l;
    depthPos.xyz = cam_pos.xyz + v;
    depthPos.w = 1.0;

    depthPos = mvp * depthPos;
    depthPos.xyz /= depthPos.w;

    gl_Position = vec4((mins + maxs) * 0.5, depthPos.z, 1.0);
    maxs = (maxs - mins) * 0.5 * winHalf;
    gl_PointSize = abs(max(maxs.x, maxs.y)) + 0.5;
}
