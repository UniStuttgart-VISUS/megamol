#ifdef SMALL_SPRITE_LIGHTING
    // for normal crowbaring on very small sprites
    lightDir.w = (clamp(gl_PointSize, 1.0, 5.0) - 1.0) / 4.0;
#else
    lightDir.w = 1.0;
#endif // SMALL_SPRITE_LIGHTING

#ifdef RETICLE
    centerFragment = gl_Position.xy / gl_Position.w;
#endif // RETICLE

    gl_PointSize = 1.0;

    // depp texture coordinates ..
    vec2 dtc = gl_Position.xy + vec2(1.0);
    dtc /= vec2(viewAttr.z, viewAttr.w);
    // .. now in 'viewport' coordinates

//#define DEPTHMIP_WIDTH depthTexParams.x
//#define DEPTHMIP_HEIGHT depthTexParams.y
//#define DEPTHMIP_MAXLEVEL depthTexParams.z

    int miplevel = min(max((int(log2(gl_PointSize))), 1), DEPTHMIP_MAXLEVEL);
    float exp = exp2(float(miplevel));

    dtc /= exp;
    ivec2 idtc = ivec2(dtc - vec2(0.5)); // because cast to "ivec2" performs a "round" as sfx !!! WTF !!!
    // now in relative coordinate of the mip level
    idtc.x += int(float(DEPTHMIP_WIDTH * (1.0 - 2.0 / exp)));
    idtc.y += DEPTHMIP_HEIGHT;

    vec4 depth1 = texelFetch2D(depthTex, idtc, 0);
    vec4 depth2 = texelFetch2D(depthTex, idtc + ivec2(1, 0), 0);
    vec4 depth3 = texelFetch2D(depthTex, idtc + ivec2(0, 1), 0);
    vec4 depth4 = texelFetch2D(depthTex, idtc + ivec2(1, 1), 0);

    float depth = max(max(depth1.x, depth2.x), max(depth3.x, depth4.x));

    vec4 depthPos; // ass of sphere in object space
    vec3 v = objPos.xyz - cam_pos.xyz;
    //float l = length(v);
    //v *= (l - rad) / l;
    depthPos.xyz = cam_pos.xyz + v;
    depthPos.w = 1.0;

    depthPos = mvp * depthPos;
    depthPos.xyz /= depthPos.w;

    depthPos.z -= gl_DepthRange.near;
    depthPos.z /= gl_DepthRange.diff;
    depthPos.z += 1.0;
    depthPos.z *= 0.5;

    depth -= gl_DepthRange.near;
    depth /= gl_DepthRange.diff;

    if (depthPos.z > depth) {
        gl_Position.w = 0.0;
    }

    // gl_FrontColor = vec4(abs(depth - depthPos.z) * 20.0, 0.0, 0.0, 1.0);
    // gl_FrontColor = vec4(abs(depthPos.z));

    // gl_Position.w = 0.0;
}
