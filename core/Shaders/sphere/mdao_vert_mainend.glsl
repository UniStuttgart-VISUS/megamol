    // clipping
    float od = inClipDat.w - 1.0;
    if (any(notEqual(inClipDat.xyz, vec3(0, 0, 0)))) {
        od = dot(vsObjPos.xyz, inClipDat.xyz) - vsRad;
    }
	
    gl_Position = vec4((mins + maxs) * 0.5, projPos.z, (od > inClipDat.w) ? 0.0 : 1.0);
    maxs = (maxs - mins) * 0.5 * winHalf;
    gl_PointSize = max(maxs.x, maxs.y) + 0.5;
}		