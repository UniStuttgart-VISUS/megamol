    gl_Position = vec4((mins + maxs) * 0.5, 0.0, (od > clipDat.w) ? 0.0 : 1.0);
    maxs = (maxs - mins) * 0.5 * winHalf;
    gl_PointSize = max(maxs.x, maxs.y) + 0.5;

    if (attenuateSubpixel == 1) {
        effectiveDiameter = gl_PointSize;
    } else {
        effectiveDiameter = 1.0;
    }

    //vec4 projPos = gl_MVPMatrix * vec4(objPos.xyz, 1.0);
    //projPos /= projPos.w;
    //gl_Position = projPos;
    //float camDist = sqrt(dot(camPos.xyz, camPos.xyz));
    //gl_PointSize = max((rad / camDist) * zNear, 1.0);

}