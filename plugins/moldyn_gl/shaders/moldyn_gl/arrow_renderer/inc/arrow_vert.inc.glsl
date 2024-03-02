attribute float colIdx;
attribute vec4 pos_attrib;
attribute vec4 col_attrib;
attribute vec3 dir;
attribute uint flags;

void main(void) {
    // remove the sphere radius from the w coordinates to the rad varyings
    vec4 inPos = pos_attrib;
    rad.x = (CONSTRAD < -0.5) ? inPos.w : CONSTRAD;
    inPos.w = 1.0;
    float cid = MAX_COLV - MIN_COLV;
    if (cid < 0.000001) {
        gl_FrontColor = col_attrib;
    } else {
        cid = (colIdx - MIN_COLV) / cid;
        cid = clamp(cid, 0.0, 1.0);

        cid *= (1.0 - 1.0 / COLTAB_SIZE);
        cid += 0.5 / COLTAB_SIZE;

        gl_FrontColor = texture1D(colTab, cid);
    }

    //const vec4 quatConst = vec4(1.0, -1.0, 0.5, 0.0);
    vec4 tmp, tmp1;

    vec3 fromPos = pos_attrib.xyz + (dir * 0.5) * lengthScale;
    vec3 toPos = pos_attrib.xyz - (dir * 0.5) * lengthScale;

    inPos = vec4(pos_attrib.xyz, 1.0);

    rad.y = rad.x * rad.x;
    //rad.z = 1.0; // half cylinder length

    rotMatT0 = toPos - fromPos;
    rad.z = length(rotMatT0);
    rotMatT0 *= 1.0 / rad.z;
    rad.z *= 0.5;

    rotMatT2 = ((rotMatT0.x > 0.9) || (rotMatT0.x < -0.9)) ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0); // normal on tmp
    rotMatT1 = cross(rotMatT0, rotMatT2);
    rotMatT1 = normalize(rotMatT1);
    rotMatT2 = cross(rotMatT1, rotMatT0);

    vec3 ttmp1 = rotMatT0;
    vec3 ttmp2 = rotMatT1;
    vec3 ttmp3 = rotMatT2;

    rotMatT0 = vec3(ttmp1.x, ttmp2.x, ttmp3.x);
    rotMatT1 = vec3(ttmp1.y, ttmp2.y, ttmp3.y);
    rotMatT2 = vec3(ttmp1.z, ttmp2.z, ttmp3.z);

    // vec4 pos = inPos;
    // pos.xyz += cor.x * rotMatT1 + cor.y * rotMatT2;
    // pos = MVP * pos;
    // pos /= pos.w;
    // gl_Position = pos;

    // calculate cam position
    tmp = MVinv[3]; // (C) by Christoph
    tmp.xyz -= inPos.xyz; // cam move
    camPos.xyz = rotMatT0 * tmp.x + rotMatT1 * tmp.y + rotMatT2 * tmp.z;
    camPos.w = 1.0;

    // calculate light position in glyph space
    //lightPos = MVinv * gl_LightSource[0].position - inPos; // note: w is bullshit now!
    //lightPos.xyz = rotMatT0 * lightPos.x + rotMatT1 * lightPos.y + rotMatT2 * lightPos.z;
    //lightPos = MVtransp * gl_LightSource[0].position; // transpose of inverse inverse -> directional light

    rotLightDir.xyz = rotMatT0 * lightDir.x + rotMatT1 * lightDir.y + rotMatT2 * lightDir.z;
    rotLightDir.w = 1.0f;

    objPos = inPos;

    // calculate point sprite
    vec2 winHalf = 2.0 / viewAttr.zw; // window size

    // lumberjack approach
    vec4 pos, projPos;
    vec4 pX, pY, pZ, pOP;
    vec2 mins, maxs, pp;

#define CYL_HALF_LEN rad.z
#define CYL_RAD rad.x

    projPos.w = 0.0;

    //pos = vec4(0.0, 0.0, 0.0, 1.0);
    //projPos.x = dot(rotMatT0.xyz, pos.xyz); // rotate
    //projPos.y = dot(rotMatT1.xyz, pos.xyz);
    //projPos.z = dot(rotMatT2.xyz, pos.xyz);
    pos = objPos; // + projPos; // move
    pos.w = 1.0; // now we're in object space
    pOP = MVP * pos;

    pos = vec4(1.0, 0.0, 0.0, 1.0);
    projPos.x = dot(rotMatT0.xyz, pos.xyz); // rotate
    projPos.y = dot(rotMatT1.xyz, pos.xyz);
    projPos.z = dot(rotMatT2.xyz, pos.xyz);
    pos = objPos + projPos; // move
    pos.w = 1.0; // now we're in object space
    projPos = MVP * pos;
    pX = (projPos - pOP) * CYL_HALF_LEN;

    pos = vec4(0.0, 1.0, 0.0, 1.0);
    projPos.x = dot(rotMatT0.xyz, pos.xyz); // rotate
    projPos.y = dot(rotMatT1.xyz, pos.xyz);
    projPos.z = dot(rotMatT2.xyz, pos.xyz);
    pos = objPos + projPos; // move
    pos.w = 1.0; // now we're in object space
    projPos = MVP * pos;
    pY = (projPos - pOP) * CYL_RAD;

    pos = vec4(0.0, 0.0, 1.0, 1.0);
    projPos.x = dot(rotMatT0.xyz, pos.xyz); // rotate
    projPos.y = dot(rotMatT1.xyz, pos.xyz);
    projPos.z = dot(rotMatT2.xyz, pos.xyz);
    pos = objPos + projPos; // move
    pos.w = 1.0; // now we're in object space
    projPos = MVP * pos;
    pZ = (projPos - pOP) * CYL_RAD;

    // Clipping
    float od = clipDat.w - 1.0;
    if (any(notEqual(clipDat.xyz, vec3(0, 0, 0)))) {
        od = dot(objPos.xyz, clipDat.xyz) - rad.x;
    }

    float l = camPos.y * camPos.y + camPos.z * camPos.z;
    vec4 p1, p2, p3, p4;
    if (l < rad.y) { // radius-square
        // camera is inside the cylinder
        //  so choose nearest cap and we are done
        gl_FrontSecondaryColor = vec4(0.5, 0.0, 0.0, 1.0);

        // calculate 4 points of doom
        pos = pOP + pX * sign(camPos.x);

        p1 = pos + pY + pZ;
        p2 = pos - pY + pZ;
        p3 = pos + pY - pZ;
        p4 = pos - pY - pZ;
        /*
        projPos = pos + cor.x * pY + cor.y * pZ;
        gl_Position = projPos / projPos.w;
        */

    } else {
        // camera is outside the cylinder
        //  so find touching planes
        gl_FrontSecondaryColor = vec4(0.0, 0.5, 0.0, 1.0);

        vec2 j = vec2(camPos.y, camPos.z);
        float d = length(j);
        j /= d;
        vec2 m = vec2(j.y, -j.x);
        float p = rad.y / d;
        float q = d - p;
        float h = sqrt(p * q);

        float minCylX = -CYL_HALF_LEN + (((rad.x - p * sign(-CYL_HALF_LEN - camPos.x)) * (-CYL_HALF_LEN - camPos.x)) / (-d * sign(-CYL_HALF_LEN - camPos.x) + rad.x));
        float maxCylX =  CYL_HALF_LEN + (((rad.x + p * sign(CYL_HALF_LEN - camPos.x)) * (CYL_HALF_LEN - camPos.x)) / (d * sign(CYL_HALF_LEN - camPos.x) + rad.x));

        //if (cor.y > 0.0) minCylX = maxCylX;

        pX /= CYL_HALF_LEN;

        vec2 i1 = j * p + m * h;
        i1 /= rad.x;
        vec2 i2 = j * p - m * h;
        i2 /= rad.x;

        //projPos = pOP + pX * minCylX + pY * i.x + pZ * i.y;
        //gl_Position = projPos / projPos.w;
        p1 = pOP + pX * minCylX + pY * i1.x + pZ * i1.y;
        p2 = pOP + pX * maxCylX + pY * i1.x + pZ * i1.y;
        p3 = pOP + pX * minCylX + pY * i2.x + pZ * i2.y;
        p4 = pOP + pX * maxCylX + pY * i2.x + pZ * i2.y;
    }

    p1 /= p1.w;
    p2 /= p2.w;
    p3 /= p3.w;
    p4 /= p4.w;

    vec2 minP = vec2(
      min(min(p1.x, p2.x), min(p3.x, p4.x)),
      min(min(p1.y, p2.y), min(p3.y, p4.y)));
    vec2 maxP = vec2(
      max(max(p1.x, p2.x), max(p3.x, p4.x)),
      max(max(p1.y, p2.y), max(p3.y, p4.y)));

    if (bitflag_test(flags, FLAG_SELECTED, FLAG_SELECTED)) {
        gl_FrontColor = vec4(1.0, 0.0, 0.0, 1.0);
    }

    discardFrag = uint(0);
    if ((bool(flagsAvailable) && !bitflag_isVisible(flags))) {
        discardFrag = uint(1);
    }

    if (discardFrag == 0u) {
        float w = (od > clipDat.w) ? 0.0 : 1.0;
        w = ( CYL_HALF_LEN < lengthFilter * 0.5) ? 0.0 : w;
        gl_Position = vec4((minP + maxP) * 0.5, 0.5, w);
        maxP = (maxP - minP) * 0.5 * winHalf;
        gl_PointSize = max(maxP.x, maxP.y) + 0.5;
    } else {
    //if (bitflag_test(flags, FLAG_FILTERED, FLAG_FILTERED)) {
        gl_Position = vec4(0.0, 0.0, -1.0, 0.0);
        gl_PointSize = 1.0;
    }
}
