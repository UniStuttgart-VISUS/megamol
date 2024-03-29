#version 110

#include "protein_cuda/protein_cuda/commondefines.glsl"

uniform vec4 viewAttr; // TODO: check fragment position if viewport starts not in (0, 0)

#ifndef CALC_CAM_SYS
uniform vec3 camIn;
uniform vec3 camUp;
uniform vec3 camRight;
#endif // CALC_CAM_SYS

#ifdef DEBUG
uniform vec2 circleAttr;
#endif // DEBUG

//
attribute vec2 inParams;
attribute vec4 quatC; // conjugate quaternion
//

// colors of cylinder
attribute vec3 color1;
attribute vec3 color2;

varying vec4 objPos;
varying vec4 camPos;
varying vec4 lightPos;
varying vec3 radz;

varying vec3 rotMatT0;
varying vec3 rotMatT1; // rotation matrix from the quaternion
varying vec3 rotMatT2;

#ifdef RETICLE
varying vec2 centerFragment;
#endif // RETICLE

void main(void) {
    const vec4 quatConst = vec4(1.0, -1.0, 0.5, 0.0);
    vec4 tmp, tmp1;

    // remove the sphere radius from the w coordinates to the rad varyings
    vec4 inPos = gl_Vertex;

    radz.x = inParams.x;
    radz.y = radz.x * radz.x;
    radz.z = inParams.y * 0.5;

    inPos.w = 1.0;


    // object pivot point in object space
    objPos = inPos; // no w-div needed, because w is 1.0 (Because I know)


    // orientation quaternion to inverse rotation matrix conversion
    // Begin: Holy code!
    tmp = quatC.xzyw * quatC.yxzw;                        // tmp <- (xy, xz, yz, ww)
    tmp1 = quatC * quatC.w;                                // tmp1 <- (xw, yw, zw, %)
    tmp1.w = -quatConst.z;                                // tmp1 <- (xw, yw, zw, -0.5)

    rotMatT0.xyz = tmp1.wzy * quatConst.xxy + tmp.wxy;    // matrix0 <- (ww-0.5, xy+zw, xz-yw, %)
    rotMatT0.x = quatC.x * quatC.x + rotMatT0.x;        // matrix0 <- (ww+x*x-0.5, xy+zw, xz-yw, %)
    rotMatT0 = rotMatT0 + rotMatT0;                     // matrix0 <- (2(ww+x*x)-1, 2(xy+zw), 2(xz-yw), %)

    rotMatT1.xyz = tmp1.zwx * quatConst.yxx + tmp.xwz;     // matrix1 <- (xy-zw, ww-0.5, yz+xw, %)
    rotMatT1.y = quatC.y * quatC.y + rotMatT1.y;         // matrix1 <- (xy-zw, ww+y*y-0.5, yz+xw, %)
    rotMatT1 = rotMatT1 + rotMatT1;                     // matrix1 <- (2(xy-zw), 2(ww+y*y)-1, 2(yz+xw), %)

    rotMatT2.xyz = tmp1.yxw * quatConst.xyx + tmp.yzw;     // matrix2 <- (xz+yw, yz-xw, ww-0.5, %)
    rotMatT2.z = quatC.z * quatC.z + rotMatT2.z;         // matrix2 <- (xz+yw, yz-xw, ww+zz-0.5, %)
    rotMatT2 = rotMatT2 + rotMatT2;                     // matrix2 <- (2(xz+yw), 2(yz-xw), 2(ww+zz)-1, %)
    // End: Holy code!


    // calculate cam position
    tmp = gl_ModelViewMatrixInverse[3]; // (C) by Christoph
    tmp.xyz -= objPos.xyz; // cam move
    camPos.xyz = rotMatT0 * tmp.x + rotMatT1 * tmp.y + rotMatT2 * tmp.z;


    // calculate light position in glyph space
    // USE THIS LINE TO GET POSITIONAL LIGHTING
    //lightPos = gl_ModelViewMatrixInverse * gl_LightSource[0].position - objPos; // note: w is bullshit now!
    // USE THIS LINE TO GET DIRECTIONAL LIGHTING
    lightPos = gl_ModelViewMatrixInverse * normalize( gl_LightSource[0].position);
    lightPos.xyz = rotMatT0 * lightPos.x + rotMatT1 * lightPos.y + rotMatT2 * lightPos.z;


    // send color to fragment shader
    gl_FrontColor = gl_Color;

    // calculate point sprite
    vec2 winHalf = 2.0 / viewAttr.zw; // window size

    // lumberjack approach
    vec4 pos, projPos;
    vec4 pX, pY, pZ, pOP;
    vec2 mins, maxs, pp;

#define CYL_HALF_LEN radz.z
#define CYL_RAD radz.x

    projPos.w = 0.0;

    //pos = vec4(0.0, 0.0, 0.0, 1.0);
    //projPos.x = dot(rotMatT0.xyz, pos.xyz); // rotate
    //projPos.y = dot(rotMatT1.xyz, pos.xyz);
    //projPos.z = dot(rotMatT2.xyz, pos.xyz);
    pos = objPos; // + projPos; // move
    pos.w = 1.0; // now we're in object space
    pOP = gl_ModelViewProjectionMatrix * pos;

    pos = vec4(1.0, 0.0, 0.0, 1.0);
    projPos.x = dot(rotMatT0.xyz, pos.xyz); // rotate
    projPos.y = dot(rotMatT1.xyz, pos.xyz);
    projPos.z = dot(rotMatT2.xyz, pos.xyz);
    pos = objPos + projPos; // move
    pos.w = 1.0; // now we're in object space
    projPos = gl_ModelViewProjectionMatrix * pos;
    pX = (projPos - pOP) * CYL_HALF_LEN;

    pos = vec4(0.0, 1.0, 0.0, 1.0);
    projPos.x = dot(rotMatT0.xyz, pos.xyz); // rotate
    projPos.y = dot(rotMatT1.xyz, pos.xyz);
    projPos.z = dot(rotMatT2.xyz, pos.xyz);
    pos = objPos + projPos; // move
    pos.w = 1.0; // now we're in object space
    projPos = gl_ModelViewProjectionMatrix * pos;
    pY = (projPos - pOP) * CYL_RAD;

    pos = vec4(0.0, 0.0, 1.0, 1.0);
    projPos.x = dot(rotMatT0.xyz, pos.xyz); // rotate
    projPos.y = dot(rotMatT1.xyz, pos.xyz);
    projPos.z = dot(rotMatT2.xyz, pos.xyz);
    pos = objPos + projPos; // move
    pos.w = 1.0; // now we're in object space
    projPos = gl_ModelViewProjectionMatrix * pos;
    pZ = (projPos - pOP) * CYL_RAD;

    // 8 corners of doom
    pos = pOP + pX;
    projPos = pos + pY + pZ;
    mins = maxs = projPos.xy / projPos.w;

    projPos = pos - pY + pZ;
    pp = projPos.xy / projPos.w;
    mins = min(mins, pp);
    maxs = max(maxs, pp);

    projPos = pos + pY - pZ;
    pp = projPos.xy / projPos.w;
    mins = min(mins, pp);
    maxs = max(maxs, pp);

    projPos = pos - pY - pZ;
    pp = projPos.xy / projPos.w;
    mins = min(mins, pp);
    maxs = max(maxs, pp);


    pos = pOP - pX;
    projPos = pos + pY + pZ;
    pp = projPos.xy / projPos.w;
    mins = min(mins, pp);
    maxs = max(maxs, pp);

    projPos = pos - pY + pZ;
    pp = projPos.xy / projPos.w;
    mins = min(mins, pp);
    maxs = max(maxs, pp);

    projPos = pos + pY - pZ;
    pp = projPos.xy / projPos.w;
    mins = min(mins, pp);
    maxs = max(maxs, pp);

    projPos = pos - pY - pZ;
    pp = projPos.xy / projPos.w;
    mins = min(mins, pp);
    maxs = max(maxs, pp);


    gl_Position = vec4((mins + maxs) * 0.5, 0.0, 1.0);
    maxs = (maxs - mins) * 0.5 * winHalf;
    gl_PointSize = max(maxs.x, maxs.y);


//    // oabb-approach
//    // calculate pOP, pX, pY, and pZ as above
//
//    pos = pOP + pX * circleAttr.x;
//    gl_Position = vec4(pos.xyz / pos.w, 1.0);
//
//    float l = camPos.y * camPos.y + camPos.z * camPos.z;
//    if (l < radz.y) { // radius-square
//        // camera is inside the cylinder
//        //  so choose nearest cap and we are done
//        gl_FrontSecondaryColor = vec4(0.5, 0.0, 0.0, 1.0);
//
//        // calculate 4 points of doom
//        pos = pOP + pX * sign(camPos.x);
//
//        projPos = pos + pY + pZ;
//        mins = maxs = projPos.xy / projPos.w;
//
//        projPos = pos - pY + pZ;
//        pp = projPos.xy / projPos.w;
//        mins = min(mins, pp);
//        maxs = max(maxs, pp);
//
//        projPos = pos + pY - pZ;
//        pp = projPos.xy / projPos.w;
//        mins = min(mins, pp);
//        maxs = max(maxs, pp);
//
//        projPos = pos - pY - pZ;
//        pp = projPos.xy / projPos.w;
//        mins = min(mins, pp);
//        maxs = max(maxs, pp);
//
//        gl_Position = vec4((mins + maxs) * 0.5, 0.0, 1.0);
//        maxs = (maxs - mins) * 0.5 * winHalf;
//        gl_PointSize = max(maxs.x, maxs.y);
//
//    } else {
//        // camera is outside the cylinder
//        //  so find touching planes
//        gl_FrontSecondaryColor = vec4(0.0, 0.5, 0.0, 1.0);
//
//        vec2 j = vec2(camPos.y, camPos.z);
//        float d = length(j);
//        j /= d;
//        vec2 m = vec2(j.y, -j.x);
//        float p = radz.y / d;
//        float q = d - p;
//        float h = sqrt(p * q);
//
//        vec2 i = j * p + m * h * circleAttr.x;
//
//        i /= radz.x;
//
////        pos = pOP + pX * circleAttr.x + pY * n.y + pZ * n.z;
////        pos = pOP + pX + pY * circleAttr.y + pZ * circleAttr.x;
//
//        float minCylX = -CYL_HALF_LEN + (((radz.x - p * sign(-CYL_HALF_LEN - camPos.x)) * (-CYL_HALF_LEN - camPos.x)) / (-d * sign(-CYL_HALF_LEN - camPos.x) + radz.x)); // tut net so ganz
//        float maxCylX =  CYL_HALF_LEN + (((radz.x + p * sign(CYL_HALF_LEN - camPos.x)) * (CYL_HALF_LEN - camPos.x)) / (d * sign(CYL_HALF_LEN - camPos.x) + radz.x)); // tut, keine Ahnung warum
//
//
//        float fX = mix(minCylX, maxCylX, (circleAttr.y + 1.0) * 0.5);
//
//        pX /= CYL_HALF_LEN;
//
//        pos = pOP + pX * fX + pY * i.x + pZ * i.y;
//
//        gl_Position = vec4(pos.xyz / pos.w, 1.0);
//
//    }


#ifdef RETICLE
    centerFragment = gl_Position.xy / gl_Position.w;
#endif // RETICLE

    // set colors
    gl_FrontColor.r = color1.r;
    gl_FrontColor.g = color1.g;
    gl_FrontColor.b = color1.b;

    gl_FrontSecondaryColor.r = color2.r;
    gl_FrontSecondaryColor.g = color2.g;
    gl_FrontSecondaryColor.b = color2.b;

}
