#version 430

#include "protein_gl/simplemolecule/sm_common_defines.glsl"
#include "protein_gl/simplemolecule/sm_common_input_vert.glsl"

//attribute vec2 inParams;
//attribute vec4 quatC; // conjugate quaternion

out vec3 radz;

out vec3 rotMatT0;
out vec3 rotMatT1; // rotation matrix from the quaternion
out vec3 rotMatT2;

void main(void) {
    const vec4 quatConst = vec4(1.0, -1.0, 0.5, 0.0);
    vec4 tmp, tmp1;

    // remove the sphere radius from the w coordinates to the rad varyings
    vec4 inPos = vert_position;

    radz.x = cyl_params.x;
    radz.y = radz.x * radz.x;
    radz.z = cyl_params.y * 0.5;

    inPos.w = 1.0;

    
    // object pivot point in object space    
    objPos = inPos; // no w-div needed, because w is 1.0 (Because I know)


    // orientation quaternion to inverse rotation matrix conversion
    // Begin: Holy code!
    tmp = cyl_quat.xzyw * cyl_quat.yxzw;                        // tmp <- (xy, xz, yz, ww)
    tmp1 = cyl_quat * cyl_quat.w;                                // tmp1 <- (xw, yw, zw, %)
    tmp1.w = -quatConst.z;                                // tmp1 <- (xw, yw, zw, -0.5)

    rotMatT0.xyz = tmp1.wzy * quatConst.xxy + tmp.wxy;    // matrix0 <- (ww-0.5, xy+zw, xz-yw, %)
    rotMatT0.x = cyl_quat.x * cyl_quat.x + rotMatT0.x;        // matrix0 <- (ww+x*x-0.5, xy+zw, xz-yw, %)
    rotMatT0 = rotMatT0 + rotMatT0;                     // matrix0 <- (2(ww+x*x)-1, 2(xy+zw), 2(xz-yw), %)

    rotMatT1.xyz = tmp1.zwx * quatConst.yxx + tmp.xwz;     // matrix1 <- (xy-zw, ww-0.5, yz+xw, %)
    rotMatT1.y = cyl_quat.y * cyl_quat.y + rotMatT1.y;         // matrix1 <- (xy-zw, ww+y*y-0.5, yz+xw, %)
    rotMatT1 = rotMatT1 + rotMatT1;                     // matrix1 <- (2(xy-zw), 2(ww+y*y)-1, 2(yz+xw), %)

    rotMatT2.xyz = tmp1.yxw * quatConst.xyx + tmp.yzw;     // matrix2 <- (xz+yw, yz-xw, ww-0.5, %)
    rotMatT2.z = cyl_quat.z * cyl_quat.z + rotMatT2.z;         // matrix2 <- (xz+yw, yz-xw, ww+zz-0.5, %)
    rotMatT2 = rotMatT2 + rotMatT2;                     // matrix2 <- (2(xz+yw), 2(yz-xw), 2(ww+zz)-1, %)    
    // End: Holy code!

    // calculate cam position
    tmp = MVinv[3]; // (C) by Christoph
    tmp.xyz -= objPos.xyz; // cam move
    camPos.xyz = rotMatT0 * tmp.x + rotMatT1 * tmp.y + rotMatT2 * tmp.z; 

    // send color to fragment shader
    move_color = vec4(vert_color, 1);

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
    
    // set colors
    move_color.r = cyl_color1.r;
    move_color.g = cyl_color1.g;
    move_color.b = cyl_color1.b;

    move_color2.r = cyl_color2.r;
    move_color2.g = cyl_color2.g;
    move_color2.b = cyl_color2.b;

    if(applyFiltering) {
        if(vert_filter == 0) {
            gl_PointSize = 0;
        }
    }
}
