#version 430

#include "protein_gl/simplemolecule/sm_common_defines.glsl"
#include "protein_gl/moleculeses/mses_common_defines.glsl"

layout (location = 0) in vec4 vert_position;
layout (location = 1) in vec4 vert_color;
layout (location = 2) in vec3 torus_params;
layout (location = 3) in vec4 torus_quat;
layout (location = 4) in vec4 torus_sphere;
layout (location = 5) in vec3 torus_cuttingplane;

out vec4 objPos;
out vec4 camPos;

out vec4 radii;
out vec4 visibilitySphere;

out vec3 rotMatT0;
out vec3 rotMatT1; // rotation matrix from the quaternion
out vec3 rotMatT2;

out float maxAngle;

out vec4 colors;
out vec3 cuttingPlane;

void main(void) {
    const vec4 quatConst = vec4(1.0, -1.0, 0.5, 0.0);
    vec4 tmp, tmp1;

    // remove the sphere radius from the w coordinates to the rad varyings
    vec4 inPos = vert_position;

    radii.x = torus_params.x;
    radii.y = radii.x * radii.x;
    radii.z = torus_params.y;
    radii.w =  radii.z * radii.z;
    
    maxAngle = torus_params.z;
    
    colors = vert_color;
    
    inPos.w = 1.0;
    
    // object pivot point in object space    
    objPos = inPos; // no w-div needed, because w is 1.0 (Because I know)

    // orientation quaternion to inverse rotation matrix conversion
    // Begin: Holy code!
    tmp = torus_quat.xzyw * torus_quat.yxzw;                        // tmp <- (xy, xz, yz, ww)
    tmp1 = torus_quat * torus_quat.w;                                    // tmp1 <- (xw, yw, zw, %)
    tmp1.w = -quatConst.z;                                // tmp1 <- (xw, yw, zw, -0.5)

    rotMatT0.xyz = tmp1.wzy * quatConst.xxy + tmp.wxy;    // matrix0 <- (ww-0.5, xy+zw, xz-yw, %)
    rotMatT0.x = torus_quat.x * torus_quat.x + rotMatT0.x;                // matrix0 <- (ww+x*x-0.5, xy+zw, xz-yw, %)
    rotMatT0 = rotMatT0 + rotMatT0;                     // matrix0 <- (2(ww+x*x)-1, 2(xy+zw), 2(xz-yw), %)

    rotMatT1.xyz = tmp1.zwx * quatConst.yxx + tmp.xwz;     // matrix1 <- (xy-zw, ww-0.5, yz+xw, %)
    rotMatT1.y = torus_quat.y * torus_quat.y + rotMatT1.y;             // matrix1 <- (xy-zw, ww+y*y-0.5, yz+xw, %)
    rotMatT1 = rotMatT1 + rotMatT1;                     // matrix1 <- (2(xy-zw), 2(ww+y*y)-1, 2(yz+xw), %)

    rotMatT2.xyz = tmp1.yxw * quatConst.xyx + tmp.yzw;     // matrix2 <- (xz+yw, yz-xw, ww-0.5, %)
    rotMatT2.z = torus_quat.z * torus_quat.z + rotMatT2.z;             // matrix2 <- (xz+yw, yz-xw, ww+zz-0.5, %)
    rotMatT2 = rotMatT2 + rotMatT2;                     // matrix2 <- (2(xz+yw), 2(yz-xw), 2(ww+zz)-1, %)    
    // End: Holy code!

    // rotate and copy the visibility sphere
    visibilitySphere.xyz = rotMatT0 * torus_sphere.x + rotMatT1 * torus_sphere.y + rotMatT2 * torus_sphere.z;
    visibilitySphere.w = torus_sphere.w;
    
    cuttingPlane = rotMatT0 * torus_cuttingplane.x + rotMatT1 * torus_cuttingplane.y + rotMatT2 * torus_cuttingplane.z;
    
    // calculate cam position
    tmp = viewInverse[3]; // (C) by Christoph
    tmp.xyz -= objPos.xyz; // cam move
    camPos.xyz = rotMatT0 * tmp.x + rotMatT1 * tmp.y + rotMatT2 * tmp.z;
    
    // Sphere-Touch-Plane-Approach
    vec2 winHalf = 2.0 / viewAttr.zw; // window size

    vec2 d, p, q, h, dd;

    vec2 mins, maxs;
    vec3 testPos;
    vec4 projPos;

    // projected camera vector
    vec3 c2 = vec3(dot(tmp.xyz, camRight), dot(tmp.xyz, camUp), dot(tmp.xyz, camIn));

    vec3 cpj1 = camIn * c2.z + camRight * c2.x;
    vec3 cpm1 = camIn * c2.x - camRight * c2.z;

    vec3 cpj2 = camIn * c2.z + camUp * c2.y;
    vec3 cpm2 = camIn * c2.y - camUp * c2.z;

    d.x = length(cpj1);
    d.y = length(cpj2);

    dd = vec2(1.0) / d;

    ////p = (radii.x + radii.z)*(radii.x + radii.z) * dd;
    p = torus_sphere.w*torus_sphere.w * dd;
    q = d - p;
    h = sqrt(p * q);
    //h = vec2(0.0);

    p *= dd;
    h *= dd;

    cpj1 *= p.x;
    cpm1 *= h.x;
    cpj2 *= p.y;
    cpm2 *= h.y;

    // TODO: rewrite only using four projections, additions in homogenous coordinates and delayed perspective divisions.
    ////testPos = objPos.xyz + cpj1 + cpm1;
    testPos = torus_sphere.xyz + objPos.xyz + cpj1 + cpm1;    
    projPos = mvp * vec4(testPos, 1.0);
    projPos /= projPos.w;
    mins = projPos.xy;
    maxs = projPos.xy;

    testPos -= 2.0 * cpm1;
    projPos = mvp * vec4(testPos, 1.0);
    projPos /= projPos.w;
    mins = min(mins, projPos.xy);
    maxs = max(maxs, projPos.xy);

    ////testPos = objPos.xyz + cpj2 + cpm2;
    testPos = torus_sphere.xyz + objPos.xyz + cpj2 + cpm2;
    projPos = mvp * vec4(testPos, 1.0);
    projPos /= projPos.w;
    mins = min(mins, projPos.xy);
    maxs = max(maxs, projPos.xy);

    testPos -= 2.0 * cpm2;
    projPos = mvp * vec4(testPos, 1.0);
    projPos /= projPos.w;
    mins = min(mins, projPos.xy);
    maxs = max(maxs, projPos.xy);

    // set position and point size
    gl_Position = vec4((mins + maxs) * 0.5, 0.0, 1.0);
    gl_PointSize = max((maxs.x - mins.x) * winHalf.x, (maxs.y - mins.y) * winHalf.y) * 0.5;
}
