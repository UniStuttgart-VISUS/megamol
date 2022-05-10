#version 460

#include "protein_cuda/molecule_cb/mcbc_common.glsl"

layout(location = 0) in vec4 torus_pos;
layout(location = 1) in vec4 torus_axis;
layout(location = 2) in vec4 torus_sphere;
layout(location = 3) in vec4 torus_color;

uniform vec4 viewAttr; // TODO: check fragment position if viewport starts not in (0, 0)
uniform vec3 zValues;
uniform vec3 fogCol;

#ifndef CALC_CAM_SYS
uniform vec3 camIn;
uniform vec3 camUp;
uniform vec3 camRight;
#endif // CALC_CAM_SYS
uniform mat4 invview;
uniform mat4 mvp;

out vec4 objPos;
out vec4 camPos;
out vec4 radii;
out vec4 visibilitySphere;

out vec3 rotMatT0;
out vec3 rotMatT1; // rotation matrix from the quaternion
out vec3 rotMatT2;

out vec4 colors;

void main(void) {
    vec4 tmp, tmp1;

    // remove the sphere radius from the w coordinates to the rad varyings
    vec4 inPos = torus_pos;

    radii.x = torus_axis.w;
    radii.y = radii.x * radii.x;
    radii.z = inPos.w;
    radii.w =  radii.z * radii.z;
        
    colors = torus_color;
    
    inPos.w = 1.0;
    
    // object pivot point in object space    
    objPos = inPos; // no w-div needed, because w is 1.0 (Because I know)
 
    rotMatT0 = torus_axis.xyz;
 
    rotMatT2 = ((rotMatT0.x > 0.9) || (rotMatT0.x < -0.9)) ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0); // normal on tmp
    rotMatT1 = cross(rotMatT0, rotMatT2);
    rotMatT1 = normalize(rotMatT1);
    rotMatT2 = cross(rotMatT1, rotMatT0);
 
    vec3 ttmp1 = rotMatT2;
    vec3 ttmp2 = rotMatT1;
    vec3 ttmp3 = rotMatT0;
 
    rotMatT0 = vec3(ttmp1.x, ttmp2.x, ttmp3.x);
    rotMatT1 = vec3(ttmp1.y, ttmp2.y, ttmp3.y);
    rotMatT2 = vec3(ttmp1.z, ttmp2.z, ttmp3.z);
 
    // rotate and copy the visibility sphere
    visibilitySphere.xyz = rotMatT0 * torus_sphere.x + rotMatT1 * torus_sphere.y + rotMatT2 * torus_sphere.z;
    visibilitySphere.w = torus_sphere.w;
        
    // calculate cam position
    tmp = invview[3]; // (C) by Christoph
    tmp.xyz -= objPos.xyz; // cam move
    camPos.xyz = rotMatT0 * tmp.x + rotMatT1 * tmp.y + rotMatT2 * tmp.z;
    
    // Sphere-Touch-Plane-Approach
    vec2 winHalf = 2.0 / viewAttr.zw; // window size

    vec2 d, p, q, h, dd;

    // get camera orthonormal coordinate system

#ifdef CALC_CAM_SYS
    // camera coordinate system in object space
    tmp = invview[3] + invview[2];
    vec3 camIn = normalize(tmp.xyz);
    tmp = invview[3] + invview[1];
    vec3 camUp = tmp.xyz;
    vec3 camRight = normalize(cross(camIn, camUp));
    camUp = cross(camIn, camRight);
#endif // CALC_CAM_SYS

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
