#version 430

#include "protein_gl/simplemolecule/sm_common_defines.glsl"

uniform vec4 viewAttr; // TODO: check fragment position if viewport starts not in (0, 0)
uniform vec3 zValues;
uniform vec3 fogCol;
uniform vec2 texOffset;
uniform mat4 view;
uniform mat4 proj;
uniform mat4 viewInverse;
uniform mat4 mvp;

#ifndef CALC_CAM_SYS
uniform vec3 camIn;
uniform vec3 camUp;
uniform vec3 camRight;
#endif // CALC_CAM_SYS
layout (location = 0) in vec4 vert_position;
layout (location = 1) in vec3 vert_color;
layout (location = 2) in vec4 tri_attrib1;
layout (location = 3) in vec4 tri_attrib2;
layout (location = 4) in vec4 tri_attrib3;
layout (location = 5) in vec3 tri_tex_coord1;
layout (location = 6) in vec3 tri_tex_coord2;
layout (location = 7) in vec3 tri_tex_coord3;

out vec4 objPos;
out vec4 camPos;
out vec4 lightPos;

out vec4 inVec1;
out vec4 inVec2;
out vec4 inVec3;
out vec3 inColors;
out vec3 texCoord1;
out vec3 texCoord2;
out vec3 texCoord3;
    
void main(void) {
    inVec1 = tri_attrib1;
    inVec2 = tri_attrib2;
    inVec3 = tri_attrib3;
    inColors = vert_color;
    
    texCoord1 = tri_tex_coord1;
    texCoord2 = tri_tex_coord2;
    texCoord3 = tri_tex_coord3;
    
    // remove the sphere radius from the w coordinates to the rad varyings
    vec4 inPos = vert_position;
    inVec1.w = inPos.w;
    inVec2.w = inPos.w * inPos.w;
    inPos.w = 1.0;
    
    // DEBUG
    //texCoord2.xy = vec2( tri_attrib1.w, tri_attrib2.w);
    
    // object pivot point in object space    
    objPos = inPos; // no w-div needed, because w is 1.0 (Because I know)
    
    // calculate cam position
    camPos = viewInverse[3]; // (C) by Christoph
    camPos.xyz -= objPos.xyz; // cam pos to glyph space
    
    // calculate light position in glyph space
    // USE THIS LINE TO GET POSITIONAL LIGHTING
    //lightPos = viewInverse * gl_LightSource[0].position - objPos;
    // USE THIS LINE TO GET DIRECTIONAL LIGHTING
    //lightPos = viewInverse * normalize( gl_LightSource[0].position);
    lightPos = vec4(0,0,0,1); // dummy

    // Sphere-Touch-Plane-Approach
    vec2 winHalf = 2.0 / viewAttr.zw; // window size
    
    vec2 d, p, q, h, dd;
    
    // get camera orthonormal coordinate system
    vec4 tmp;
    
    #ifdef CALC_CAM_SYS
    // camera coordinate system in object space
    tmp = viewInverse[3] + viewInverse[2];
    vec3 camIn = normalize(tmp.xyz);
    tmp = viewInverse[3] + viewInverse[1];
    vec3 camUp = tmp.xyz;
    vec3 camRight = normalize(cross(camIn, camUp));
    camUp = cross(camIn, camRight);
    #endif // CALC_CAM_SYS
    
    vec2 mins, maxs;
    vec3 testPos;
    vec4 projPos;
    
    // projected camera vector
    vec3 c2 = vec3(dot(camPos.xyz, camRight), dot(camPos.xyz, camUp), dot(camPos.xyz, camIn));
    
    vec3 cpj1 = camIn * c2.z + camRight * c2.x;
    vec3 cpm1 = camIn * c2.x - camRight * c2.z;
    
    vec3 cpj2 = camIn * c2.z + camUp * c2.y;
    vec3 cpm2 = camIn * c2.y - camUp * c2.z;
    
    d.x = length(cpj1);
    d.y = length(cpj2);
    
    dd = vec2(1.0) / d;
    
    p = inVec2.w * dd;
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
    testPos = objPos.xyz + cpj1 + cpm1;
    projPos = mvp * vec4(testPos, 1.0);
    projPos /= projPos.w;
    mins = projPos.xy;
    maxs = projPos.xy;
    
    testPos -= 2.0 * cpm1;
    projPos = mvp * vec4(testPos, 1.0);
    projPos /= projPos.w;
    mins = min(mins, projPos.xy);
    maxs = max(maxs, projPos.xy);
    
    testPos = objPos.xyz + cpj2 + cpm2;
    projPos = mvp * vec4(testPos, 1.0);
    projPos /= projPos.w;
    mins = min(mins, projPos.xy);
    maxs = max(maxs, projPos.xy);
    
    testPos -= 2.0 * cpm2;
    projPos = mvp * vec4(testPos, 1.0);
    projPos /= projPos.w;
    mins = min(mins, projPos.xy);
    maxs = max(maxs, projPos.xy);
    
    gl_Position = vec4((mins + maxs) * 0.5, 0.0, 1.0);
    gl_PointSize = max((maxs.x - mins.x) * winHalf.x, (maxs.y - mins.y) * winHalf.y) * 0.5;
    
    #ifdef SMALL_SPRITE_LIGHTING
    // for normal crowbaring on very small sprites
    lightPos.w = (clamp(gl_PointSize, 1.0, 5.0) - 1.0) / 4.0;
    #endif // SMALL_SPRITE_LIGHTING    
}