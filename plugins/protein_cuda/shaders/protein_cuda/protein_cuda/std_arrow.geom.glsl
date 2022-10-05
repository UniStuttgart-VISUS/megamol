#version 140

#include "protein_cuda/protein_cuda/commondefines.glsl"
#include "lightdirectional.glsl"

#extension GL_ARB_gpu_shader5 : enable
#extension GL_EXT_geometry_shader4 : enable

layout (points) in;
layout (triangle_strip, max_vertices = 4) out;

uniform mat4 modelview;
uniform mat4 proj;
uniform vec4 viewAttr; // TODO: check fragment position if viewport starts not in (0, 0)
uniform vec4 lightPos;

#ifndef CALC_CAM_SYS
uniform vec3 camIn;
uniform vec3 camUp;
uniform vec3 camRight;
#endif // CALC_CAM_SYS

#ifdef DEBUG
uniform vec2 circleAttr;
#endif // DEBUG

uniform float radScale;

in vec4 inPos0[1];
in vec4 inPos1[1];
in vec4 inColor[1];

out vec4 objPos;
out vec4 camPos;
out vec4 light;
out vec4 radz; /* (cyl-Rad, tip-Rad, overall-Len, tip-Len) */
out vec3 rotMatT0;
out vec3 rotMatT1; // rotation matrix from the quaternion
out vec3 rotMatT2;
out vec4 colOut;

#ifdef RETICLE
out vec2 centerFragment;
#endif // RETICLE

mat4 modelviewproj = proj*modelview; // TODO Move this to the CPU?
mat4 modelviewInv = inverse(modelview);

void main(void) {
    //const vec4 quatConst = vec4(1.0, -1.0, 0.5, 0.0);
    vec4 tmp, tmp1;

    //vec4 vertex2 = gl_MultiTexCoord0;

    //vec4 quatC = vec4(0.0, 0.0, 0.0, 1.0);
    vec2 inParams = vec2(inPos0[0].w, length(inPos0[0].xyz - inPos1[0].xyz));

    //vec3 xAxis = gl_Vertex.xyz - gl_MultiTexCoord0.xyz;
    vec3 xAxis = inPos1[0].xyz - inPos0[0].xyz;
    xAxis /= inParams.y;

    // remove the sphere radius from the w coordinates to the rad varyings
    vec4 inPos = inPos0[0];

    radz.x = inParams.x * radScale;
    radz.y = radz.x * 1.5;
    radz.z = inParams.y;
//    radz.w = min(gl_MultiTexCoord0.w, radz.z);
    radz.w = radz.z * 0.4;

    inPos.w = 1.0;

    // object pivot point in object space
    objPos = inPos; // no w-div needed, because w is 1.0 (Because I know)


    // orientation matrix based on direction vector
    vec3 tmpy = (xAxis.x == 0.0) ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0);
    vec3 tmpz = normalize(cross(tmpy, xAxis));
    tmpy = normalize(cross(xAxis, tmpz));

    rotMatT0 = vec3(xAxis.x, tmpy.x, tmpz.x);
    rotMatT1 = vec3(xAxis.y, tmpy.y, tmpz.y);
    rotMatT2 = vec3(xAxis.z, tmpy.z, tmpz.z);

    //rotMatT0 = xAxis;
    //rotMatT0 = normalize(rotMatT0);
    //rotMatT1 = ((rotMatT0.x > 0.9) || (rotMatT0.x < -0.9)) ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0); // normal on tmp
    //rotMatT1 = cross(rotMatT1, rotMatT0);
    //rotMatT1 = normalize(rotMatT1);
    //rotMatT2 = cross(rotMatT0, rotMatT1);


    //rotMatT0 = xAxis;
    //rotMatT1 = (xAxis.x == 0.0) ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0);
    //rotMatT2 = normalize(cross(rotMatT0, rotMatT1));
    //rotMatT1 = normalize(cross(rotMatT0, rotMatT2));


    // calculate cam position
    tmp = modelviewInv[3]; // (C) by Christoph
    tmp.xyz -= objPos.xyz; // cam move
    camPos.xyz = rotMatT0 * tmp.x + rotMatT1 * tmp.y + rotMatT2 * tmp.z;


    // calculate light position in glyph space
    // USE THIS LINE TO GET POSITIONAL LIGHTING
    //lightPos = gl_ModelViewMatrixInverse * gl_LightSource[0].position - objPos; // note: w is bullshit now!
    // USE THIS LINE TO GET DIRECTIONAL LIGHTING
    light = modelviewInv * normalize(lightPos);
    light.xyz = rotMatT0 * light.x + rotMatT1 * light.y + rotMatT2 * light.z;


    // send color to fragment shader
    colOut = inColor[0];
/*
    float sc = 1.0 - inParams.x;
    sc *= sc;
    gl_FrontColor = vec4(0.0, 1.0 - sc, sc, 1.0);
*/

    // calculate point sprite
    //vec2 winHalf = 2.0 / viewAttr.zw; // window size

    // lumberjack approach
    vec4 pos, projPos;
    vec4 pX, pY, pZ, pOP;
    vec2 mins, maxs, pp;

#define CYL_LEN radz.z
#define CYL_RAD radz.y

    projPos.w = 0.0;

    //pos = vec4(0.0, 0.0, 0.0, 1.0);
    //projPos.x = dot(rotMatT0.xyz, pos.xyz); // rotate
    //projPos.y = dot(rotMatT1.xyz, pos.xyz);
    //projPos.z = dot(rotMatT2.xyz, pos.xyz);
    pos = objPos; // + projPos; // move
    pos.w = 1.0; // now we're in object space
    pOP = modelviewproj * pos;

    pos = vec4(1.0, 0.0, 0.0, 1.0);
    projPos.x = dot(rotMatT0.xyz, pos.xyz); // rotate
    projPos.y = dot(rotMatT1.xyz, pos.xyz);
    projPos.z = dot(rotMatT2.xyz, pos.xyz);
    pos = objPos + projPos; // move
    pos.w = 1.0; // now we're in object space
    projPos = modelviewproj * pos;
    pX = (projPos - pOP) * CYL_LEN;

    pos = vec4(0.0, 1.0, 0.0, 1.0);
    projPos.x = dot(rotMatT0.xyz, pos.xyz); // rotate
    projPos.y = dot(rotMatT1.xyz, pos.xyz);
    projPos.z = dot(rotMatT2.xyz, pos.xyz);
    pos = objPos + projPos; // move
    pos.w = 1.0; // now we're in object space
    projPos = modelviewproj * pos;
    pY = (projPos - pOP) * CYL_RAD;

    pos = vec4(0.0, 0.0, 1.0, 1.0);
    projPos.x = dot(rotMatT0.xyz, pos.xyz); // rotate
    projPos.y = dot(rotMatT1.xyz, pos.xyz);
    projPos.z = dot(rotMatT2.xyz, pos.xyz);
    pos = objPos + projPos; // move
    pos.w = 1.0; // now we're in object space
    projPos = modelviewproj * pos;
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

    pos = pOP;
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

/*    gl_Position = vec4((mins + maxs) * 0.5, 0.0, 1.0);
    maxs = (maxs - mins) * 0.5 * winHalf;
    gl_PointSize = max(maxs.x, maxs.y);

#ifdef RETICLE
    centerFragment = gl_Position.xy / gl_Position.w;
#endif // RETICLE*/

    gl_Position = vec4(mins.x, maxs.y, 0.0, inPos.w); EmitVertex();
    gl_Position = vec4(mins.x, mins.y, 0.0, inPos.w); EmitVertex();
    gl_Position = vec4(maxs.x, maxs.y, 0.0, inPos.w); EmitVertex();
    gl_Position = vec4(maxs.x, mins.y, 0.0, inPos.w); EmitVertex();
    EndPrimitive();
}
