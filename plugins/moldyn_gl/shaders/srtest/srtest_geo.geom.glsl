#version 450

layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

#include "srtest_ubo.glsl"

in VPoint {
    flat vec3  objPos;
    flat float rad;
    flat float sqrRad;
    flat vec4  pointColor;
    flat vec3  oc_pos;
}
v_pp[];

out Point{
    flat vec3 objPos;
    flat float rad;
    flat float sqrRad;
    flat vec4 pointColor;
    flat vec3 oc_pos;
}
g_pp;

#include "srtest_touchplane.glsl"

void main() {
    int idx = 0;

    /*float dd = dot(v_pp[idx].oc_pos, v_pp[idx].oc_pos);

    float s = (v_pp[idx].sqrRad) / (dd);

    float vi = v_pp[idx].rad / sqrt(1.0f - s);

    vec3 vr = normalize(cross(v_pp[idx].oc_pos, camUp)) * vi;
    vec3 vu = normalize(cross(v_pp[idx].oc_pos, vr)) * vi;

    vec4 v[4];
    v[0] = vec4(v_pp[idx].objPos - vr - vu, 1.0f);
    v[1] = vec4(v_pp[idx].objPos + vr - vu, 1.0f);
    v[2] = vec4(v_pp[idx].objPos + vr + vu, 1.0f);
    v[3] = vec4(v_pp[idx].objPos - vr + vu, 1.0f);

    v[0] = MVP * v[0];
    v[1] = MVP * v[1];
    v[2] = MVP * v[2];
    v[3] = MVP * v[3];*/
    mat4 v;
    touchplane_old(v_pp[idx].objPos, v_pp[idx].rad, v_pp[idx].oc_pos, v);

    /*v[0] /= v[0].w;
    v[1] /= v[1].w;
    v[2] /= v[2].w;
    v[3] /= v[3].w;*/

    //vec4 projPos = MVP * vec4(v_pp[idx].objPos + v_pp[idx].rad * (camDir), 1.0f);
    //projPos = projPos / projPos.w;

    //gl_Position = vec4(v[0].xy, projPos.z, projPos.w);
    gl_Position = v[0];
    g_pp.objPos = v_pp[idx].objPos;
    g_pp.rad = v_pp[idx].rad;
    g_pp.sqrRad = v_pp[idx].sqrRad;
    g_pp.pointColor = v_pp[idx].pointColor;
    g_pp.oc_pos = v_pp[idx].oc_pos;
    EmitVertex();
    //gl_Position = vec4(v[1].xy, projPos.z, projPos.w);
    gl_Position = v[1];
    g_pp.objPos = v_pp[idx].objPos;
    g_pp.rad = v_pp[idx].rad;
    g_pp.sqrRad = v_pp[idx].sqrRad;
    g_pp.pointColor = v_pp[idx].pointColor;
    g_pp.oc_pos = v_pp[idx].oc_pos;
    EmitVertex();
    //gl_Position = vec4(v[3].xy, projPos.z, projPos.w);
    gl_Position = v[3];
    g_pp.objPos = v_pp[idx].objPos;
    g_pp.rad = v_pp[idx].rad;
    g_pp.sqrRad = v_pp[idx].sqrRad;
    g_pp.pointColor = v_pp[idx].pointColor;
    g_pp.oc_pos = v_pp[idx].oc_pos;
    EmitVertex();
    //gl_Position = vec4(v[2].xy, projPos.z, projPos.w);
    gl_Position = v[2];
    g_pp.objPos = v_pp[idx].objPos;
    g_pp.rad = v_pp[idx].rad;
    g_pp.sqrRad = v_pp[idx].sqrRad;
    g_pp.pointColor = v_pp[idx].pointColor;
    g_pp.oc_pos = v_pp[idx].oc_pos;
    EmitVertex();
    EndPrimitive();
}
