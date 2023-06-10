#version 450

layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

#include "srtest_ubo.glsl"

in VPoint {
    flat float rad;
    flat float sqrRad;
    flat vec4  pointColor;
    flat vec3  oc_pos;
}
v_pp[];

out Point{
    flat float rad;
    flat float sqrRad;
    flat vec4 pointColor;
    flat vec3 oc_pos;
}
g_pp;

#include "srtest_touchplane.glsl"

void main() {
    int idx = 0;

    mat4 v;
    touchplane_old(gl_in[idx].gl_Position.xyz, v_pp[idx].rad, v_pp[idx].oc_pos, v);

    gl_Position = v[0];
    g_pp.rad = v_pp[idx].rad;
    g_pp.sqrRad = v_pp[idx].sqrRad;
    g_pp.pointColor = v_pp[idx].pointColor;
    g_pp.oc_pos = v_pp[idx].oc_pos;
    EmitVertex();

    gl_Position = v[1];
    g_pp.rad = v_pp[idx].rad;
    g_pp.sqrRad = v_pp[idx].sqrRad;
    g_pp.pointColor = v_pp[idx].pointColor;
    g_pp.oc_pos = v_pp[idx].oc_pos;
    EmitVertex();

    gl_Position = v[3];
    g_pp.rad = v_pp[idx].rad;
    g_pp.sqrRad = v_pp[idx].sqrRad;
    g_pp.pointColor = v_pp[idx].pointColor;
    g_pp.oc_pos = v_pp[idx].oc_pos;
    EmitVertex();

    gl_Position = v[2];
    g_pp.rad = v_pp[idx].rad;
    g_pp.sqrRad = v_pp[idx].sqrRad;
    g_pp.pointColor = v_pp[idx].pointColor;
    g_pp.oc_pos = v_pp[idx].oc_pos;
    EmitVertex();
    EndPrimitive();
}
