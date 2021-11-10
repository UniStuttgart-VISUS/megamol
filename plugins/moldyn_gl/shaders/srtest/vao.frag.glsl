#version 450

uniform vec4 viewAttr;

uniform mat4 MVP;
uniform mat4 MVPinv;
uniform mat4 MVPtransp;

uniform vec3 camPos;

uniform float near;
uniform float far;

flat in vec3 objPos;
flat in float rad;

layout(location = 0) out vec4 outColor;

#include "lightdirectional.glsl"

void main(void) {
   vec3 pos_w = gl_FragCoord.xyz;

   vec3 pos_ndc = vec3(2.0*(pos_w.xy/viewAttr.zw) -1.0, (2.0 * pos_w.z)/(far-near)-1.0);
   //vec3 pos_clip = pos_ndc;// / gl_FragCoord.w;
   
   vec4 pos_clip = (MVPinv * vec4(pos_ndc, 1.0));
   vec3 pos_obj = pos_clip.xyz/pos_clip.w;

   outColor = vec4(1, 0, 0, 1);

   vec3 ray = normalize(pos_obj-camPos);

   vec3 oc = camPos-objPos;
   float b = dot(ray, oc);
   float c = dot(oc, oc)- rad*rad;

   float d = b*b-c;

   if (d >= 0.0) {
       outColor = vec4(1);
       float ta = -b-sqrt(d);
       float tb = -b+sqrt(d);
       float t = min(ta, tb);
       vec3 new_pos = camPos+t*ray;
    //    vec4 new_pos_clip = MVP*vec4(new_pos, 1.0);
    //    vec3 new_pos_ndc = new_pos_clip.xyz/new_pos_clip.w;
    //    float new_z = 0.5*(far-near)*new_pos_ndc.z+0.5*(far+near);
    //    gl_FragDepth = new_z;

    vec4 Ding = vec4(new_pos, 1.0);
    float depth = dot(MVPtransp[2], Ding);
    float depthW = dot(MVPtransp[3], Ding);
    gl_FragDepth = ((depth / depthW) + 1.0) * 0.5;
   }
}
