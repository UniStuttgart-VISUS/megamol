#version 450

uniform vec4 viewAttr;

uniform vec3 camPos;
uniform vec3 camUp;
uniform vec3 camRight;
uniform vec3 camIn;

uniform mat4 MVP;
uniform mat4 P;

flat out vec3 objPos;
flat out float rad;

layout(location=0) in vec4 inPosition;

void main(void) {
  
    // Remove the sphere radius from the w coordinates to the rad varyings
    vec4 inPos = inPosition;
    rad = inPos.w;
    inPos.w = 1.0;

    vec2 mins, maxs;

    vec3 di = inPos.xyz-camPos;
    float dd = dot(di, di);

    float s = (rad*rad)/(dd);

    float v = rad/sqrt(1-s);

    vec3 vr = normalize(cross(di, camUp))*v;
    vec3 vu = normalize(cross(di, vr))*v;

    vec4 v1 = MVP * vec4(inPos.xyz + vr, 1.0);
    vec4 v2 = MVP * vec4(inPos.xyz - vr, 1.0);
    vec4 v3 = MVP * vec4(inPos.xyz + vu, 1.0);
    vec4 v4 = MVP * vec4(inPos.xyz - vu, 1.0);

    v1 /= v1.w;
    v2 /= v2.w;
    v3 /= v3.w;
    v4 /= v4.w;

    mins = v1.xy;
    maxs = v1.xy;
    mins = min(mins, v2.xy);
    maxs = max(maxs, v2.xy);
    mins = min(mins, v3.xy);
    maxs = max(maxs, v3.xy);
    mins = min(mins, v4.xy);
    maxs = max(maxs, v4.xy);

    v1.xy = 0.5*v1.xy*viewAttr.zw+0.5*viewAttr.zw;
    v2.xy = 0.5*v2.xy*viewAttr.zw+0.5*viewAttr.zw;
    v3.xy = 0.5*v3.xy*viewAttr.zw+0.5*viewAttr.zw;
    v4.xy = 0.5*v4.xy*viewAttr.zw+0.5*viewAttr.zw;


    // vec3 zd = cross(camIn, camPos-inPos.xyz);
    // float zw = sqrt((length(camIn)*length(camIn)) * (length(camPos-inPos.xyz)*length(camPos-inPos.xyz))) + dot(camIn, camPos-inPos.xyz);
    // vec4 q = vec4(zd, zw);
    // q = normalize(q);
    // vec4 q_c = vec4(-q.x, -q.y, -q.z, q.w);

    // vec4 v1a = vec4(camRight * rad  + camUp * -rad, 0);
    // vec4 v2a = vec4(camRight * rad  + camUp * rad , 0);
    // vec4 v3a = vec4(camRight * -rad + camUp * -rad, 0);
    // vec4 v4a = vec4(camRight * -rad + camUp * rad , 0);

    // // vec4 v1a = vec4(vec3(1,0,0) * rad  + vec3(0,1,0) * -rad, 0);
    // // vec4 v2a = vec4(vec3(1,0,0) * rad  + vec3(0,1,0) * rad , 0);
    // // vec4 v3a = vec4(vec3(1,0,0) * -rad + vec3(0,1,0) * -rad, 0);
    // // vec4 v4a = vec4(vec3(1,0,0) * -rad + vec3(0,1,0) * rad , 0);

    // vec4 v1 = MVP * vec4(inPos.xyz + (v1a).xyz, 1.0);
    // vec4 v2 = MVP * vec4(inPos.xyz + (v2a).xyz, 1.0);
    // vec4 v3 = MVP * vec4(inPos.xyz + (v3a).xyz, 1.0);
    // vec4 v4 = MVP * vec4(inPos.xyz + (v4a).xyz, 1.0);

    // v1 /= v1.w;
    // v2 /= v2.w;
    // v3 /= v3.w;
    // v4 /= v4.w;

    // mins = v1.xy;
    // maxs = v1.xy;
    // mins = min(mins, v2.xy);
    // maxs = max(maxs, v2.xy);
    // mins = min(mins, v3.xy);
    // maxs = max(maxs, v3.xy);
    // mins = min(mins, v4.xy);
    // maxs = max(maxs, v4.xy);

    // v1.xy = 0.5*v1.xy*viewAttr.zw+0.5*viewAttr.zw;
    // v2.xy = 0.5*v2.xy*viewAttr.zw+0.5*viewAttr.zw;
    // v3.xy = 0.5*v3.xy*viewAttr.zw+0.5*viewAttr.zw;
    // v4.xy = 0.5*v4.xy*viewAttr.zw+0.5*viewAttr.zw;



    vec4 projPos = MVP * vec4(inPos.xyz+rad*(camIn), 1.0);
    //vec4 projPos = MVP*inPos;
    projPos = projPos/projPos.w;

    vec2 vw = (v1 - v2).xy;
    vec2 vh = (v3 - v4).xy;

    // vec2 vw = (v1 - v3).xy;
    // vec2 vh = (v2 - v1).xy;

    objPos = inPos.xyz;

    //gl_PointSize = max(length(vw)*viewAttr.z, length(vh)*viewAttr.w);
    gl_PointSize = max(length(vw), length(vh));
    //gl_Position = projPos;

    gl_Position = vec4((mins + maxs) * 0.5, projPos.z, 1.0);
}
