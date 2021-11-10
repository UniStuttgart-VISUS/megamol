#version 450

uniform vec4 viewAttr;

uniform vec3 camPos;
uniform vec3 camUp;
uniform vec3 camRight;
uniform vec3 camDir;

uniform mat4 MVP;

flat out vec3 objPos;
flat out float rad;
flat out float sqrRad;

layout(location = 0) in vec4 inPosition;

void main(void) {

    // Remove the sphere radius from the w coordinates to the rad varyings
    vec4 inPos = inPosition;
    rad = inPos.w;
    inPos.w = 1.0;

    vec2 mins, maxs;

    vec3 di = inPos.xyz - camPos;
    float dd = dot(di, di);

    sqrRad = rad * rad;

    float s = (sqrRad) / (dd);

    float v = rad / sqrt(1 - s);

    vec3 vr = normalize(cross(di, camUp)) * v;
    vec3 vu = normalize(cross(di, vr)) * v;

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

    v1.xy = 0.5 * v1.xy * viewAttr.zw + 0.5 * viewAttr.zw;
    v2.xy = 0.5 * v2.xy * viewAttr.zw + 0.5 * viewAttr.zw;
    v3.xy = 0.5 * v3.xy * viewAttr.zw + 0.5 * viewAttr.zw;
    v4.xy = 0.5 * v4.xy * viewAttr.zw + 0.5 * viewAttr.zw;

    vec4 projPos = MVP * vec4(inPos.xyz + rad * (camDir), 1.0);
    projPos = projPos / projPos.w;

    vec2 vw = (v1 - v2).xy;
    vec2 vh = (v3 - v4).xy;

    objPos = inPos.xyz;

    gl_PointSize = max(length(vw), length(vh));

    gl_Position = vec4((mins + maxs) * 0.5, projPos.z, 1.0);
}
