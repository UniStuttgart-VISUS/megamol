void touchplane(vec3 objPos, float rad, out vec4 projPos, out float l) {
    vec2 mins, maxs;

    vec3 di = objPos - camPos;
    float dd = dot(di, di);

    float sqrRad = rad * rad;

    float s = (sqrRad) / (dd);

    float v = rad / sqrt(1.0f - s);

    vec3 vr = normalize(cross(di, camUp)) * v;
    vec3 vu = normalize(cross(di, vr)) * v;

    vec4 v1 = MVP * vec4(objPos + vr, 1.0f);
    vec4 v2 = MVP * vec4(objPos - vr, 1.0f);
    vec4 v3 = MVP * vec4(objPos + vu, 1.0f);
    vec4 v4 = MVP * vec4(objPos - vu, 1.0f);

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

    vec2 factor = 0.5f * viewAttr.zw;
    v1.xy = factor * (v1.xy + 1.0f);
    v2.xy = factor * (v2.xy + 1.0f);
    v3.xy = factor * (v3.xy + 1.0f);
    v4.xy = factor * (v4.xy + 1.0f);

    vec2 vw = (v1 - v2).xy;
    vec2 vh = (v3 - v4).xy;

    projPos = MVP * vec4(objPos + rad * (camDir), 1.0f);
    projPos = projPos / projPos.w;

    projPos.xy = (mins + maxs) * 0.5f;
    l = max(length(vw), length(vh));
}
