void touchplane(vec3 objPos, float rad, out vec4 projPos, out float l) {
    vec2 mins, maxs;

    vec3 di = objPos - camPos;
    float dd = dot(di, di);

    float sqrRad = rad * rad;

    float s = (sqrRad) / (dd);

    float p = sqrRad / sqrt(dd);

    float v = rad / sqrt(1.0f - s);
    v = v / sqrt(dd) * (sqrt(dd) - p);

    vec3 vr = normalize(cross(di, camUp)) * v;
    vec3 vu = normalize(cross(di, vr)) * v;

    vec3 base = objPos - p * camDir;

    vec4 v1 = MVP * vec4(base + vr, 1.0f);
    vec4 v2 = MVP * vec4(base - vr, 1.0f);
    vec4 v3 = MVP * vec4(base + vu, 1.0f);
    vec4 v4 = MVP * vec4(base - vu, 1.0f);

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

    projPos = MVP * vec4(objPos - p * camDir, 1.0f);
    projPos = projPos / projPos.w;

    //projPos.xy = (mins + maxs) * 0.5f;
    l = max(length(vw), length(vh));
}


void touchplane(vec3 objPos, float rad, vec3 oc_pos, out mat4 v) {
    float dd = dot(oc_pos, oc_pos);

    float s = (rad * rad) / (dd);

    float vi = rad / sqrt(1.0f - s);

    float d = sqrt(dd);
    vi = vi / d * (d - rad);

    vec3 vr = normalize(cross(oc_pos, camUp)) * vi;
    vec3 vu = normalize(cross(oc_pos, vr)) * vi;

    vec3 base_pos = objPos - rad * normalize(oc_pos);

    v[0] = vec4(base_pos - vr - vu, 1.0f);
    v[1] = vec4(base_pos + vr - vu, 1.0f);
    v[2] = vec4(base_pos + vr + vu, 1.0f);
    v[3] = vec4(base_pos - vr + vu, 1.0f);

    v[0] = MVP * v[0];
    v[1] = MVP * v[1];
    v[2] = MVP * v[2];
    v[3] = MVP * v[3];
}


void touchplane_v2(vec3 objPos, float rad, vec3 oc_pos, out mat4 v) {
    float dd = dot(oc_pos, oc_pos);

    float s = (rad * rad) / (dd);

    float vi = rad / sqrt(1.0f - s);

    float d = sqrt(dd);
    vi = vi / d * (d - rad);

    vec3 vr = normalize(cross(oc_pos, camUp)) * vi;
    vec3 vu = normalize(cross(oc_pos, vr)) * vi;

    vec3 base_pos = objPos - rad * normalize(oc_pos);

    v[0] = vec4(base_pos - vr + vu, 1.0f);
    v[1] = vec4(base_pos - vr - vu, 1.0f);
    v[2] = vec4(base_pos + vr + vu, 1.0f);
    v[3] = vec4(base_pos + vr - vu, 1.0f);

    v[0] = MVP * v[0];
    v[1] = MVP * v[1];
    v[2] = MVP * v[2];
    v[3] = MVP * v[3];
}


void touchplane_woMVP(vec3 objPos, float rad, vec3 oc_pos, out mat4 v) {
    float dd = dot(oc_pos, oc_pos);

    float s = (rad * rad) / (dd);

    float vi = rad / sqrt(1.0f - s);

    float d = sqrt(dd);
    vi = vi / d * (d - rad);

    vec3 vr = normalize(cross(oc_pos, camUp)) * vi;
    vec3 vu = normalize(cross(oc_pos, vr)) * vi;

    vec3 base_pos = objPos - rad * normalize(oc_pos);

    v[0] = vec4(base_pos - vr - vu, 1.0f);
    v[1] = vec4(base_pos + vr - vu, 1.0f);
    v[2] = vec4(base_pos + vr + vu, 1.0f);
    v[3] = vec4(base_pos - vr + vu, 1.0f);
}


void touchplane_old(vec3 objPos, float rad, vec3 oc_pos, out vec4 projPos, out float l) {
    // Sphere-Touch-Plane-Approach

    vec2 winHalf = viewAttr.zw; // window size

    vec2 d, p, q, h, dd;

    // get camera orthonormal coordinate system
    vec4 tmp;

    vec2 mins, maxs;
    vec3 testPos;

    float sqrRad = rad * rad;

    // projected camera vector
    vec3 c2 = vec3(dot(-oc_pos, camRight), dot(-oc_pos, camUp), dot(-oc_pos, camDir));

    vec3 cpj1 = camDir * c2.z + camRight * c2.x;
    vec3 cpm1 = camDir * c2.x - camRight * c2.z;

    vec3 cpj2 = camDir * c2.z + camUp * c2.y;
    vec3 cpm2 = camDir * c2.y - camUp * c2.z;

    d.x = length(cpj1);
    d.y = length(cpj2);

    dd = vec2(1.0) / d;

    p = sqrRad * dd;
    q = d - p;
    h = sqrt(p * q);

    p *= dd;
    h *= dd;

    cpj1 *= p.x;
    cpm1 *= h.x;
    cpj2 *= p.y;
    cpm2 *= h.y;

    testPos = objPos.xyz + cpj1 + cpm1;
    projPos = MVP * vec4(testPos, 1.0);
    projPos /= projPos.w;
    mins = projPos.xy;
    maxs = projPos.xy;

    testPos -= 2.0 * cpm1;
    projPos = MVP * vec4(testPos, 1.0);
    projPos /= projPos.w;
    mins = min(mins, projPos.xy);
    maxs = max(maxs, projPos.xy);

    testPos = objPos.xyz + cpj2 + cpm2;
    projPos = MVP * vec4(testPos, 1.0);
    projPos /= projPos.w;
    mins = min(mins, projPos.xy);
    maxs = max(maxs, projPos.xy);

    testPos -= 2.0 * cpm2;
    projPos = MVP * vec4(testPos, 1.0);
    projPos /= projPos.w;
    mins = min(mins, projPos.xy);
    maxs = max(maxs, projPos.xy);

    testPos = objPos.xyz - camDir * rad;
    projPos = MVP * vec4(testPos, 1.0);
    projPos /= projPos.w;

    projPos = vec4((mins + maxs) * 0.5, projPos.z, 1.0);
    maxs = (maxs - mins) * 0.5 * winHalf;
    l = max(maxs.x, maxs.y) + 0.5;
}


void touchplane_old(vec3 objPos, float rad, vec3 oc_pos, out mat4 v) {
    // Sphere-Touch-Plane-Approach

    vec2 winHalf = viewAttr.zw; // window size

    vec2 d, p, q, h, dd;

    // get camera orthonormal coordinate system
    vec4 tmp;

    vec2 mins, maxs;
    vec3 testPos;
    vec4 projPos;

    float sqrRad = rad * rad;

    // projected camera vector
    vec3 c2 = vec3(dot(-oc_pos, camRight), dot(-oc_pos, camUp), dot(-oc_pos, camDir));

    vec3 cpj1 = camDir * c2.z + camRight * c2.x;
    vec3 cpm1 = camDir * c2.x - camRight * c2.z;

    vec3 cpj2 = camDir * c2.z + camUp * c2.y;
    vec3 cpm2 = camDir * c2.y - camUp * c2.z;

    d.x = length(cpj1);
    d.y = length(cpj2);

    dd = vec2(1.0) / d;

    p = sqrRad * dd;
    q = d - p;
    h = sqrt(p * q);

    p *= dd;
    h *= dd;

    cpj1 *= p.x;
    cpm1 *= h.x;
    cpj2 *= p.y;
    cpm2 *= h.y;

    testPos = objPos.xyz + cpj1 + cpm1;
    projPos = MVP * vec4(testPos, 1.0);
    projPos /= projPos.w;
    mins = projPos.xy;
    maxs = projPos.xy;

    testPos -= 2.0 * cpm1;
    projPos = MVP * vec4(testPos, 1.0);
    projPos /= projPos.w;
    mins = min(mins, projPos.xy);
    maxs = max(maxs, projPos.xy);

    testPos = objPos.xyz + cpj2 + cpm2;
    projPos = MVP * vec4(testPos, 1.0);
    projPos /= projPos.w;
    mins = min(mins, projPos.xy);
    maxs = max(maxs, projPos.xy);

    testPos -= 2.0 * cpm2;
    projPos = MVP * vec4(testPos, 1.0);
    projPos /= projPos.w;
    mins = min(mins, projPos.xy);
    maxs = max(maxs, projPos.xy);

    testPos = objPos.xyz - camDir * rad;
    projPos = MVP * vec4(testPos, 1.0);
    projPos /= projPos.w;

    v[2] = vec4(maxs, projPos.z, 1.0);
    v[0] = vec4(mins, projPos.z, 1.0);

    v[3] = vec4(mins.x, maxs.y, projPos.z, 1.0);
    v[1] = vec4(maxs.x, mins.y, projPos.z, 1.0);
}


void touchplane_old_v2(vec3 objPos, float rad, vec3 oc_pos, out mat4 v) {
    // Sphere-Touch-Plane-Approach

    vec2 winHalf = viewAttr.zw; // window size

    vec2 d, p, q, h, dd;

    // get camera orthonormal coordinate system
    vec4 tmp;

    vec2 mins, maxs;
    vec3 testPos;
    vec4 projPos;

    float sqrRad = rad * rad;

    // projected camera vector
    vec3 c2 = vec3(dot(-oc_pos, camRight), dot(-oc_pos, camUp), dot(-oc_pos, camDir));

    vec3 cpj1 = camDir * c2.z + camRight * c2.x;
    vec3 cpm1 = camDir * c2.x - camRight * c2.z;

    vec3 cpj2 = camDir * c2.z + camUp * c2.y;
    vec3 cpm2 = camDir * c2.y - camUp * c2.z;

    d.x = length(cpj1);
    d.y = length(cpj2);

    dd = vec2(1.0) / d;

    p = sqrRad * dd;
    q = d - p;
    h = sqrt(p * q);

    p *= dd;
    h *= dd;

    cpj1 *= p.x;
    cpm1 *= h.x;
    cpj2 *= p.y;
    cpm2 *= h.y;

    testPos = objPos.xyz + cpj1 + cpm1;
    projPos = MVP * vec4(testPos, 1.0);
    projPos /= projPos.w;
    mins = projPos.xy;
    maxs = projPos.xy;

    testPos -= 2.0 * cpm1;
    projPos = MVP * vec4(testPos, 1.0);
    projPos /= projPos.w;
    mins = min(mins, projPos.xy);
    maxs = max(maxs, projPos.xy);

    testPos = objPos.xyz + cpj2 + cpm2;
    projPos = MVP * vec4(testPos, 1.0);
    projPos /= projPos.w;
    mins = min(mins, projPos.xy);
    maxs = max(maxs, projPos.xy);

    testPos -= 2.0 * cpm2;
    projPos = MVP * vec4(testPos, 1.0);
    projPos /= projPos.w;
    mins = min(mins, projPos.xy);
    maxs = max(maxs, projPos.xy);

    testPos = objPos.xyz - camDir * rad;
    projPos = MVP * vec4(testPos, 1.0);
    projPos /= projPos.w;

    v[2] = vec4(maxs.xy, projPos.z, 1.0);
    v[1] = vec4(mins.xy, projPos.z, 1.0);

    v[0] = vec4(mins.x, maxs.y, projPos.z, 1.0);
    v[3] = vec4(maxs.x, mins.y, projPos.z, 1.0);
}
