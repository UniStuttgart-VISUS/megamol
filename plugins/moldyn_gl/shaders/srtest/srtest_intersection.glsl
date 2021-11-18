void intersection(vec3 objPos, float sqrRad, vec3 oc_pos, float c, float rad, out vec4 new_pos, out vec3 normal, out vec3 ray) {
    vec4 pos_ndc =
        vec4(2.0f * (gl_FragCoord.xy / viewAttr.zw) - 1.0f, (2.0f * gl_FragCoord.z) / (far - near) - 1.0f, 1.0f);
    vec4 pos_clip = MVPinv * pos_ndc;
    vec3 pos_obj = pos_clip.xyz / pos_clip.w;

    ray = normalize(pos_obj - camPos);

    float b = dot(-oc_pos, ray);
    vec3 temp = oc_pos + b * ray;
    float delta = sqrRad - dot(temp, temp);

    if (delta < 0.0f)
        discard;

    float sign = b >= 0.0f ? 1.0f : -1.0f;
    float q = b + sign * sqrt(delta);

    float t = min(c / q, q);

    new_pos = vec4(camPos + t * ray, 1.0f);

    normal = (new_pos.xyz - objPos) / rad;
}
