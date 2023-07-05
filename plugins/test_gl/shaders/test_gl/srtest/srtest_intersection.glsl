void intersection(vec3 objPos, float sqrRad, vec3 oc_pos, float c, float rad, out vec4 new_pos, out vec3 normal,
    out vec3 ray, out float t) {
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

    t = min(c / q, q);

    new_pos = vec4(camPos + t * ray, 1.0f);

    normal = (new_pos.xyz - objPos) / rad;
}


bool intersection_old(vec3 oc_pos, float sqrRad, float rad, out vec3 normal, out vec3 ray, out float t) {
    // transform fragment coordinates from window coordinates to view coordinates.
    vec4 coord = gl_FragCoord * vec4(2.0f / viewAttr.z, 2.0f / viewAttr.w, 2.0, 0.0) + vec4(-1.0, -1.0, -1.0, 1.0);

    // transform fragment coordinates from view coordinates to object coordinates.
    coord = MVPinv * coord;
    coord /= coord.w;

    ray = normalize(coord.xyz - camPos);

    // calculate the geometry-ray-intersection
    float b = dot(oc_pos, ray); // projected length of the cam-sphere-vector onto the ray
    vec3 temp = b * ray - oc_pos;
    float delta = sqrRad - dot(temp, temp); // Raytracing Gem Magic (http://www.realtimerendering.com/raytracinggems/)

    if (delta < 0.0f)
        return false;

    float c = dot(oc_pos, oc_pos) - sqrRad;

    float s = b < 0.0f ? -1.0f : 1.0f;
    float q = b + s * sqrt(delta);
    t = min(c / q, q);

    vec3 sphereintersection = t * ray - oc_pos; // intersection point
    normal = (sphereintersection) / rad;

    return true;
}
