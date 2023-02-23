struct Ray {
    vec3 o;
    vec3 d;
};

Ray generateRay(ivec2 pixel_coords) {
    Ray ray;

    // Transform pixel to clip coordinates
    vec2 clip_space_pixel_coords =
        vec2((pixel_coords.x / rt_resolution.x) * 2.0f - 1.0f, (pixel_coords.y / rt_resolution.y) * 2.0f - 1.0f);

    // Unproject a point on the near plane and use as an origin
    mat4 inv_view_proj_mx = inverse(proj_mx * view_mx);
    vec4 unproj = inv_view_proj_mx * vec4(clip_space_pixel_coords, -1.0f, 1.0f);

    ray.o = unproj.xyz / unproj.w;

    // Unproject a point at the same pixel, but further away from the near plane
    // to compute a ray direction in world space
    unproj = inv_view_proj_mx * vec4(clip_space_pixel_coords, 0.0f, 1.0f);

    ray.d = normalize((unproj.xyz / unproj.w) - ray.o);

    return ray;
}

bool intersectBox(Ray r, vec3 boxmin, vec3 boxmax, out float tnear, out float tfar) {
    vec3 invR = vec3(1.0f) / r.d;
    vec3 tbot = invR * (boxmin - r.o);
    vec3 ttop = invR * (boxmax - r.o);

    // Special case for a ray lying in a bounding plane.
    if (r.d.x == 0.0f && r.o.x == boxmax.x) {
        ttop.x = -FLT_MAX;
        tbot.x = FLT_MAX;
    }
    if (r.d.y == 0.0f && r.o.y == boxmax.y) {
        ttop.y = -FLT_MAX;
        tbot.y = FLT_MAX;
    }
    if (r.d.z == 0.0f && r.o.z == boxmax.z) {
        ttop.z = -FLT_MAX;
        tbot.z = FLT_MAX;
    }

    vec3 tmin = min(ttop, tbot);
    vec3 tmax = max(ttop, tbot);

    float largest_tmin = max(max(tmin.x, tmin.y), max(tmin.x, tmin.z));
    float smallest_tmax = min(min(tmax.x, tmax.y), min(tmax.x, tmax.z));

    tnear = largest_tmin;
    tfar = smallest_tmax;

    return smallest_tmax > largest_tmin;
}

float wang_hash(uint seed) {
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);

    return float(seed) / 4294967296.0;
}

float calculate_depth(vec3 pos) {
    vec4 clip_pos = proj_mx * view_mx * vec4(pos, 1.0f);

    return ((clip_pos.z / clip_pos.w) + 1.0f) / 2.0f;
}

vec3 calculate_normal(vec3 texCoords) {
    const float left = textureOffset(volume_tx3D, texCoords, ivec3(-1, 0, 0)).x;
    const float right = textureOffset(volume_tx3D, texCoords, ivec3(1, 0, 0)).x;

    const float bottom = textureOffset(volume_tx3D, texCoords, ivec3(0, -1, 0)).x;
    const float top = textureOffset(volume_tx3D, texCoords, ivec3(0, 1, 0)).x;

    const float front = textureOffset(volume_tx3D, texCoords, ivec3(0, 0, -1)).x;
    const float back = textureOffset(volume_tx3D, texCoords, ivec3(0, 0, 1)).x;

    return normalize(vec3(
        (left - right) / halfVoxelSize.x,
        (bottom - top) / halfVoxelSize.y,
        (front - back) / halfVoxelSize.z));
}
