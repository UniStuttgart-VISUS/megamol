in vec3 rad; // r, r^2, len
in vec4 cam_pos;
in vec4 obj_pos;
in vec4 rot_light_dir;

in vec3 rot_mat_t0;
in vec3 rot_mat_t1;
in vec3 rot_mat_t2;
flat in uint discard_frag;

void main(void) {
    vec4 coord;
    vec3 ray, tmp;
    const float max_lambda = 50000.0;

    if (discard_frag > uint(0)) {
        discard;
    }

    // transform fragment coordinates from window coordinates to view coordinates.
    coord = gl_FragCoord
        * vec4(view_attr.z, view_attr.w, 2.0, 0.0)
        + vec4(-1.0, -1.0, -1.0, 1.0);

    // transform fragment coordinates from view coordinates to object coordinates.
    coord = mvp_i * coord;
    coord /= coord.w;
    coord -= obj_pos; // ... and move


    // calc the viewing ray
    ray = (rot_mat_t0 * coord.x) + (rot_mat_t1 * coord.y) + (rot_mat_t2 * coord.z);
    ray = normalize(ray - cam_pos.xyz);

    vec4 cpos = cam_pos + vec4(rad.z, 0.0, 0.0, 0.0);

    // calculate the geometry-ray-intersection

    // arrow parameters
#define CYL_RAD (rad.x * 0.5)
#define CYL_RAD_SQ (rad.y * 0.25)
#define CYL_LEN (rad.z * 2.0)
#define TIP_RAD rad.x
#define TIP_LEN (rad.z * 0.8)

    // super unoptimized cone code

    float cone_f = TIP_RAD / TIP_LEN;
    cone_f *= cone_f;
    float cone_a = cone_f * ray.x * ray.x - ray.y * ray.y - ray.z * ray.z;
    float cone_b = 2.0 * (cone_f * ray.x * cpos.x - ray.y * cpos.y - ray.z * cpos.z);
    float cone_c = cone_f * cpos.x * cpos.x - cpos.y * cpos.y - cpos.z * cpos.z;

    float r_dc = dot(ray.yz, cpos.yz);
    float r_dr = dot(ray.yz, ray.yz);

    vec2 radicand = vec2(
        (r_dc * r_dc) - (r_dr * (dot(cpos.yz, cpos.yz) - CYL_RAD_SQ)),
        cone_b * cone_b - 4.0 * cone_a * cone_c);
    vec2 divisor = vec2(r_dr, 2.0 * cone_a);
    vec2 radix = sqrt(radicand);
    vec2 minus_b = vec2(-r_dc, -cone_b);

    vec4 lambda = vec4(
        (minus_b.x - radix.x) / divisor.x,
        (minus_b.y + radix.y) / divisor.y,
        (minus_b.x + radix.x) / divisor.x,
        (minus_b.y - radix.y) / divisor.y);

    bvec4 invalid = bvec4(
        (divisor.x == 0.0) || (radicand.x < 0.0),
        (divisor.y == 0.0) || (radicand.y < 0.0),
        (divisor.x == 0.0) || (radicand.x < 0.0),
        (divisor.y == 0.0) || (radicand.y < 0.0));

    vec4 ix = cpos.xxxx + ray.xxxx * lambda;


    invalid.x = invalid.x || (ix.x < TIP_LEN) || (ix.x > CYL_LEN);
    invalid.y = invalid.y || (ix.y < 0.0) || (ix.y > TIP_LEN);
    invalid.z = invalid.z || !(((ix.z > TIP_LEN) || (ix.x > CYL_LEN)) && (ix.z < CYL_LEN));
    invalid.w = invalid.w || !((ix.w > 0.0) && (ix.w < TIP_LEN));

    if (invalid.x && invalid.y && invalid.z && invalid.w) {
#ifdef CLIP
        discard;
#endif // CLIP
    }

    vec3 intersection, color;
    vec3 normal = vec3(1.0, 0.0, 0.0);
    color = gl_Color.rgb;

    if (!invalid.y) {
        invalid.xzw = bvec3(true, true, true);
        intersection = cpos.xyz + (ray * lambda.y);
        normal = normalize(vec3(-TIP_RAD / TIP_LEN, normalize(intersection.yz)));
//        color = vec3(1.0, 0.0, 0.0);
    }
    if (!invalid.x) {
        invalid.zw = bvec2(true, true);
        intersection = cpos.xyz + (ray * lambda.x);
        normal = vec3(0.0, normalize(intersection.yz));
    }
    if (!invalid.z) {
        invalid.w = true;
        lambda.z = (CYL_LEN - cpos.x) / ray.x;
        intersection = cpos.xyz + (ray * lambda.z);
    }
    if (!invalid.w) {
        lambda.w = (TIP_LEN - cpos.x) / ray.x;
        intersection = cpos.xyz + (ray * lambda.w);
    }

    //color.r = 1.0 - intersection.x / CYL_LEN;
    //color.g = 0.0; // intersection.x / CYL_LEN;
    //color.b = intersection.x / CYL_LEN;

    // phong lighting with directional light
    //gl_FragColor = vec4(color.rgb, 1.0);
    gl_FragColor = vec4(LocalLighting(ray, normal, rot_light_dir.xyz, color), 1.0);
    gl_FragColor.rgb = (0.75 * gl_FragColor.rgb) + (0.25 * color);

    // calculate depth
#ifdef DEPTH
    intersection -= (ray * dot(ray, vec3(rad.z, 0.0, 0.0)));
    tmp = intersection;
    intersection.x = dot(rot_mat_t0, tmp.xyz);
    intersection.y = dot(rot_mat_t1, tmp.xyz);
    intersection.z = dot(rot_mat_t2, tmp.xyz);

    intersection += obj_pos.xyz;

    vec4 ding = vec4(intersection, 1.0);
    float depth = dot(mvp_t[2], ding);
    float depth_w = dot(mvp_t[3], ding);
#ifndef CLIP
    if (invalid.x && invalid.y && invalid.z && invalid.w) {
        gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0);
        gl_FragDepth = 0.99999;
    } else {
#endif // CLIP
    gl_FragDepth = ((depth / depth_w) + 1.0) * 0.5;
#ifndef CLIP
    }
#endif // CLIP

//    gl_FragColor.rgb *= ;

#endif // DEPTH

#ifdef RETICLE
    coord = gl_FragCoord
        * vec4(view_attr.z, view_attr.w, 2.0, 0.0)
        + vec4(-1.0, -1.0, -1.0, 1.0);
    if (min(abs(coord.x - center_fragment.x), abs(coord.y - center_fragment.y)) < 0.002) {
        gl_FragColor.rgb += vec3(0.3, 0.3, 0.5);
    }
#endif // RETICLE
}
