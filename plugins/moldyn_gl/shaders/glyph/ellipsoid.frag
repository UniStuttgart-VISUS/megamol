in vec4 obj_pos;
in vec3 cam_pos;
in vec4 vert_color;

in vec3 inv_rad;

in flat vec3 dir_color;
in flat vec3 transformed_normal;

in mat3 rotate_world_into_tensor;
in mat3 rotate_points;

out layout(location = 0) vec4 albedo_out;
out layout(location = 1) vec3 normal_out;
out layout(location = 2) float depth_out;

void main() {
    vec4 coord;
    vec3 ray, tmp;
    float lambda;

    // transform fragment coordinates from window coordinates to view coordinates.
    coord = gl_FragCoord
        * vec4(view_attr.z, view_attr.w, 2.0, 0.0)
        + vec4(-1.0, -1.0, -1.0, 1.0);


    // transform fragment coordinates from view coordinates to object coordinates.
    coord = mvp_i * coord;
    // TODO: correct order? doesnt /w need to be after inverse(projection)?
    coord /= coord.w;
    coord -= obj_pos; // ... and move

    // ... and rotate with rot_mat transposed
    ray = rotate_world_into_tensor * coord.xyz;
    ray *= inv_rad;

    // calc the viewing ray
    ray = normalize(ray - cam_pos.xyz);

    // calculate the geometry-ray-intersection (sphere with radius = 1)
    float d1 = -dot(cam_pos.xyz, ray);                       // projected length of the cam-sphere-vector onto the ray
    float d2s = dot(cam_pos.xyz, cam_pos.xyz) - d1 * d1;      // off axis of cam-sphere-vector and ray
    float radicand = 1.0 - d2s;                             // square of difference of projected length and lambda
    if (radicand < 0.0) { discard; }
    lambda = d1 - sqrt(radicand);                           // lambda
    vec3 sphereintersection = lambda * ray + cam_pos.xyz;    // intersection point


    // "calc" normal at intersection point
    vec3 normal = sphereintersection;
    vec3 color = step(normal, vec3(0));

    normal = rotate_points * normal;
    ray = rotate_points * ray;

    tmp = sphereintersection / inv_rad;

    //sphereintersection.x = dot(rot_mat_t0, tmp.xyz);
    //sphereintersection.y = dot(rot_mat_t1, tmp.xyz);
    //sphereintersection.z = dot(rot_mat_t2, tmp.xyz);
    sphereintersection = rotate_world_into_tensor * tmp.xyz;

    // phong lighting with directional light, colors are hardcoded as they're inaccessible for some reason
    //out_frag_color = vec4(LocalLighting(ray, normal, lightPos.xyz, color), 1.0);
    //out_frag_color = vec4(LocalLighting(ray, normalize(sphereintersection), lightPos.xyz, color), 1.0);

    albedo_out = vec4(mix(dir_color, vert_color.rgb, color_interpolation), 1.0);
    normal_out = normal;

    sphereintersection += obj_pos.xyz;

    vec4 ding = vec4(sphereintersection, 1.0);
    float depth = dot(mvp_t[2], ding);
    float depth_w = dot(mvp_t[3], ding);
    depth_out = ((depth / depth_w) + 1.0) * 0.5;

    //depth_out = gl_FragCoord.z;
    //albedo_out = mix(dir_color, vert_color.rgb, color_interpolation);
    //normal_out = transformed_normal;
}
