in vec4 obj_pos;
in vec3 cam_pos;
in vec4 vert_color;

in vec3 inv_rad;
in vec3 absradii;

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

    // transform fragment coordinates from screen coordinates to ndc coordinates.
    coord = gl_FragCoord
        * vec4(view_attr.z, view_attr.w, 2.0, 0.0)
        + vec4(-1.0, -1.0, -1.0, 1.0);

    // transform ndc coordinates to object coordinates.
    coord = mvp_i * coord;
    coord /= coord.w;
    coord -= obj_pos; // ... and move

    // ... and rotate with rot_mat transposed
    tmp = rotate_world_into_tensor * coord.xyz;
    tmp *= inv_rad;

    // calc the viewing ray
    ray = normalize(tmp - cam_pos.xyz);

    // calculate the geometry-ray-intersection (sphere with radius = 1)
    float d1 = -dot(cam_pos.xyz, ray);                       // projected length of the cam-sphere-vector onto the ray
    float d2s = dot(cam_pos.xyz, cam_pos.xyz) - d1 * d1;      // off axis of cam-sphere-vector and ray
    float radicand = 1.0 - d2s;                             // square of difference of projected length and lambda
    if (radicand < 0.0) { discard; }
    lambda = d1 - sqrt(radicand);                           // lambda
    vec3 sphereintersection = lambda * ray + cam_pos.xyz;    // intersection point

    // re-scale intersection point
    tmp = sphereintersection / inv_rad;

    // calc normal at (re-scaled) intersection point
    vec3 normal = normalize( ( sphereintersection / absradii ) / absradii);
    normal = normalize(rotate_points * normal);

    // re-transform intersection into world space
    sphereintersection = inverse(rotate_world_into_tensor) * tmp.xyz;
    sphereintersection += obj_pos.xyz;

    // calc depth
    float far = gl_DepthRange.far;
    float near = gl_DepthRange.near;
    vec4 ding = vec4(sphereintersection, 1.0);
    float depth = dot(mvp_t[2], ding);
    float depth_w = dot(mvp_t[3], ding);
    float depth_ndc = depth / depth_w;
    float depth_ss = ((far - near) * depth_ndc + (far + near)) / 2.0;


    // outputs
    albedo_out = vec4(mix(dir_color, vert_color.rgb, color_interpolation), 1.0);
    normal_out = normal;
    depth_out = depth_ss;
    gl_FragDepth = depth_ss;
}
