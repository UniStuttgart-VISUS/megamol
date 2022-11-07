// TODO: copyright note of links used?
/**
 * MegaMol
 * Copyright (c) 2022, MegaMol Dev Team
 * All rights reserved.
 */

in vec4 obj_pos;
in vec3 cam_pos;
in vec3 absradii;
in mat3 rotate_world_into_tensor;

in vec4 vert_color;
in flat vec3 dir_color;

uniform float exponent;

out layout(location = 0) vec4 albedo_out;
out layout(location = 1) vec3 normal_out;
out layout(location = 2) float depth_out;

void main() {
    // transform fragment coordinates from screen coordinates to ndc coordinates.
    vec4 coord = gl_FragCoord
        * vec4(view_attr.z, view_attr.w, 2.0, 0.0)
        + vec4(-1.0, -1.0, -1.0, 1.0);

    // transform ndc coordinates to object coordinates.
    coord = mvp_i * coord;
    coord /= coord.w;
    coord -= obj_pos; // ... and move

    // ... and rotate with rot_mat transposed
    vec3 tmp = rotate_world_into_tensor * coord.xyz;

    // calc the viewing ray
    vec3 cpos = cam_pos;
    vec3 ray = normalize(tmp - cpos);

    // iterative ray marching approch
    // see: http://cg.inf.h-bonn-rhein-sieg.de/wp-content/uploads/2009/04/introductiontoraytracing.pdf
    // start solution with box intersection
    float t = length(tmp - cpos);
    float radius = 1.0;
    vec3 superquadric = absradii;
    float result = pow(abs((cpos.x + t * ray.x) / superquadric.x), exponent) +
                   pow(abs((cpos.y + t * ray.y) / superquadric.y), exponent) +
                   pow(abs((cpos.z + t * ray.z) / superquadric.z), exponent) -
                   radius;

    // set a threshold at which the result is good enough
    float threshold = 0.0001;
    float t1 = t;
    t += 0.0001;
    // march along ray until we find a hit
    while(result > threshold) {
        float old_result = result;
        result = pow(abs((cpos.x + t * ray.x) / superquadric.x), exponent) +
                 pow(abs((cpos.y + t * ray.y) / superquadric.y), exponent) +
                 pow(abs((cpos.z + t * ray.z) / superquadric.z), exponent) -
                 radius;

        // if we move further away from the object
        // discard
        if(result >= old_result) {
            discard;
        }

        float tmp_t = (result * (t - t1)) / (result - old_result);
        t1 = t;
        t -= tmp_t;
    }

    vec3 intersection = cpos + t * ray;

    // normal calculation for superquadric
    // see: http://cg.inf.h-bonn-rhein-sieg.de/wp-content/uploads/2009/04/introductiontoraytracing.pdf
    vec3 normal = intersection / superquadric;
    float k = exponent - 1.0;
    if( normal.x > 0.0 ) normal.x = pow( normal.x, k );
    else normal.x = -pow( -normal.x, k );
    if( normal.y > 0.0 ) normal.y = pow( normal.y, k );
    else normal.y = -pow( -normal.y, k );
    if( normal.z > 0.0 ) normal.z = pow( normal.z, k );
    else normal.z = -pow( -normal.z, k );
    normal = normalize( normal );

    // transform normal and intersection point into tensor
    normal = transpose(rotate_world_into_tensor) * normal;
    normal = normalize(normal);

    // translate point back to original position
    intersection = inverse(rotate_world_into_tensor) * intersection;
    intersection += obj_pos.xyz;

    // calc depth
    float far = gl_DepthRange.far;
    float near = gl_DepthRange.near;
    vec4 ding = vec4(intersection, 1.0);
    // calc non-linear depth
    float depth = dot(mvp_t[2], ding);
    float depth_w = dot(mvp_t[3], ding);
    float depth_ndc = depth / depth_w;
    float depth_ss = ((far - near) * depth_ndc + (far + near)) / 2.0;

    albedo_out = vec4(mix(dir_color, vert_color.rgb, color_interpolation),1.0);
    normal_out = normal;
    depth_out = depth_ss;
    gl_FragDepth = depth_ss;
}
