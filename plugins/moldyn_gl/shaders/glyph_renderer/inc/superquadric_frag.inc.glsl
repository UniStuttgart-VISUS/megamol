// TODO: copyright note of links used?
/**
 * MegaMol
 * Copyright (c) 2022, MegaMol Dev Team
 * All rights reserved.
 */

in vec4 obj_pos;
in vec3 cam_pos;
// TODO: scale radii below radii --> i.e. x * absradii with 0.0 < absradii < 1.0
// to fit into the box (which is scaled by radii?)
in vec3 absradii;
in mat3 rotate_world_into_tensor;

in vec4 vert_color;
in flat vec3 dir_color;

uniform float exponent;

out layout(location = 0) vec4 albedo_out;
out layout(location = 1) vec3 normal_out;
out layout(location = 2) float depth_out;

void main() {
    // transform fragment coordinates from window coordinates to view coordinates.
    vec4 coord = gl_FragCoord
        * vec4(view_attr.z, view_attr.w, 2.0, 0.0)
        + vec4(-1.0, -1.0, -1.0, 1.0);

    // transform fragment coordinates from view coordinates to object coordinates.
    coord = mvp_i * coord;
    coord /= coord.w;
    // is this the box position?
    vec3 box_pos = coord.xyz;
    coord -= obj_pos; // ... and move

    // ... and rotate with rot_mat transposed
    vec3 tmp = rotate_world_into_tensor * coord.xyz;

    // calc the viewing ray
    vec3 ray = normalize(tmp - cam_pos);

    // iterative ray marching approch
    // see: http://cg.inf.h-bonn-rhein-sieg.de/wp-content/uploads/2009/04/introductiontoraytracing.pdf
    // start solution with box intersection
    float t = length(tmp - cam_pos);
    // TODO: what radius to use?
    float radius = 1.0;
    // TODO: what are the superqadrics parameters? the radii?
    // TODO: what are the exponents? the radii?
    vec3 superquadric = absradii;
    float result = pow(abs((cam_pos.x + t * ray.x) / superquadric.x), exponent) +
                   pow(abs((cam_pos.y + t * ray.y) / superquadric.y), exponent) +
                   pow(abs((cam_pos.z + t * ray.z) / superquadric.z), exponent) -
                   radius;

    // set a threshold at which the result is good enough
    float threshold = 0.0001;
    float t1 = t;
    t += 0.0001;
    // march along ray until we find a hit
    while(result > threshold) {
        float old_result = result;
        result = pow(abs((cam_pos.x + t * ray.x) / superquadric.x), exponent) +
                 pow(abs((cam_pos.y + t * ray.y) / superquadric.y), exponent) +
                 pow(abs((cam_pos.z + t * ray.z) / superquadric.z), exponent) -
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

    vec3 intersection = cam_pos + t * ray;

    // normal calculation for superquadric
    // see: http://cg.inf.h-bonn-rhein-sieg.de/wp-content/uploads/2009/04/introductiontoraytracing.pdf
    vec3 normal = intersection / superquadric;
    float k = exponent - 1;
    if( normal.x > 0 ) normal.x = pow( normal.x, k );
    else normal.x = -pow( -normal.x, k );
    if( normal.y > 0 ) normal.y = pow( normal.y, k );
    else normal.y = -pow( -normal.y, k );
    if( normal.z > 0 ) normal.z = pow( normal.z, k );
    else normal.z = -pow( -normal.z, k );
    normal = normalize( normal );

    //albedo_out = vec4(normal, 1.0);
    albedo_out = vec4(mix(dir_color, vert_color.rgb, color_interpolation),1.0);
    normal_out = normal;

    vec4 ding = vec4(intersection, 1.0);
    float depth = dot(mvp_t[2], ding);
    float depth_w = dot(mvp_t[3], ding);
    depth_out = ((depth / depth_w) + 1.0) * 0.5;
}
