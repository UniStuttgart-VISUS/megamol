// TODO: copyright note of links used?
/**
 * MegaMol
 * Copyright (c) 2022, MegaMol Dev Team
 * All rights reserved.
 */

in vec4 obj_pos;
in vec3 cam_pos;
in vec3 absradii;
in vec3 radii;
in mat3 rotate_tensor_into_world;

in vec4 vert_color;

uniform float exponent;

out layout(location = 0) vec4 albedo_out;
out layout(location = 1) vec3 normal_out;
out layout(location = 2) float depth_out;

void main() {
    // transform fragment coordinates from screen coordinates to ndc coordinates.
    vec4 coord = gl_FragCoord
        * vec4(view_attr.z, view_attr.w, 2.0, 0.0)
        + vec4(-1.0, -1.0, -1.0, 1.0);

    // transform ndc coordinates to tensor coordinates.
    coord = mvp_i * coord;
    coord /= coord.w;
    coord -= obj_pos; // ... and move

    // since fragment coordinate is from rotated box
    // we need to take out the rotation to get the object coordinates
    vec3 os_coord = rotate_tensor_into_world * coord.xyz;

    // calc the viewing ray
    vec3 ray = normalize(os_coord - cam_pos);


    // iterative ray marching approach
    // see: http://cg.inf.h-bonn-rhein-sieg.de/wp-content/uploads/2009/04/introductiontoraytracing.pdf
    // start solution with box intersection
    float t = length(os_coord - cam_pos);
    float radius = 1.0;
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

    vec3 intersection = cam_pos + t1 * ray;

    // TODO: link is offline, new reference required!
    // normal calculation for superquadric
    // see: http://cg.inf.h-bonn-rhein-sieg.de/wp-content/uploads/2009/04/introductiontoraytracing.pdf
    // basically just derivating the superquadric equation
    vec3 normal = intersection / superquadric;
    float k = exponent - 1.0;
    if( normal.x > 0.0 ) normal.x = pow( normal.x, k );
    else normal.x = -pow( -normal.x, k );
    if( normal.y > 0.0 ) normal.y = pow( normal.y, k );
    else normal.y = -pow( -normal.y, k );
    if( normal.z > 0.0 ) normal.z = pow( normal.z, k );
    else normal.z = -pow( -normal.z, k );
    normal = normalize( normal / superquadric );

    // transform normal and intersection point into tensor
    normal = transpose(rotate_tensor_into_world) * normal;
    normal = normalize(normal);

    // translate point back to original position
    intersection = transpose(rotate_tensor_into_world) * intersection;
    intersection += obj_pos.xyz;


    // color stuff
    vec3 dir_color1 = max( vec3(0), normal * sign(radii) );
    vec3 dir_color2 = vec3(1) + normal * sign(radii);
    vec3 dir_color = any( lessThan( dir_color2, vec3( 0.5 ) ) ) ? dir_color2 * vec3(0.5) : dir_color1;


    // calc non-linear depth
    float far = gl_DepthRange.far;      // set by glDepthRange()
    float near = gl_DepthRange.near;    // set by glDepthRange()
    vec4 hc_int = vec4(intersection, 1.0);
    // only rows 2 and 3 (or columns of transposed matrix) are needed
    // to calc transformed depth
    float depth = dot(mvp_t[2], hc_int);
    float depth_w = dot(mvp_t[3], hc_int);
    float depth_ndc = depth / depth_w;
    float depth_ss = ((far - near) * depth_ndc + (far + near)) / 2.0;

    // linear depth
    // near_ - and far_plane from camera view frustum
    // uniforms probably missing, add them in code if you need it
    //depth_ss = (2.0 * near_plane * far_plane) / (far_plane + near_plane - depth_ndc * (far_plane - near_plane));
    //depth_ss /= far_plane; // normalize for visualization


    // outputs
    albedo_out = vec4(mix(dir_color, vert_color.rgb, color_interpolation),1.0);
    normal_out = normal;
    depth_out = depth_ss; // can probably removed at some point (SimpleRrenderTarget doesn't use it anymore)
    gl_FragDepth = depth_ss;
}
