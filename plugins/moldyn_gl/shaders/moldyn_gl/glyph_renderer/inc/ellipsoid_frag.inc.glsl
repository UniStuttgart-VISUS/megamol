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
    vec3 normalized_os_coord = os_coord / absradii;

    // calc the viewing ray
    vec3 ray = normalize(normalized_os_coord - cam_pos.xyz);

    // calculate the geometry-ray-intersection (sphere with radius = 1)
    float d1 = -dot(cam_pos.xyz, ray);                       // projected length of the cam-sphere-vector onto the ray
    float d2s = dot(cam_pos.xyz, cam_pos.xyz) - d1 * d1;     // off axis of cam-sphere-vector and ray
    float radicand = 1.0 - d2s;                              // square of difference of projected length and lambda
    if (radicand < 0.0) { discard; }
    float lambda = d1 - sqrt(radicand);
    vec3 sphere_intersection = lambda * ray + cam_pos.xyz;

    // re-scale intersection point
    sphere_intersection *= absradii;

    // calc normal at (re-scaled) intersection point
    vec3 normal = normalize( ( sphere_intersection / absradii ) / absradii );
    normal = normalize( transpose( rotate_tensor_into_world ) * normal );

    // re-transform intersection into world space
    sphere_intersection = transpose(rotate_tensor_into_world) * sphere_intersection.xyz;
    sphere_intersection += obj_pos.xyz;


    // color stuff
    vec3 dir_color1 = max( vec3(0), normal * sign(radii) );
    vec3 dir_color2 = vec3(1) + normal * sign(radii);
    vec3 dir_color = any( lessThan( dir_color2, vec3( 0.5 ) ) ) ? dir_color2 * vec3(0.5) : dir_color1;


    // calc depth
    float far = gl_DepthRange.far;
    float near = gl_DepthRange.near;
    vec4 ding = vec4(sphere_intersection, 1.0);
    float depth = dot(mvp_t[2], ding);
    float depth_w = dot(mvp_t[3], ding);
    float depth_ndc = depth / depth_w;
    float depth_ss = ((far - near) * depth_ndc + (far + near)) / 2.0;

    // linear depth
    // near_ - and far_plane from camera view frustum
    // uniforms probably missing, add them in code if you need it
    //depth_ss = (2.0 * near_plane * far_plane) / (far_plane + near_plane - depth_ndc * (far_plane - near_plane));
    //depth_ss /= far_plane; // normalize for visualization


    // outputs
    albedo_out = vec4(mix(dir_color, vert_color.rgb, color_interpolation), 1.0);
    normal_out = normal;
    depth_out = depth_ss; // can probably removed at some point (SimpleRrenderTarget doesn't use it anymore)
    gl_FragDepth = depth_ss;
}
