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

float superquadricsFunc(float x, float y, float z, float alpha, float beta) {
    return pow(pow(abs(y), 2.0f / alpha) + pow(abs(z), 2.0f / alpha), alpha / beta) + pow(abs(x), 2.0f / beta) - 1.0f;
}

vec3 superquadricsNormal(float x, float y, float z, float alpha, float beta) {
    vec3 normal;
    const float first_derived = pow(pow(abs(y), 2.0f / alpha) + pow(abs(z), 2.0f / alpha), alpha / beta - 1.0f);
    normal.x = 2.0f * x * pow(abs(x), 2.0f / beta - 1.0f);
    normal.y = 2.0f * y * pow(abs(y), 2.0f / alpha - 1.0f) * first_derived;
    normal.z = 2.0f * z * pow(abs(z), 2.0f / alpha - 1.0f) * first_derived;
    return normalize(normal);
}

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


    // iterative ray marching approach
    // see: Kindlmann, Superquadric Tensor Glyphs
    // start solution with box intersection
    float t = 0.9f * length(normalized_os_coord - cam_pos);

    // sort non-negative eigenvalues
    vec3 superquadric;
    if (absradii.x >= absradii.y && absradii.x >= absradii.z) {
        superquadric.x = absradii.x;
        if (absradii.y >= absradii.z) {
            superquadric.y = absradii.y;
            superquadric.z = absradii.z;
        } else {
            superquadric.y = absradii.z;
            superquadric.z = absradii.y;
        }
    } else if (absradii.y >= absradii.z) {
        superquadric.x = absradii.y;
        if (absradii.x >= absradii.z) {
            superquadric.y = absradii.x;
            superquadric.z = absradii.z;
        } else {
            superquadric.y = absradii.z;
            superquadric.z = absradii.x;
        }
    } else {
        superquadric.x = absradii.z;
        if (absradii.x >= absradii.y) {
            superquadric.y = absradii.x;
            superquadric.z = absradii.y;
        } else {
            superquadric.y = absradii.y;
            superquadric.z = absradii.x;
        }
    }

    // calculate anisotropy
    const float eigenvalue_sum = superquadric.x + superquadric.y + superquadric.z;
    const float linear_anisotropy = (superquadric.x - superquadric.y) / eigenvalue_sum;
    const float planar_anisotropy = 2.0f * (superquadric.y - superquadric.z) / eigenvalue_sum;
    const float spherical_anisotropy = 3.0f * superquadric.z / eigenvalue_sum;

    // calculate geometric anisotropy metrics
    const float alpha = (linear_anisotropy >= planar_anisotropy) ? pow(1.0f - planar_anisotropy, exponent)
                                                                 : pow(1.0f - linear_anisotropy, exponent);
    const float beta = (linear_anisotropy >= planar_anisotropy) ? pow(1.0f - linear_anisotropy, exponent)
                                                                : pow(1.0f - planar_anisotropy, exponent);

    float x = cam_pos.x + t * ray.x;
    float y = cam_pos.y + t * ray.y;
    float z = cam_pos.z + t * ray.z;

    float result = superquadricsFunc(x, y, z, alpha, beta);

    if (result < 0.0f) {
        discard;
    }

    // set a threshold at which the result is good enough
    float threshold = 0.0001;
    float t1 = t;
    t += 0.0001;
    // march along ray until we find a hit
    while (result > threshold) {
        x = cam_pos.x + t * ray.x;
        y = cam_pos.y + t * ray.y;
        z = cam_pos.z + t * ray.z;

        const float old_result = result;
        result = superquadricsFunc(x, y, z, alpha, beta);

        // if we move further away from the object
        // discard
        if (result >= old_result) {
            discard;
        }

        float tmp_t = 0.9f * (result * (t - t1)) / (result - old_result);
        t1 = t;
        t -= tmp_t;
    }

    vec3 intersection = cam_pos + t1 * ray;

    // normal calculation for superquadric
    // see: http://courses.cms.caltech.edu/cs171/assignments/hw7/hw7-notes/notes-hw7.html
    // basically just derivating the superquadric equation
    vec3 normal = superquadricsNormal(x, y, z, alpha, beta);

    // translate point back to original position
    // transform normal and intersection point into tensor
    mat3 scaling = mat3(absradii.x, 0.0f, 0.0f, 0.0f, absradii.y, 0.0f, 0.0f, 0.0f, absradii.z);

    normal = transpose(inverse(transpose(rotate_tensor_into_world) * scaling)) * normal;
    normal = normalize(normal);

    intersection = transpose(rotate_tensor_into_world) * scaling * intersection;
    intersection += obj_pos.xyz;


    // color stuff
    vec3 dir_color1 = max( vec3(0), normal * sign(radii) );
    vec3 dir_color2 = vec3(1) + normal * sign(radii);
    vec3 dir_color = any( lessThan( dir_color2, vec3( 0.5 ) ) ) ? dir_color2 * vec3(0.5) : dir_color1;


    // calc depth
    float far = gl_DepthRange.far;
    float near = gl_DepthRange.near;
    vec4 ding = vec4(intersection, 1.0);
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
