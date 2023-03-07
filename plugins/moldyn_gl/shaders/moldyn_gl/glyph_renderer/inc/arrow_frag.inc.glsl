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

uniform int orientation;


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


    // check which axis is used for alignment
    // 0 - x
    // 1 - y
    // 2 - z
    // 3 - largest (default)
    //
    // The trick here is that we change the order of coordinates according to the alignment,
    // so that the intersection calculation below can stay the same without having to cover
    // all cases. So, the case distinction is implicitly done here
    // CAUTION: If you change anything here, you have to adjust the re-alignment further down below accordingly!
    // (can most probably be done better)
    int alignment = 0;
    vec3 aligned_absradii = absradii;
    if(orientation == 3) {
        if(absradii.y >= absradii.x && absradii.y >= absradii.z) {
            alignment = 1;
            aligned_absradii = absradii.yxz;
        }
        else if(absradii.z >= absradii.x && absradii.z >= absradii.y){
            alignment = 2;
            aligned_absradii = absradii.zxy;
        }
    }
    if(orientation == 1) {
        alignment = 1;
        aligned_absradii = absradii.yxz;
    }
    if(orientation == 2) {
        alignment = 2;
        aligned_absradii = absradii.zxy;
    }


    // INTERSECTION CALCULATION
    // cylinder and cone parameters
    float length_cylinder = aligned_absradii.x * radius_scaling;
    float length_cylinder_half = length_cylinder / 2.0;
    float min_radius = min(aligned_absradii.y, aligned_absradii.z);
    float radius_cylinder = 0.8 * min_radius * radius_scaling;
    float radius_cone = min_radius * radius_scaling;
    float height_cone = aligned_absradii.z * radius_scaling;

    // calc the viewing ray
    vec3 shift = vec3((length_cylinder - radius_cylinder) * 0.7, 0.0, 0.0);

    if(alignment == 1) {
        shift = shift.yxz; // only matters that x is in middle
    }
    else if(alignment == 2) {
        shift = shift.yzx; // only matters that x is last
    }

    vec3 shifted_cam = cam_pos - shift;
    vec3 ray = normalize( (os_coord - shift) - shifted_cam );
    vec3 aligned_cam = shifted_cam;

    // re-assign coordinates to account for the alignment change
    // this way the code below doesn't need to be changed
    if(alignment == 1) {
        ray = ray.yxz;
        aligned_cam = aligned_cam.yxz;
    }
    else if(alignment == 2) {
        ray = ray.zxy;
        aligned_cam = aligned_cam.zxy;
    }

    // helpers needed later
    float ray_dot =      dot(ray.yz, ray.yz);
    float ray_cpos_dot = dot(ray.yz, aligned_cam.yz);
    float cpos_dot =     dot(aligned_cam.yz, aligned_cam.yz);

    // early exit check if cam is too close or within arrow
    // (actually just checks if it is within the cylinder)
    if(cpos_dot <= 1.5 * radius_cylinder * radius_cylinder &&
       aligned_cam.x >= -length_cylinder &&
       aligned_cam.x <= height_cone) {
        discard;
    }


    ////////////////////////////////////////
    // CYLINDER
    // see: https://pbr-book.org/3ed-2018/Shapes/Cylinders
    ////////////////////////////////////////
    // a cylinder intersection test
    float a_cyl = ray_dot;
    float b_cyl = 2.0 * ray_cpos_dot;
    float c_cyl = cpos_dot - radius_cylinder * radius_cylinder;
    float div_cyl = 2.0 * a_cyl;
    float radicand_cyl = b_cyl * b_cyl - 4.0 * a_cyl * c_cyl;


    ////////////////////////////////////////
    // CONE
    // see: https://pbr-book.org/3ed-2018/Shapes/Other_Quadrics
    ////////////////////////////////////////
    float k = radius_cone / height_cone;
    k = k * k;
    float cam_x_minus_height = aligned_cam.x - height_cone;
    float a_cone = ray_dot - k * ray.x * ray.x;
    float b_cone = 2.0 * (ray_cpos_dot - k * ray.x * cam_x_minus_height);
    float c_cone = cpos_dot - k * cam_x_minus_height * cam_x_minus_height;
    float div_cone = 2.0 * a_cone;
    float radicand_cone = b_cone * b_cone - 4.0 * a_cone * c_cone;


    // calculate the geometry-ray-intersection
    vec2 radicand = vec2(radicand_cyl, radicand_cone);
    vec2 divisor = vec2(div_cyl, div_cone);
    vec2 radix = sqrt(radicand);
    vec2 minus_b = vec2(-b_cyl, -b_cone);

    vec4 lambda = vec4(
        (minus_b.x - radix.x) / divisor.x,          // near cylinder
        (minus_b.y + radix.y) / divisor.y,          // far cone
        (minus_b.x + radix.x) / divisor.x,          // far cylinder
        (minus_b.y - radix.y) / divisor.y);         // near cone

    bvec4 invalid = bvec4(
        (divisor.x == 0.0) || (radicand.x < 0.0),   // cylinder
        (divisor.y == 0.0) || (radicand.y < 0.0),   // cone
        (divisor.x == 0.0) || (radicand.x < 0.0),   // cylinder
        (divisor.y == 0.0) || (radicand.y < 0.0));  // cone


    // ix.x = near cylinder intersection
    // ix.y = far cone intersection
    // ix.z = far cylinder intersection
    // ix.w = near cone intersection
    vec4 ix = aligned_cam.xxxx + ray.xxxx * lambda;

    // is near cylinder hit in bounds?
    invalid.x = invalid.x || (ix.x > 0.0) || (ix.x < -length_cylinder);
    // is far cone hit in bounds and do we hit the disk on the cone side?
    invalid.y = invalid.y || !(((ix.y < height_cone) || (ix.w < 0.0)) && (ix.y > 0.0));
    // is far cylinder hit in bounds and do we hit the disk on the left side?
    invalid.z = invalid.z || !(((ix.z < 0.0) || (ix.x < -length_cylinder)) && (ix.z > -length_cylinder));
    // is near cone in bounds?
    invalid.w = invalid.w || (ix.w < 0.0) || (ix.w > height_cone);

    if (invalid.x && invalid.y && invalid.z && invalid.w) {
        discard;
    }


    // default disk normal
    // arrow looks in positive axis-direction, therefore normal has to look the opposite way
    vec3 normal = vec3(-1.0, 0.0, 0.0);
    vec3 intersection = vec3(0.0);

    // be aware of coordinate order for alignments
    // alignment 0 --> xyz
    // alignment 1 --> yxz
    // alignment 2 --> zxy

    // cone
    if (!invalid.w) {
        invalid.xyz = bvec3(true, true, true);
        intersection = aligned_cam + (ray * lambda.w);
        normal = normalize(vec3(radius_cone / height_cone, normalize(intersection.yz)));
    }
    // cylinder
    if (!invalid.x) {
        invalid.zy = bvec2(true, true);
        intersection = aligned_cam + (ray * lambda.x);
        normal = normalize(vec3(0.0, normalize(intersection.yz)));
    }
    // no need for normal alignment adjustment for disks,
    // since it is already implicitly done when normal is initialized
    // left cylinder disk
    if (!invalid.z) {
        invalid.y = true;
        lambda.z = (-length_cylinder - aligned_cam.x) / ray.x;
        intersection = aligned_cam + (ray * lambda.z);
    }
    // cone disk
    if (!invalid.y) {
        lambda.w = (0.0 - aligned_cam.x) / ray.x;
        intersection = aligned_cam + (ray * lambda.w);
        float pyth = dot(intersection.yz, intersection.yz);
        if(pyth > radius_cone * radius_cone) {
            discard;
        }
    }

    // re-re-align coordinates
    if(alignment == 1) {
        intersection = intersection.yxz;
        normal = normal.yxz;
    }
    else if(alignment == 2) {
        intersection = intersection.yzx;
        normal = normal.yzx;
    }

    // re-shift
    intersection += shift;

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
    albedo_out = vec4(mix(dir_color, vert_color.rgb, color_interpolation), 1.0);
    normal_out = normal;
    depth_out = depth_ss; // can probably removed at some point (SimpleRrenderTarget doesn't use it anymore)
    gl_FragDepth = depth_ss;
}
