// TODO: copyright note of links used?
// TODO: one of the cylinders is not rotated correctly, compare to ellipsoids
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
    coord -= obj_pos; // ... and move

    // ... and rotate with rot_mat transposed
    vec3 tmp = rotate_world_into_tensor * coord.xyz;

    // calc the viewing ray
    vec3 ray = normalize(tmp - cam_pos);
    // TODO: which radius to use? y or z?

    // cylinder length
    float length_cylinder = absradii.x * radius_scaling;
    float length_cylinder_half = length_cylinder / 2.0;

    // shift cam to the left, so that end of cylinder = start of cone = (0,0,0)
    vec3 cpos = cam_pos - vec3(length_cylinder_half, 0.0, 0.0);

    // helpers needed later
    float ray_yz_dot =      dot(ray.yz, ray.yz);
    float ray_cpos_yz_dot = dot(ray.yz, cpos.yz);
    float cpos_yz_dot =     dot(cpos.yz, cpos.yz);

    ////////////////////////////////////////
    // CYLINDER
    // see: https://pbr-book.org/3ed-2018/Shapes/Cylinders
    ////////////////////////////////////////
    float radius_cylinder = absradii.y * radius_scaling;

    // a cylinder intersection test
    float a_cyl = ray_yz_dot;
    float b_cyl = 2.0 * ray_cpos_yz_dot;
    float c_cyl = cpos_yz_dot - radius_cylinder * radius_cylinder;
    float div_cyl = 2.0 * a_cyl;
    float radicand_cyl = b_cyl * b_cyl - 4.0 * a_cyl * c_cyl;


    ////////////////////////////////////////
    // CONE
    // see: https://pbr-book.org/3ed-2018/Shapes/Other_Quadrics
    ////////////////////////////////////////
    float radius_cone = 1.5 * radius_cylinder;
    // TODO: which cone height to use? one of the radii of half the length of the cylinder or whatever?
    float height_cone = absradii.z * radius_scaling;
    float k = radius_cone / height_cone;
    k = k * k;
    float cam_x_minus_height = cpos.x - height_cone;
    float a_cone = ray_yz_dot - k * ray.x * ray.x;
    float b_cone = 2.0 * (ray_cpos_yz_dot - k * ray.x * cam_x_minus_height);
    float c_cone = cpos_yz_dot - k * cam_x_minus_height * cam_x_minus_height;
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

#define CYL_RAD (radius_cylinder)
#define CYL_LEN (length_cylinder)
#define CYL_LEN_HALF (length_cylinder_half)
#define TIP_RAD (radius_cone)
#define TIP_LEN (height_cone)

    // ix.x = near cylinder intersection
    // ix.y = far cone intersection
    // ix.z = far cylinder intersection
    // ix.w = near cone intersection
    vec4 ix = cpos.xxxx + ray.xxxx * lambda;

    // is near cylinder hit in bounds?
    invalid.x = invalid.x || (ix.x > 0.0) || (ix.x < -CYL_LEN);
    // is far cone hit in bounds and do we hit the disk on the cone side?
    invalid.y = invalid.y || !(((ix.y < TIP_LEN) || (ix.w < 0.0)) && (ix.y > 0.0));
    // is far cylinder hit in bounds and do we hit the disk on the left side?
    invalid.z = invalid.z || !(((ix.z < 0.0) || (ix.x < -CYL_LEN)) && (ix.z > -CYL_LEN));
    // is near cone in bounds?
    invalid.w = invalid.w || (ix.w < 0.0) || (ix.w > TIP_LEN);

    if (invalid.x && invalid.y && invalid.z && invalid.w) {
        discard;
    }

    // default disk normal
    // arrow looks in positive x-direction, therefore normal has to look the opposite way
    vec3 normal = vec3(-1.0, 0.0, 0.0);
    vec3 intersection = vec3(0.0);

    // cone
    if (!invalid.w) {
        invalid.xyz = bvec3(true, true, true);
        intersection = cpos + (ray * lambda.w);
        normal = normalize(vec3(-TIP_RAD / TIP_LEN, normalize(intersection.yz)));
    }
    // cylinder
    if (!invalid.x) {
        invalid.zy = bvec2(true, true);
        intersection = cpos + (ray * lambda.x);
        normal = vec3(0.0, normalize(intersection.yz));
    }
    // left cylinder disk
    if (!invalid.z) {
        invalid.y = true;
        lambda.z = (CYL_LEN - cpos.x) / ray.x;
        intersection = cpos + (ray * lambda.z);
    }
    // cone disk
    if (!invalid.y) {
        lambda.w = (0.0 - cpos.x) / ray.x;
        intersection = cpos + (ray * lambda.w);
        float pyth = dot(intersection.yz, intersection.yz);
        if(pyth > radius_cone * radius_cone) {
            discard;
        }
    }

    normal = transpose(rotate_world_into_tensor) * normal;


    //albedo_out = vec4(normal, 1.0);
    albedo_out = vec4(mix(dir_color, vert_color.rgb, color_interpolation),1.0);
    normal_out = normal;

    vec4 ding = vec4(intersection, 1.0);
    float depth = dot(mvp_t[2], ding);
    float depth_w = dot(mvp_t[3], ding);
    depth_out = ((depth / depth_w) + 1.0) * 0.5;
}
