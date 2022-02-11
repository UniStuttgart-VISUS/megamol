// TODO: copyright note of links used?
// TODO: one of the cylinders is not rotated correctly, compare to ellipsoids
/**
 * MegaMol
 * Copyright (c) 2022, MegaMol Dev Team
 * All rights reserved.
 */

in vec4 obj_pos;
in vec3 cam_pos;
in vec3 inv_rad;
// TODO: scale radii below radii --> i.e. x * absradii with 0.0 < absradii < 1.0
// to fit into the box (which is scaled by radii?)
in vec3 absradii;
in mat3 rotate_world_into_tensor;

in vec4 vert_color;
in flat vec3 dir_color;

in flat vec3 normal;
in flat vec3 transformed_normal;

out layout(location = 0) vec4 albedo_out;
out layout(location = 1) vec3 normal_out;
out layout(location = 2) float depth_out;

void solveQuadraticEquation(float a, float b, float c, inout float radicand, inout float t_min, inout float t_max) {
    // solve quadratic equation
    radicand = b * b - 4.0 * a * c;
    // if(radicand < 0.0) {
    //     return;
    // }

    t_min = (-b - sqrt(radicand)) / (2.0 * a);
    t_max = (-b + sqrt(radicand)) / (2.0 * a);
    // if(t_min < 0.0) {
    //     discard;
    // }
}

void main() {
    vec3 intersection = vec3(0.0);
    bool isValidConeHit     = false;
    bool isValidCylinderHit = false;
    bool isValidCylDiskHit  = false;
    bool isValidConeDiskHit = false;

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

    // centered around (0, 0, 0)
    float radius_cylinder = absradii.y * radius_scaling;
    float length_cylinder = absradii.x * radius_scaling;
    float x_max_cylinder = length_cylinder / 2.0;
    float x_min_cylinder = -x_max_cylinder;
    // shift cam to the left, so that middle of cylinder = vec3(0,0,0)
    vec3 cpos = cam_pos - vec3(x_max_cylinder, 0.0, 0.0);


    ////////////////////////////////////////
    // CONE
    // see: https://pbr-book.org/3ed-2018/Shapes/Other_Quadrics
    ////////////////////////////////////////
    float radius_cone = 1.5 * radius_cylinder;
    // TODO: which cone height to use? one of the radii of half the length of the cylinder or whatever?
    float height_cone = absradii.z * radius_scaling;
    float x_min_cone = x_max_cylinder;
    float x_max_cone = x_max_cylinder + height_cone;
    float k = radius_cone / height_cone;
    k = k * k;
    // TODO: optimize
    // e.g. ray.y * ray.y is also used in the cylinder intersection test
    // reuse it!
    float cam_x_minus_height = cpos.x - height_cone;
    float a_cone = ray.y * ray.y + ray.z * ray.z - k * ray.x * ray.x;
    float b_cone = 2.0 * (ray.y * cpos.y + ray.z * cpos.z - k * ray.x * cam_x_minus_height);
    float c_cone = cpos.y * cpos.y + cpos.z * cpos.z - k * cam_x_minus_height * cam_x_minus_height;
    float div_cone = 2.0 * a_cone;

    float t_min_cone = 0.0;
    float t_max_cone = 0.0;
    float radicand_cone = 0.0;
    solveQuadraticEquation(a_cone, b_cone, c_cone, radicand_cone, t_min_cone, t_max_cone);

    // TODO: all intersection point can be optimized by just calculating the x-coord
    // all others are not needed until the final intersection calculation
    vec3 front_intersection_cone = cpos + t_min_cone * ray;
    vec3 back_intersection_cone = cpos + t_max_cone * ray;


    ////////////////////////////////////////
    // CYLINDER
    // see: https://pbr-book.org/3ed-2018/Shapes/Cylinders
    ////////////////////////////////////////
    // TODO: remove helpers and plug them in below directly when done
    float ray1 = ray.y;
    float ray2 = ray.z;
    float cam1 = cpos.y;
    float cam2 = cpos.z;

    // a cylinder intersection test
    float a_cyl = ray1 * ray1 + ray2 * ray2; //dot(ray.yz, ray.yz)
    float b_cyl = 2.0 * (ray1 * cam1 + ray2 * cam2); // 2.0 * dot(ray.yz, cpos.yz)
    float c_cyl = cam1 * cam1 + cam2 * cam2 - radius_cylinder * radius_cylinder; // dot(cpos.yz, cpos.yz) - radius_cylinder * radius_cylinder
    float div_cyl = 2.0 * a_cyl;

    // solve quadratic equation
    float t_min_cyl = 0.0;
    float t_max_cyl = 0.0;
    float radicand_cyl = 0.0;
    solveQuadraticEquation(a_cyl, b_cyl, c_cyl, radicand_cyl, t_min_cyl, t_max_cyl);

    vec3 front_intersection_cyl = cpos + t_min_cyl * ray;
    vec3 back_intersection_cyl = cpos + t_max_cyl * ray;


    ////////////////////////////////////////
    // DISKS
    // see: https://pbr-book.org/3ed-2018/Shapes/Disks
    ////////////////////////////////////////
    // intersect disks to close the cylinder
    float t_disk_cone = (x_max_cylinder - cpos.x) / ray.x;
    float t_disk_left = (x_min_cylinder - cpos.x) / ray.x;
    vec3 disk_intersection_cone = cpos + t_disk_cone * ray;
    vec3 disk_intersection_left = cpos + t_disk_left * ray;


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
#define CYL_LEN_HALF (x_max_cylinder)
#define TIP_RAD (radius_cone)
#define TIP_LEN (height_cone)

    // ix.x = near cylinder intersection
    // ix.y = far cone intersection
    // ix.z = far cylinder intersection
    // ix.w = near cone intersection
    vec4 ix = cpos.xxxx + ray.xxxx * lambda;

    // is near cylinder hit in bounds?
    //invalid.x = invalid.x || (ix.x < TIP_LEN) || (ix.x > CYL_LEN);
    invalid.x = invalid.x || (ix.x > 0.0) || (ix.x < -CYL_LEN);
    // is far cone hit in bounds and do we hit the disk on the cone side?
    //invalid.y = invalid.y || (ix.y < 0.0) || (ix.y > TIP_LEN);
    invalid.y = invalid.y || !(((ix.y < TIP_LEN) || (ix.w < 0.0)) && (ix.y > 0.0));
    // is far cylinder hit in bounds and do we hit the disk on the left side?
    //invalid.z = invalid.z || !(((ix.z > TIP_LEN) || (ix.x > CYL_LEN)) && (ix.z < CYL_LEN));
    invalid.z = invalid.z || !(((ix.z < 0.0) || (ix.x < -CYL_LEN)) && (ix.z > -CYL_LEN));
    // is near cone in bounds?
    //invalid.w = invalid.w || !((ix.w > 0.0) && (ix.w < TIP_LEN));
    invalid.w = invalid.w || (ix.w < 0.0) || (ix.w > TIP_LEN);

    if (invalid.x && invalid.y && invalid.z && invalid.w) {
        discard;
    }

    vec3 normal = vec3(1.0, 0.0, 0.0);

    // cone
    if (!invalid.w) {
        //if(lambda.w < lambda.x) {
            invalid.xyz = bvec3(true, true, true);
            intersection = vec3(0.0, 0.0, 1.0);//cpos.xyz + (ray * lambda.w);
            normal = normalize(vec3(-TIP_RAD / TIP_LEN, normalize(intersection.yz)));
        //}
    }
    // cylinder
    if (!invalid.x) {
        invalid.zy = bvec2(true, true);
        intersection = vec3(1.0, 1.0, 1.0);//cpos.xyz + (ray * lambda.x);
        normal = vec3(0.0, normalize(intersection.yz));
    }
    // left cylinder disk
    if (!invalid.z) {
        invalid.y = true;
        lambda.z = (CYL_LEN - cpos.x) / ray.x;
        intersection = vec3(1.0, 0.0, 0.0);//cpos.xyz + (ray * lambda.z);
        normal = vec3(-1.0, 0.0, 0.0);
    }
    // cone disk
    if (!invalid.y) {
        lambda.w = (CYL_LEN_HALF - cpos.x) / ray.x;
        intersection = cpos.xyz + (ray * lambda.w);
        float pyth = dot(intersection.yz, intersection.yz);
        if(pyth > radius_cone * radius_cone) {
            discard;
        }
        intersection = vec3(0.0, 1.0, 0.0);
        normal = vec3(-1.0, 0.0, 0.0);
    }

    normal = rotate_world_into_tensor * normal;


    // TODO: adjust dir_color
    //albedo_out = vec4(intersection, 1.0);
    albedo_out = vec4(mix(dir_color, vert_color.rgb, color_interpolation),1.0);
    // TODO: adjust normal, transformed_normal is not correct
    normal_out = normal;
    //depth_out = gl_FragCoord.z;


    vec4 ding = vec4(intersection, 1.0);
    float depth = dot(mvp_t[2], ding);
    float depth_w = dot(mvp_t[3], ding);
    depth_out = ((depth / depth_w) + 1.0) * 0.5;
}
