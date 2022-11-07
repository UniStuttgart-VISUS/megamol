/**
 * MegaMol
 * Copyright (c) 2022, MegaMol Dev Team
 * All rights reserved.
 */

in vec4 obj_pos;
in vec3 cam_pos;
in flat vec3 radii;
in vec3 absradii;
in mat3 rotate_world_into_tensor;

in vec4 vert_color;

uniform float arrow_thickness;

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
    vec3 ray_base = normalize(tmp - cam_pos);

    bvec3 discard_frag = bvec3(false);
    vec3 normal;
    vec3 normals[3] = {
        vec3(0.0),
        vec3(0.0),
        vec3(0.0)
    };
    vec3 intersection;
    vec3 intersections[3] = {
        vec3(10000.0),
        vec3(10000.0),
        vec3(10000.0)
    };
    vec3 arrow_dir;
    vec3 arrow_dirs[3] = {
        vec3(1.0, 0.0, 0.0),
        vec3(0.0, 1.0, 0.0),
        vec3(0.0, 0.0, 1.0)
    };
    vec3 intersection_lengths = vec3(10000.0);

    for(int i = 0; i < 3; ++i) {
        // TODO: is something not reset here?
        normal = vec3(0.0);
        intersection = vec3(10000.0);
        vec3 ray = ray_base;

        int orientation = i;
        vec3 aligned_absradii = absradii;
        int alignment = 0;
        // check which axis is used alignment
        // 0 - x (default)
        // 1 - y
        // 2 - z
        if(orientation == 1) {
            alignment = 1;
            aligned_absradii = absradii.yxz;
        }
        if(orientation == 2) {
            alignment = 2;
            aligned_absradii = absradii.zxy;
        }


        // cylinder length
        float length_cylinder = 0.8 * aligned_absradii.x * radius_scaling;
        float length_cylinder_half = length_cylinder / 2.0;
        float radius_cylinder = arrow_thickness; // controlled via paramslot later --> arrow thickness
        float radius_cone = 1.5 * radius_cylinder * radius_scaling;
        float height_cone = 0.2 * aligned_absradii.x * radius_scaling;

        // shift cam, so that end of cylinder = start of cone = (0,0,0)
        vec3 shift = vec3(0.0);
        float shift_value = length_cylinder;
        shift.x = alignment == 0 ? shift_value : 0.0;
        shift.y = alignment == 1 ? shift_value : 0.0;
        shift.z = alignment == 2 ? shift_value : 0.0;
        vec3 cpos = cam_pos - shift;

        // re-assign coordinates to account for the alignment change
        // this way the code below doesn't need to be changed
        if(alignment == 1) {
            ray = ray_base.yxz;
            cpos = cpos.yxz;
        }
        else if(alignment == 2) {
            ray = ray_base.zxy;
            cpos = cpos.zxy;
        }

        // helpers needed later
        float ray_dot =      dot(ray.yz, ray.yz);
        float ray_cpos_dot = dot(ray.yz, cpos.yz);
        float cpos_dot =     dot(cpos.yz, cpos.yz);

        // early exit check if cam is too close or within arrow
        // (actually just checks if it is within the cylinder)
        if(cpos_dot <= 1.5 * radius_cylinder * radius_cylinder &&
        cpos.x >= -length_cylinder &&
        cpos.x <= height_cone) {
            discard_frag[i] = true;
            continue;
            //discard;
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
        // TODO: which cone height to use? one of the radii of half the length of the cylinder or whatever?
        float k = radius_cone / height_cone;
        k = k * k;
        float cam_x_minus_height = cpos.x - height_cone;
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
            discard_frag[i] = true;
            continue;
            //discard;
        }

        // default disk normal
        // arrow looks in positive axis-direction, therefore normal has to look the opposite way
        vec3 normal = vec3(-1.0, 0.0, 0.0);

        // be aware of coordinate order for alignments
        // alignment 0 --> xyz
        // alignment 1 --> yxz
        // alignment 2 --> zxy

        // cone
        if (!invalid.w) {
            invalid.xyz = bvec3(true, true, true);
            intersection = cpos + (ray * lambda.w);
            normal = normalize(vec3(TIP_RAD / TIP_LEN, normalize(intersection.yz)));
        }
        // cylinder
        if (!invalid.x) {
            invalid.zy = bvec2(true, true);
            intersection = cpos + (ray * lambda.x);
            normal = normalize(vec3(0.0, normalize(intersection.yz)));
        }
        // no need for alignment adjustment for disks, since it is already done
        // when normal is initialized
        // left cylinder disk
        if (!invalid.z) {
            invalid.y = true;
            lambda.z = (-CYL_LEN - cpos.x) / ray.x;
            intersection = cpos + (ray * lambda.z);
        }
        // cone disk
        if (!invalid.y) {
            lambda.w = (0.0 - cpos.x) / ray.x;
            intersection = cpos + (ray * lambda.w);
            float pyth = dot(intersection.yz, intersection.yz);
            if(pyth > radius_cone * radius_cone) {
                discard_frag[i] = true;
                continue;
                //discard;
            }
        }

        // this is a problem I think!
        // because of rotation, it is possible, that a further away intersection
        // gets closer than a previously closer intersection after rotation
        intersection_lengths[i] = length(intersection - cpos);

        // re-re-align coordinates
        if(alignment == 1) {
            intersection = intersection.yxz;
            normal = normal.yxz;
        }
        else if(alignment == 2) {
            intersection = intersection.yzx;
            normal = normal.yzx;
        }

        normals[i] = normal;
        intersections[i] = intersection;
    }

    // is there a glsl command to check all? vec3() == vec3(true) or smth like that?
    // 'early' exit
    if(discard_frag.x == true && discard_frag.y == true && discard_frag.z == true) {
        discard;
    }
    // TODO: since ssao result is not optimal, there may be something wrong here
    // decide, which intersection is closest to camera
    // TODO: also, some cases are missing
    int inst = 0;
    if(intersection_lengths[1] < intersection_lengths[0] && intersection_lengths[1] < intersection_lengths[2]
        && discard_frag.y == false) {
        inst = 1;
    } else if(intersection_lengths[2] < intersection_lengths[0] && intersection_lengths[2] < intersection_lengths[1]
        && discard_frag.z == false) {
        inst = 2;
    }

    // transform normal and intersection point into tensor
    normal = transpose(rotate_world_into_tensor) * normals[inst];
    normal = normalize(normal);

    // translate point back to original position
    intersection = inverse(rotate_world_into_tensor) * intersections[inst];
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


    vec3 dir_color1 = max(vec3(0), arrow_dirs[inst] * sign(radii));
    vec3 dir_color2 = vec3(1) + arrow_dirs[inst] * sign(radii);
    vec3 dir_color = any(lessThan(dir_color2, vec3(0.5))) ? dir_color2 * vec3(0.5) : dir_color1;

    albedo_out = vec4(mix(dir_color, vert_color.rgb, color_interpolation),1.0);
    normal_out = normal;
    depth_out = depth_ss;
    gl_FragDepth = depth_ss;
}
