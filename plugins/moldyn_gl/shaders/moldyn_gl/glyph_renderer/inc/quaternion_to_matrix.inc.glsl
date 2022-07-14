mat3 quaternion_to_matrix(vec4 quat) {
    const vec4 quat_const = vec4(1.0, -1.0, 0.5, 0.0);
    vec4 tmp, tmp1;

    // rotation matrix columns from the quaternion
    vec3 rot_mat_c0;
    vec3 rot_mat_c1;
    vec3 rot_mat_c2;

    tmp = quat.xzyw * quat.yxzw;
    tmp1 = quat * quat.w;
    tmp1.w = -quat_const.z;

    rot_mat_c0.xyz = tmp1.wzy * quat_const.xxy + tmp.wxy;  // matrix0 <- (ww-0.5, xy+zw, xz-yw, %)
    rot_mat_c0.x = quat.x * quat.x + rot_mat_c0.x;          // matrix0 <- (ww+x*x-0.5, xy+zw, xz-yw, %)
    rot_mat_c0 = rot_mat_c0 + rot_mat_c0;                     // matrix0 <- (2(ww+x*x)-1, 2(xy+zw), 2(xz-yw), %)

    rot_mat_c1.xyz = tmp1.zwx * quat_const.yxx + tmp.xwz;  // matrix1 <- (xy-zw, ww-0.5, yz+xw, %)
    rot_mat_c1.y = quat.y * quat.y + rot_mat_c1.y;          // matrix1 <- (xy-zw, ww+y*y-0.5, yz+xw, %)
    rot_mat_c1 = rot_mat_c1 + rot_mat_c1;                     // matrix1 <- (2(xy-zw), 2(ww+y*y)-1, 2(yz+xw), %)

    rot_mat_c2.xyz = tmp1.yxw * quat_const.xyx + tmp.yzw;  // matrix2 <- (xz+yw, yz-xw, ww-0.5, %)
    rot_mat_c2.z = quat.z * quat.z + rot_mat_c2.z;          // matrix2 <- (xz+yw, yz-xw, ww+zz-0.5, %)
    rot_mat_c2 = rot_mat_c2 + rot_mat_c2;                     // matrix2 <- (2(xz+yw), 2(yz-xw), 2(ww+zz)-1, %)

    return mat3(rot_mat_c0, rot_mat_c1, rot_mat_c2);
}
