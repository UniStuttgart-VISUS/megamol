mat3 quaternion_to_matrix(vec4 quat) {
    const vec4 quatConst = vec4(1.0, -1.0, 0.5, 0.0);
    vec4 tmp, tmp1;
    vec3 rotMatC0;
    vec3 rotMatC1; // rotation matrix columns from the quaternion
    vec3 rotMatC2;
    tmp = quat.xzyw * quat.yxzw;
    tmp1 = quat * quat.w;
    tmp1.w = -quatConst.z;
    rotMatC0.xyz = tmp1.wzy * quatConst.xxy + tmp.wxy;	// matrix0 <- (ww-0.5, xy+zw, xz-yw, %)
    rotMatC0.x = quat.x * quat.x + rotMatC0.x;			// matrix0 <- (ww+x*x-0.5, xy+zw, xz-yw, %)
    rotMatC0 = rotMatC0 + rotMatC0;                           	// matrix0 <- (2(ww+x*x)-1, 2(xy+zw), 2(xz-yw), %)

    rotMatC1.xyz = tmp1.zwx * quatConst.yxx + tmp.xwz; 	// matrix1 <- (xy-zw, ww-0.5, yz+xw, %)
    rotMatC1.y = quat.y * quat.y + rotMatC1.y;     			// matrix1 <- (xy-zw, ww+y*y-0.5, yz+xw, %)
    rotMatC1 = rotMatC1 + rotMatC1;                           	// matrix1 <- (2(xy-zw), 2(ww+y*y)-1, 2(yz+xw), %)

    rotMatC2.xyz = tmp1.yxw * quatConst.xyx + tmp.yzw; 	// matrix2 <- (xz+yw, yz-xw, ww-0.5, %)
    rotMatC2.z = quat.z * quat.z + rotMatC2.z;     			// matrix2 <- (xz+yw, yz-xw, ww+zz-0.5, %)
    rotMatC2 = rotMatC2 + rotMatC2;                           	// matrix2 <- (2(xz+yw), 2(yz-xw), 2(ww+zz)-1, %)    

    return mat3(rotMatC0, rotMatC1, rotMatC2);
}