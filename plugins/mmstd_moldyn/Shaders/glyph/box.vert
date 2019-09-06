struct pos {
    float x;
    float y;
    float z;
};

struct quat {
    float x;
    float y;
    float z;
    float w;
};

layout(std430, binding = 0) buffer pos_data {
    pos posArray[];
};
layout(std430, binding = 1) buffer quat_data {
    quat quatArray[];
};
layout(std430, binding = 2) buffer rad_data {
    pos radArray[];
};
layout(std430, binding = 3) buffer col_data {
    quat colArray[];
};
// binding 4 is going to be the flags

out vec4 wsPos;
out vec4 vertColor;

out vec3 invRad;

out flat vec3 dirColor;
out flat vec3 normal;
out flat vec3 transformedNormal;

// https://stackoverflow.com/questions/28375338/cube-using-single-gl-triangle-strip
const vec3 cube_strip[14] = vec3[14](
    vec3(-1.f, 1.f, 1.f),     // Front-top-left
    vec3(1.f, 1.f, 1.f),      // Front-top-right
    vec3(-1.f, -1.f, 1.f),    // Front-bottom-left
    vec3(1.f, -1.f, 1.f),     // Front-bottom-right
    vec3(1.f, -1.f, -1.f),    // Back-bottom-right
    vec3(1.f, 1.f, 1.f),      // Front-top-right
    vec3(1.f, 1.f, -1.f),     // Back-top-right
    vec3(-1.f, 1.f, 1.f),     // Front-top-left
    vec3(-1.f, 1.f, -1.f),    // Back-top-left
    vec3(-1.f, -1.f, 1.f),    // Front-bottom-left
    vec3(-1.f, -1.f, -1.f),   // Back-bottom-left
    vec3(1.f, -1.f, -1.f),    // Back-bottom-right
    vec3(-1.f, 1.f, -1.f),    // Back-top-left
    vec3(1.f, 1.f, -1.f)      // Back-top-right
);

// provoking vertex is usually LAST
const vec3 cube_normals[14] = vec3[14](
    vec3(0.0f,0.0f,0.0f),     // never last
    vec3(0.0f,0.0f,0.0f),     // never last
    vec3(0.f, 0.f, 1.f),    // front1
    vec3(0.f, 0.f, 1.f),     // front2
    vec3(0.f, -1.f, 0.f),    // bottom1
    vec3(1.f, 0.f, 0.f),      // right1
    vec3(1.f, 0.f, 0.f),     // right2
    vec3(0.f, 1.f, 0.f),     // top1
    vec3(0.f, 1.f, 0.f),    // top2
    vec3(-1.f, 0.f, 0.f),    // left1
    vec3(-1.f, 0.f, 0.f),   // left2
    vec3(0.f, -1.f, 0.f),    // bottom2
    vec3(0.f, 0.f, -1.f),    // back1
    vec3(0.f, 0.f, -1.f)      // back2
);

void main() {
    const vec4 quatConst = vec4(1.0, -1.0, 0.5, 0.0);
    vec4 tmp, tmp1;
    vec3 tmp2;
    
    uint inst = gl_InstanceID; //gl_VertexID / 14;
    uint corner = gl_VertexID;// % 14;

    vec3 rotMatC0;
    vec3 rotMatC1; // rotation matrix columns from the quaternion
    vec3 rotMatC2;
    vec4 inPos = vec4(posArray[inst].x, posArray[inst].y, posArray[inst].z, 1.0);
    vec3 radii = vec3(radArray[inst].x, radArray[inst].y, radArray[inst].z); //rad[inst];
    vec3 absradii = abs(radii);
    vec4 quatC = vec4(quatArray[inst].x, quatArray[inst].y, quatArray[inst].z, quatArray[inst].w); //quat[inst];
    invRad = 1.0 / absradii;
    
    tmp = quatC.xzyw * quatC.yxzw;
    tmp1 = quatC * quatC.w;
    tmp1.w = -quatConst.z;
    rotMatC0.xyz = tmp1.wzy * quatConst.xxy + tmp.wxy;	// matrix0 <- (ww-0.5, xy+zw, xz-yw, %)
    rotMatC0.x = quatC.x * quatC.x + rotMatC0.x;			// matrix0 <- (ww+x*x-0.5, xy+zw, xz-yw, %)
    rotMatC0 = rotMatC0 + rotMatC0;                           	// matrix0 <- (2(ww+x*x)-1, 2(xy+zw), 2(xz-yw), %)

    rotMatC1.xyz = tmp1.zwx * quatConst.yxx + tmp.xwz; 	// matrix1 <- (xy-zw, ww-0.5, yz+xw, %)
    rotMatC1.y = quatC.y * quatC.y + rotMatC1.y;     			// matrix1 <- (xy-zw, ww+y*y-0.5, yz+xw, %)
    rotMatC1 = rotMatC1 + rotMatC1;                           	// matrix1 <- (2(xy-zw), 2(ww+y*y)-1, 2(yz+xw), %)

    rotMatC2.xyz = tmp1.yxw * quatConst.xyx + tmp.yzw; 	// matrix2 <- (xz+yw, yz-xw, ww-0.5, %)
    rotMatC2.z = quatC.z * quatC.z + rotMatC2.z;     			// matrix2 <- (xz+yw, yz-xw, ww+zz-0.5, %)
    rotMatC2 = rotMatC2 + rotMatC2;                           	// matrix2 <- (2(xz+yw), 2(yz-xw), 2(ww+zz)-1, %)    
    // End: Holy code!

    mat3 rotate_world_into_tensor = mat3(rotMatC0, rotMatC1, rotMatC2);
    mat3 rotate_points = transpose(rotate_world_into_tensor);
    mat3 rotate_vectors = transpose(inverse(rotate_points));

    normal = cube_normals[corner]; //(MV_T * vec4(cube_normals[corner], 0.0f)).xyz;

    vec3 dirColor1 = max(vec3(0), normal * sign(radii));
    vec3 dirColor2 = vec3(1) + normal * sign(radii);

    dirColor = any(lessThan(dirColor2, vec3(0.5)))? dirColor2 * vec3(0.5) : dirColor1;

    transformedNormal = (rotate_vectors * normal).xyz;
    //lightDir = vec4(//vec4(rotate_wit_IT * light.xyz, 0.0);
    

    // send color to fragment shader
    // WIP todo integrate to bitflags and tf stuff by matthias
    if (bool(useGlobalColor)) {
        vertColor = globalColor;
    } else {
        if (bool(intensityOnly)) {
            
        } else {
            vertColor = vec4(colArray[inst].x, colArray[inst].y, colArray[inst].z, colArray[inst].w);;
        }
    }

    vec4 cornerPos;

    cornerPos.xyz = cube_strip[corner];
    cornerPos.xyz *= absradii; // scale
    cornerPos.xyz = rotate_points * cornerPos.xyz;
    cornerPos.w = 0.0;
    wsPos = inPos + cornerPos; // move

    gl_Position =  MVP * wsPos;
}