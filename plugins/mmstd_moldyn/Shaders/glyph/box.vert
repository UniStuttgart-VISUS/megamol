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

out vec4 objPos;
out vec4 camPos;
out vec4 lightPos;
out vec4 vertColor;

out vec3 invRad; // ?

out mat3 rotMat;
out vec3 rotMatT0;
out vec3 rotMatT1; // rotation matrix from the quaternion
out vec3 rotMatT2;
out mat3 rotMatIT;

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

    vec4 inPos = vec4(posArray[inst].x, posArray[inst].y, posArray[inst].z, 1.0);
    vec3 radii = vec3(radArray[inst].x, radArray[inst].y, radArray[inst].z); //rad[inst];
    vec3 absradii = abs(radii);
    vec4 quatC = vec4(quatArray[inst].x, quatArray[inst].y, quatArray[inst].z, quatArray[inst].w); //quat[inst];
    invRad = 1.0 / absradii;
    
    objPos = inPos;
    
    tmp = quatC.xzyw * quatC.yxzw;
    tmp1 = quatC * quatC.w;
    tmp1.w = -quatConst.z;
    rotMatT0.xyz = tmp1.wzy * quatConst.xxy + tmp.wxy;	// matrix0 <- (ww-0.5, xy+zw, xz-yw, %)
    rotMatT0.x = quatC.x * quatC.x + rotMatT0.x;			// matrix0 <- (ww+x*x-0.5, xy+zw, xz-yw, %)
    rotMatT0 = rotMatT0 + rotMatT0;                           	// matrix0 <- (2(ww+x*x)-1, 2(xy+zw), 2(xz-yw), %)

    rotMatT1.xyz = tmp1.zwx * quatConst.yxx + tmp.xwz; 	// matrix1 <- (xy-zw, ww-0.5, yz+xw, %)
    rotMatT1.y = quatC.y * quatC.y + rotMatT1.y;     			// matrix1 <- (xy-zw, ww+y*y-0.5, yz+xw, %)
    rotMatT1 = rotMatT1 + rotMatT1;                           	// matrix1 <- (2(xy-zw), 2(ww+y*y)-1, 2(yz+xw), %)

    rotMatT2.xyz = tmp1.yxw * quatConst.xyx + tmp.yzw; 	// matrix2 <- (xz+yw, yz-xw, ww-0.5, %)
    rotMatT2.z = quatC.z * quatC.z + rotMatT2.z;     			// matrix2 <- (xz+yw, yz-xw, ww+zz-0.5, %)
    rotMatT2 = rotMatT2 + rotMatT2;                           	// matrix2 <- (2(xz+yw), 2(yz-xw), 2(ww+zz)-1, %)    
    // End: Holy code!

    rotMatIT = mat3(rotMatT0, rotMatT1, rotMatT2);
    rotMatIT = transpose(rotMatIT);

    normal = cube_normals[corner]; //(MV_T * vec4(cube_normals[corner], 0.0f)).xyz;
    transformedNormal = (MV_T * vec4(normal, 0.0f)).xyz;
    lightPos = MV_T * light; // transpose of inverse inverse -> directional light
    

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

    vec4 pos, projPos;

    pos.xyz = cube_strip[corner];
    pos.xyz *= absradii; // scale
    // projPos.x = dot(rotMatT0, pos.xyz); // rotate
    // projPos.y = dot(rotMatT1, pos.xyz);
    // projPos.z = dot(rotMatT2, pos.xyz);
    projPos.xyz = rotMatIT * pos.xyz;
    //projPos.xyz = pos.xyz;
    projPos.w = 0.0;
    pos = objPos + projPos; // move
    //pos.w = 1.0; // now we're in object space

    objPos = pos;

    gl_Position =  MVP * pos;
}