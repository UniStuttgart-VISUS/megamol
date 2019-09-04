    
layout(packed, binding = 0) buffer pos_data {
    vec3 pos[];
};
layout(packed, binding = 1) buffer quat_data {
    vec4 quat[];
};
layout(packed, binding = 2) buffer rad_data {
    vec3 rad[];
};
layout(packed, binding = 3) buffer col_data {
    vec4 col[];
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

void main() {
    const vec4 quatConst = vec4(1.0, -1.0, 0.5, 0.0);
    vec4 tmp, tmp1;
    vec3 tmp2;
    
    vec4 inPos = vec4(pos[gl_VertexID], 1.0);
    vec3 radii = rad[gl_VertexID];
    vec3 absradii = abs(radii);
    vec4 quatC = quat[gl_VertexID];
    invRad = 1.0 / absradii;
    
    inPos.w = 1.0;
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

    // calculate cam position
    tmp = MV_I[3]; // (C) by Christoph
    tmp.xyz -= objPos.xyz; // cam move
    // TODO: cam rotation mit rotMat transposed
    //camPos.x = dot(rotMatT0, tmp.xyz);
    //camPos.y = dot(rotMatT1, tmp.xyz);
    //camPos.z = dot(rotMatT2, tmp.xyz);
    camPos.xyz = rotMatT0 * tmp.x + rotMatT1 * tmp.y + rotMatT2 * tmp.z;

    camPos.xyz *= invRad;


    // calculate light position in glyph space
    //lightPos = gl_ModelViewMatrixInverse * gl_LightSource[0].position - objPos; // note: w is bs now!
    // TODO: light rotation mit rotMat transposed
    //lightPos.x = dot(rotMatT0, lightPos.xyz);
    //lightPos.y = dot(rotMatT1, lightPos.xyz);
    //lightPos.z = dot(rotMatT2, lightPos.xyz);
    //lightPos.xyz = rotMatT0 * lightPos.x + rotMatT1 * lightPos.y + rotMatT2 * lightPos.z;
    // Do not scale light position, so we can use the "faked" parameter space normals
    lightPos = MV_T * light; // transpose of inverse inverse -> directional light
    
    //lightPos.xyz *= invRad;


    // send color to fragment shader
    // WIP todo integrate to bitflags and tf stuff by matthias
    if (bool(useGlobalColor)) {
        vertColor = globalColor;
    } else {
        if (bool(intensityOnly)) {
            
        } else {
            vertColor = col[gl_VertexID];
        }
    }


    // calculate sprite position and size
    vec2 winHalf = 2.0 / viewAttr.zw; // window size

    // lumberjack™ approach
    vec4 pos, projPos;
    vec4 pX, pY, pOP;
    vec2 mins, maxs, pp;

    projPos.w = 0.0;

    pos = vec4(-1.0, -1.0, -1.0, 1.0);
    pos.xyz *= absradii; // scale
    projPos.x = dot(rotMatT0, pos.xyz); // rotate
    projPos.y = dot(rotMatT1, pos.xyz);
    projPos.z = dot(rotMatT2, pos.xyz);
    pos = objPos + projPos; // move
    pos.w = 1.0; // now we're in object space
    pOP = MVP * pos;
    pp = pOP.xy / pOP.w; // pp = (-1, -1, -1)
    mins = maxs = pp;

    pos = vec4(1.0, -1.0, -1.0, 1.0);
    pos.xyz *= absradii; // scale
    projPos.x = dot(rotMatT0, pos.xyz); // rotate
    projPos.y = dot(rotMatT1, pos.xyz);
    projPos.z = dot(rotMatT2, pos.xyz);
    pos = objPos + projPos; // move
    pos.w = 1.0; // now we're in object space
    projPos = MVP * pos;
    pX = projPos - pOP;
    pp = projPos.xy / projPos.w; // pp = (1, -1, -1)
    mins = min(mins, pp);
    maxs = max(maxs, pp);

    pos = vec4(-1.0, 1.0, -1.0, 1.0);
    pos.xyz *= absradii; // scale
    projPos.x = dot(rotMatT0, pos.xyz); // rotate
    projPos.y = dot(rotMatT1, pos.xyz);
    projPos.z = dot(rotMatT2, pos.xyz);
    pos = objPos + projPos; // move
    pos.w = 1.0; // now we're in object space
    projPos = MVP * pos;
    pY = projPos - pOP;
    pp = projPos.xy / projPos.w; // pp = (-1, 1, -1)
    mins = min(mins, pp);
    maxs = max(maxs, pp);

    projPos += pX;
    pp = projPos.xy / projPos.w; // pp = (1, 1, -1)
    mins = min(mins, pp);
    maxs = max(maxs, pp);

    pos = vec4(-1.0, -1.0, 1.0, 1.0);
    pos.xyz *= absradii; // scale
    projPos.x = dot(rotMatT0, pos.xyz); // rotate
    projPos.y = dot(rotMatT1, pos.xyz);
    projPos.z = dot(rotMatT2, pos.xyz);
    pos = objPos + projPos; // move
    pos.w = 1.0; // now we're in object space
    pOP = MVP * pos;
    pp = pOP.xy / pOP.w; // pp = (-1, -1, 1)
    mins = min(mins, pp);
    maxs = max(maxs, pp);

    projPos = pOP + pY;
    pp = projPos.xy / projPos.w; // pp = (-1, 1, 1)
    mins = min(mins, pp);
    maxs = max(maxs, pp);

    projPos += pX;
    pp = projPos.xy / projPos.w; // pp = (1, 1, 1)
    mins = min(mins, pp);
    maxs = max(maxs, pp);

    projPos = pOP + pX;
    pp = projPos.xy / projPos.w; // pp = (1, -1, 1)
    mins = min(mins, pp);
    maxs = max(maxs, pp);

    gl_Position = vec4((mins + maxs) * 0.5, 0.0, 1.0);
    maxs = (maxs - mins) * 0.5 * winHalf;
    gl_PointSize = max(maxs.x, maxs.y);
    // #ifdef SMALL_SPRITE_LIGHTING
    // // for normal crowbaring on very small sprites
    // lightPos.w = (clamp(gl_PointSize, 1.0, 5.0) - 1.0) / 4.0;
    // #endif // SMALL_SPRITE_LIGHTING

    // #ifdef RETICLE
    // centerFragment = gl_Position.xy / gl_Position.w;
    // #endif //RETICLE
}