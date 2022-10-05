#version 130

#extension GL_EXT_gpu_shader4 : enable

uniform sampler3D uniGridTex;
uniform sampler3D curlMagTex;
uniform sampler3D randNoiseTex;
uniform sampler3D densityMapTex;
uniform sampler3D colorTex;

uniform int sliceRM;
uniform int plane;
uniform float licDirScl;
uniform int licLen;
uniform int licProj2D;
uniform float licTCScl;

uniform float licContrast;
uniform float licBrightness;

uniform float dataScl;
const float infinity = 100000.0;

float CalcLICAutoScl(int l) {

    float roundOff = 0.05;

    int i;
    vec3 v;
    vec3 stp = gl_TexCoord[0].stp;
//    float vecScl = licDirScl;
    vec4 vecScl;
    float randTCScl = licTCScl*float(textureSize(uniGridTex, 0).x)/float(textureSize(randNoiseTex, 0).x);
    float scl;

    float colLic = texture3D(randNoiseTex, stp*randTCScl).a;
    v = texture3D(uniGridTex, stp).xyz;
    if(licProj2D == 1) {
        if(plane == 0) v.x = 0.0f;
        if(plane == 1) v.y = 0.0f;
        if(plane == 2) v.z = 0.0f;
    }

    for(i = 0; i < l; i++) {

        vec3 p;
        p.x = stp.x*textureSize(uniGridTex, 0).x;
        p.y = stp.y*textureSize(uniGridTex, 0).y;
        p.z = stp.z*textureSize(uniGridTex, 0).z;
        v = normalize(v);
        vec3 lambda;
        // Check which face of the cell is crossed by the vector
        if(dot(v, vec3(1.0, 0.0, 0.0)) < 0.0f) {
            lambda.x = (floor(p.x)-roundOff-p.x)/v.x; // left
        }
        else {
            lambda.x = (floor(p.x)+1.0+roundOff-p.x)/v.x; // right
        }
        if(dot(v, vec3(0.0, 1.0, 0.0)) < 0.0f) {
            lambda.y = (floor(p.y)-roundOff-p.y)/v.y; // bottom
        }
        else {
            lambda.y = (floor(p.y)+1.0+roundOff-p.y)/v.y; // top
        }
        if(dot(v, vec3(0.0, 0.0, 1.0)) < 0.0f) {
            lambda.z = (floor(p.z)-roundOff-p.z)/v.z; // back
        }
        else {
            lambda.z = (floor(p.z)+1.0+roundOff-p.z)/v.z; // front
        }

        scl = min(min(lambda.x, lambda.y), lambda.z);
        stp -= v*scl/textureSize(uniGridTex, 0);
        stp = clamp(stp, 0.0f, 1.0f);
        colLic += texture3D(randNoiseTex, stp*randTCScl).a;
        v = texture3D(uniGridTex, stp).xyz;
        v = normalize(v);
        if(licProj2D == 1) {
            if(plane == 0) v.x = 0.0f;
            if(plane == 1) v.y = 0.0f;
            if(plane == 2) v.z = 0.0f;
        }
    }

    stp = gl_TexCoord[0].stp;

    for(i = 0; i < l; i++) {

        vec3 p = stp*textureSize(uniGridTex, 0).xyz;
        v = normalize(v);
        vec3 lambda;
        // Check which face of the cell is crossed by the vector
        if(dot(v, vec3(1.0, 0.0, 0.0)) < 0.0f) {
            lambda.x = (floor(p.x)-roundOff-p.x)/v.x; // left
        }
        else {
            lambda.x = (floor(p.x)+1.0+roundOff-p.x)/v.x; // right
        }
        if(dot(v, vec3(0.0, 1.0, 0.0)) < 0.0f) {
            lambda.y = (floor(p.y)-roundOff-p.y)/v.y; // bottom
        }
        else {
            lambda.y = (floor(p.y)+1.0+roundOff-p.y)/v.y; // top
        }
        if(dot(v, vec3(0.0, 0.0, 1.0)) < 0.0f) {
            lambda.z = (floor(p.z)-roundOff-p.z)/v.z; // back
        }
        else {
            lambda.z = (floor(p.z)+1.0+roundOff-p.z)/v.z; // front
        }

        scl = min(min(lambda.x, lambda.y), lambda.z);


        stp += v*scl/textureSize(uniGridTex, 0);
        stp = clamp(stp, 0.0f, 1.0f);
        colLic += texture3D(randNoiseTex, stp*randTCScl).a;
        v = texture3D(uniGridTex, stp).xyz;
        if(licProj2D == 1) {
            if(plane == 0) v.x = 0.0f;
            if(plane == 1) v.y = 0.0f;
            if(plane == 2) v.z = 0.0f;
        }
    }

    colLic /= float(l+l+1);

    // Apply contrast stretching and to enhance the final image
    colLic -= licContrast;
    colLic /= (1.0 - 2.0*licContrast);

    colLic += licBrightness;

    return colLic;
    //return texture3D(randNoiseTex, gl_TexCoord[0].stp*randTCScl).a;
}


float CalcLIC(int l) {

    int i;
    vec3 v;
    vec3 stp = gl_TexCoord[0].stp;
    float vecScl = licDirScl;
    float randTCScl = licTCScl*float(textureSize(uniGridTex, 0).x)/float(textureSize(randNoiseTex, 0).x);

    float colLic = texture3D(randNoiseTex, stp*randTCScl).a;
    v = normalize(texture3D(uniGridTex, stp).xyz)*0.01;
    if(licProj2D == 1) {
        if(plane == 0) v.x = 0.0f;
        if(plane == 1) v.y = 0.0f;
        if(plane == 2) v.z = 0.0f;
    }


    for(i = 0; i < l; i++) {
        stp -= v*vecScl;
        stp = clamp(stp, 0.0f, 1.0f);
        colLic += texture3D(randNoiseTex, stp*randTCScl).a;
        v = normalize(texture3D(uniGridTex, stp).xyz)*0.01;
        if(licProj2D == 1) {
            if(plane == 0) v.x = 0.0f;
            if(plane == 1) v.y = 0.0f;
            if(plane == 2) v.z = 0.0f;
        }
    }

    stp = gl_TexCoord[0].stp;

    for(i = 0; i < l; i++) {
        stp += v*vecScl;
        stp = clamp(stp, 0.0f, 1.0f);
        colLic += texture3D(randNoiseTex, stp*randTCScl).a;
        v = normalize(texture3D(uniGridTex, stp).xyz)*0.01;
        if(licProj2D == 1) {
            if(plane == 0) v.x = 0.0f;
            if(plane == 1) v.y = 0.0f;
            if(plane == 2) v.z = 0.0f;
        }
    }

    colLic /= float(l+l+1);

    // Apply contrast stretching to enhance the final image
    colLic -= licContrast;
    colLic /= (1.0 - 2.0*licContrast);

    colLic += licBrightness;

    return colLic;
    //return texture3D(randNoiseTex, gl_TexCoord[0].stp*randTCScl).a;
}


/*
 * Calculate color gradient
 */
vec4 getColorGradient (float val, float thresh, float min, float max) {
    // blue - white - red
    vec4 colLow = vec4(0.23f, 0.29f, 0.75f, 1.0f);
    //vec4 colThresh = vec4(0.7f, 0.7f, 0.7f, 1.0f);
    vec4 colThresh = vec4(1.0f, 1.0f, 1.0f, 1.0f);
    vec4 colHigh = vec4(0.75f, 0.01f, 0.15f, 1.0f);

    // Calc dipole color
    vec4 colRes;
    if (val < thresh) {
        colRes = (val-min)/(thresh-min)*colThresh + (1.0f - (val-min)/(thresh-min))*colLow;
    }
    else {
        colRes = ((val-thresh)/(max-thresh))*colHigh + (1.0f - ((val-thresh)/(max-thresh)))*colThresh;
    }
    return vec4(colRes.rgb, 1.0f);
}


// Calculate CURL/RotF
// BUGGY
/*vec3 RotF() {
    return vec3(gradZ.y - gradY.z,
                gradX.z - gradZ.x,
                gradY.x - gradX.y);
}*/


/*vec3 gradX = texture3D(gradXTex, gl_TexCoord[0].stp).rgb;
vec3 gradY = texture3D(gradYTex, gl_TexCoord[0].stp).rgb;
vec3 gradZ = texture3D(gradZTex, gl_TexCoord[0].stp).rgb;
mat3 jacobian = mat3(gradX, gradY, gradZ);*/


/*float CalcJacobianNorm() {
    // Calculate trace
    float tr = jacobian[0][0] + jacobian[1][1] + jacobian[2][2];
    float m = tr/3.0;
    float q = tr/2.0;
    mat3 tmpmat = (jacobian - m*mat3(1.0));
    // Calc square of elements of tmpMat and divide by 6
    float p = (tmpmat[0][0]*tmpmat[0][0]+
               tmpmat[0][1]*tmpmat[0][1]+
               tmpmat[0][2]*tmpmat[0][2]+
               tmpmat[1][0]*tmpmat[1][0]+
               tmpmat[1][1]*tmpmat[1][1]+
               tmpmat[1][2]*tmpmat[1][2]+
               tmpmat[2][0]*tmpmat[2][0]+
               tmpmat[2][1]*tmpmat[2][1]+
               tmpmat[2][2]*tmpmat[2][2])/6.0;
    float phi = (1.0/3.0)*atan(sqrt(p*p*p-q*q)/q);
    // Calc eigenvalues
    float lambda0, lambda1, lambda2;
    lambda0 = m + 2.0*sqrt(p)*cos(phi);
    lambda1 = m - sqrt(p)*(cos(phi) + sqrt(3.0)*sin(phi));
    lambda2 = m - sqrt(p)*(cos(phi) - sqrt(3.0)*sin(phi));

    // Find highest eigenvalue;
    if(lambda0 > lambda1) {
        if(lambda0 > lambda2) {
            return lambda0;
        }
        else {
            return lambda2;
        }
    }
    else {
        if(lambda1 > lambda2) {
            return lambda1;
        }
        else {
            return lambda2;
        }
    }
}*/


/*
 * Main
 */
void main (void) {
    if(sliceRM == 0) { // Vec magnitude
        float mag = length(texture3D(uniGridTex, gl_TexCoord[0].stp).rgb);
//        if(mag*dataScl > colThresh)
//            gl_FragData[0] = vec4(vec3(0.0, mag*dataScl, 0.0), 1.0f);
//        else
//            gl_FragData[0] = vec4(vec3(mag*dataScl), 1.0f);
        gl_FragData[0] = vec4(vec3(mag*dataScl), 1.0f);
    }
    else if (sliceRM == 1) { // Vec direction
        gl_FragData[0] = vec4((normalize(texture3D(uniGridTex, gl_TexCoord[0].stp).rgb)+1.0f)*0.5f, 1.0f);
        //gl_FragData[0] = vec4((texture3D(uniGridTex, gl_TexCoord[0].stp).rgb*dataScl+1.0f)*0.5f, 1.0f);

        // Reduce saturation
        //gl_FragData[0] = vec4(
            //mix( (normalize(texture3D(uniGridTex, gl_TexCoord[0].stp).rgb)+1.0f)*0.5f, vec3( 0.5), dataScl)
            //, 1.0f) ;
    }
    else if (sliceRM == 2) { // LIC (GPU)
            gl_FragData[0] = vec4(vec3(CalcLIC(licLen)), 1.0f);
    }
    else if (sliceRM == 3) { // Curl magnitude
        float mag = texture3D(curlMagTex, gl_TexCoord[0].stp).a;
//      if(mag*dataScl < colThresh)
//            gl_FragData[0] = vec4(vec3(0.0, mag*dataScl, 0.0), 1.0f);
//        else
//            gl_FragData[0] = vec4(vec3(mag*dataScl), 1.0f);
        gl_FragData[0] = vec4(vec3(mag*dataScl), 1.0f);
        //gl_FragData[0] = vec4(vec3(pow(mag*curlScl, 10.0)), 1.0f);
    }
    else if (sliceRM == 4) {  // Density map
        gl_FragData[0] = vec4(vec3(texture3D(densityMapTex, gl_TexCoord[0].stp).a), 1.0f);
    }

    // TiDispl avg25, offs 25               : (maxX 0.152978, maxY 0.155159, maxZ 0.160629)
    // Dipole (CELL) avg25, offs 25         : (maxX 0.034073, maxY 0.033833, maxZ 0.033232)
    // Dipole (Ba-Center/Ti) avg25, offs 25 : (maxX 0.048048, maxY 0.048744, maxZ 0.047199)
    // Dipole (O-Center/Ti) avg25, offs 25  : (maxX 0.064014, maxY 0.064880, maxZ 0.065018)

    else if (sliceRM == 5) {  // Delta x
        gl_FragData[0] = getColorGradient(normalize(texture3D(uniGridTex,
            gl_TexCoord[0].stp)).r, 0.0, -0.152978, 0.152978);
    }
    else if (sliceRM == 6) {  // Delta y
        gl_FragData[0] = getColorGradient(normalize(texture3D(uniGridTex,
            gl_TexCoord[0].stp)).g, 0.0, -0.155159, 0.155159);
    }
    else if (sliceRM == 7) {  // Delta z
        gl_FragData[0] = getColorGradient(normalize(texture3D(uniGridTex,
            gl_TexCoord[0].stp)).b, 0.0, -0.160629, 0.160629);
    }
    else if (sliceRM == 8) {  // Color texture
        vec4 color = vec4(texture3D(colorTex, gl_TexCoord[0].stp).rgb, 1.0f);
        //color.xyz = normalize(color.xyz);

        color.xyz *=dataScl;
        color.xyz += 1.0;
        color.xyz *= 0.5;
        gl_FragData[0] = color;
    }
    else { // Invalid render mode
        gl_FragData[0] = vec4(0.0f, 1.0, 1.0f, 1.0f); // Turquoise
    }
}
