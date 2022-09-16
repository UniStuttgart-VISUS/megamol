#extension GL_EXT_gpu_shader4 : enable

uniform sampler3D densityTex;
uniform sampler3D uniGridTex;
uniform sampler3D randNoiseTex;
uniform sampler3D curlMagTex;
uniform sampler3D colorTex;

uniform sampler2D srcColorBuff;
uniform sampler2D srcDepthBuff;
uniform sampler2D tcBuff;
uniform sampler2D posESBuff;
uniform sampler2D posWinSBuff;

uniform float delta;
uniform float isoVal;
uniform float alphaScl;
uniform int maxStep;

uniform float licDirScl;
uniform int licLen;
uniform float licTCScl;

uniform vec4 viewportDim; //(width, height, near far)

uniform int vColorMode;
uniform int rayMarchTex;

varying vec4 pos;
varying vec3 posES;


uniform float licContrast;
uniform float licBright;

/*vec3 gradX = texture3D(gradXTex, gl_TexCoord[0].stp).rgb;
vec3 gradY = texture3D(gradYTex, gl_TexCoord[0].stp).rgb;
vec3 gradZ = texture3D(gradZTex, gl_TexCoord[0].stp).rgb;
mat3 jacobianCenter = mat3(gradX, gradY, gradZ);*/


float CalcJacobianNorm(mat3 jacobian) {
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
}


/*float CalcLIC(int l) {

    int i;
    vec3 v;
    vec3 stp = gl_TexCoord[0].stp;
//    float vecScl = licDirScl;
    vec4 vecScl;
    //float randTCScl = 1.0*float(textureSize(uniGridTex, 0).x)/float(textureSize(randNoiseTex, 0).x);
    float randTCScl = 1.0;
    float scl;

    float colLic = texture3D(randNoiseTex, stp*randTCScl).a;
    v = texture3D(uniGridTex, stp).xyz;

    for(i = 0; i < l; i++) {

        vec3 p = stp*textureSize(uniGridTex, 0).xyz;
        v = normalize(v);
        vec3 lambda;
        // Check which face of the cell is crossed by the vector
        if(dot(v, vec3(1.0, 0.0, 0.0)) < 0.0f) {
            lambda.x = (floor(p.x)-0.01-p.x)/v.x; // left
        }
        else {
            lambda.x = (floor(p.x)+1.01-p.x)/v.x; // right
        }
        if(dot(v, vec3(0.0, 1.0, 0.0)) < 0.0f) {
            lambda.y = (floor(p.y)-0.01-p.y)/v.y; // bottom
        }
        else {
            lambda.y = (floor(p.y)+1.01-p.y)/v.y; // top
        }
        if(dot(v, vec3(0.0, 0.0, 1.0)) < 0.0f) {
            lambda.z = (floor(p.z)-0.01-p.z)/v.z; // back
        }
        else {
            lambda.z = (floor(p.z)+1.01-p.z)/v.z; // front
        }

        scl = min(min(lambda.x, lambda.y), lambda.z);


        stp -= v*scl/textureSize(uniGridTex, 0);
        stp = clamp(stp, 0.0f, 1.0f);
        colLic += texture3D(randNoiseTex, stp*randTCScl).a;
        v = texture3D(uniGridTex, stp).xyz;
    }

    stp = gl_TexCoord[0].stp;

    for(i = 0; i < l; i++) {

        vec3 p = stp*textureSize(uniGridTex, 0).xyz;
        v = normalize(v);
        vec3 lambda;
        // Check which face of the cell is crossed by the vector
        if(dot(v, vec3(1.0, 0.0, 0.0)) < 0.0f) {
            lambda.x = (floor(p.x)-0.01-p.x)/v.x; // left
        }
        else {
            lambda.x = (floor(p.x)+1.01-p.x)/v.x; // right
        }
        if(dot(v, vec3(0.0, 1.0, 0.0)) < 0.0f) {
            lambda.y = (floor(p.y)-0.01-p.y)/v.y; // bottom
        }
        else {
            lambda.y = (floor(p.y)+1.01-p.y)/v.y; // top
        }
        if(dot(v, vec3(0.0, 0.0, 1.0)) < 0.0f) {
            lambda.z = (floor(p.z)-0.01-p.z)/v.z; // back
        }
        else {
            lambda.z = (floor(p.z)+1.01-p.z)/v.z; // front
        }

        scl = min(min(lambda.x, lambda.y), lambda.z);


        stp += v*scl/textureSize(uniGridTex, 0);
        stp = clamp(stp, 0.0f, 1.0f);
        colLic += texture3D(randNoiseTex, stp*randTCScl).a;
        v = texture3D(uniGridTex, stp).xyz;
    }

    colLic /= float(l+l+1);

    // Apply contrast stretching and to enhance the final image
    colLic -= licContrast;
    colLic /= (1.0 - 2.0*licContrast);
    colLic += licScl;

    return colLic;
    //return texture3D(randNoiseTex, gl_TexCoord[0].stp*randTCScl).a;
}*/


// Calculate line integral convolution (LIC)
float CalcLIC(int l, vec3 stp_start) {

    int i;
    vec3 v;
    vec3 stp = stp_start;
    float vecScl = licDirScl*0.01;
    float randTCScl = licTCScl*float(textureSize(uniGridTex, 0).x)/float(textureSize(randNoiseTex, 0).x);

    float colLic = texture3D(randNoiseTex, stp*randTCScl).a;
    v = texture3D(uniGridTex, stp).xyz;

    for(i = 0; i < l; i++) {
        stp -= v*vecScl;
        stp = clamp(stp, 0.0f, 1.0f);
        colLic += texture3D(randNoiseTex, stp*randTCScl).a;
        v = texture3D(uniGridTex, stp).xyz;
    }

    stp = stp_start;

    for(i = 0; i < l; i++) {
        stp += v*vecScl;
        stp = clamp(stp, 0.0f, 1.0f);
        colLic += texture3D(randNoiseTex, stp*randTCScl).a;
        v = texture3D(uniGridTex, stp).xyz;
    }

    colLic /= float(l+l+1);

    // Apply contrast stretching and to enhance the final image
    colLic -= licContrast;
    colLic /= (1.0 - 2.0*licContrast);
    colLic += licBright;

    return colLic;
}


// Calculate line integral convolution (LIC)
float CalcLICProj(int l, vec3 stp, vec3 normal) {

    int i;
    vec3 v;
    //vec3 stp = gl_TexCoord[0].stp;
    float vecScl = licDirScl;
    float randTCScl = licTCScl*float(textureSize(uniGridTex, 0).x)/float(textureSize(randNoiseTex, 0).x);

    float colLic = texture3D(randNoiseTex, stp*randTCScl).a;
        if(rayMarchTex == 0) { // Density map
            v = normalize(texture3D(colorTex, stp).xyz);
        }
        else if(rayMarchTex == 1) {// Use vector magitude
            v = normalize(texture3D(uniGridTex, stp).xyz);
        }
        else if(rayMarchTex == 2) {// Use curl magitude
            v = normalize(texture3D(uniGridTex, stp).xyz);
        }
    vec3 gradient = normal;
    vec3 c = cross(gradient, v);
    vec3 d = cross(gradient, c)*0.01;
    float gradOffset = 0.01;
    vec3 pos = stp;

    for(i = 0; i < l; i++) {
        pos -= d*vecScl;
        pos = clamp(pos, 0.0f, 1.0f);
        colLic += texture3D(randNoiseTex, pos*randTCScl).a;

        if(rayMarchTex == 0) { // Density map
            v = normalize(texture3D(colorTex, pos).xyz);
        }
        else if(rayMarchTex == 1) {// Use vector magitude
            v = normalize(texture3D(uniGridTex, pos).xyz);
        }
        else if(rayMarchTex == 2) {// Use curl magitude
            v = normalize(texture3D(uniGridTex, pos).xyz);
        }

            // Compute gradient by central differences
         /*   if(rayMarchTex == 0) { // Density map
                gradient.x = texture3D(densityTex, pos + vec3(gradOffset, 0, 0)).a
                           - texture3D(densityTex, pos + vec3(-gradOffset, 0, 0)).a;
                gradient.y = texture3D(densityTex, pos + vec3(0, gradOffset, 0)).a
                           - texture3D(densityTex, pos + vec3(0, -gradOffset, 0)).a;
                gradient.z = texture3D(densityTex, pos + vec3(0, 0, gradOffset)).a
                           - texture3D(densityTex, pos + vec3(0, 0, -gradOffset)).a;
            }
            else if(rayMarchTex == 1) {// Use vector magitude
                gradient.x = length(texture3D(uniGridTex, pos + vec3(gradOffset, 0, 0)).rgb)
                           - length(texture3D(uniGridTex, pos + vec3(-gradOffset, 0, 0)).rgb);
                gradient.y = length(texture3D(uniGridTex, pos + vec3(0, gradOffset, 0)).rgb)
                           - length(texture3D(uniGridTex, pos + vec3(0, -gradOffset, 0)).rgb);
                gradient.z = length(texture3D(uniGridTex, pos + vec3(0, 0, gradOffset)).rgb)
                           - length(texture3D(uniGridTex,pos  + vec3(0, 0, -gradOffset)).rgb);
            }
            else if(rayMarchTex == 2) {// Use curl magitude
                gradient.x = texture3D(curlMagTex, pos+ vec3(gradOffset, 0, 0)).a
                           - texture3D(curlMagTex, pos + vec3(-gradOffset, 0, 0)).a;
                gradient.y = texture3D(curlMagTex, pos + vec3(0, gradOffset, 0)).a
                           - texture3D(curlMagTex, pos + vec3(0, -gradOffset, 0)).a;
                gradient.z = texture3D(curlMagTex, pos + vec3(0, 0, gradOffset)).a
                           - texture3D(curlMagTex, pos + vec3(0, 0, -gradOffset)).a;
                // Let gradient point in the opposite direction
                gradient.x *= -1.0f;
                gradient.y *= -1.0f;
                gradient.z *= -1.0f;
            }
        gradient = normalize(gradient);

        vec3 c = cross(gradient, v);
        vec3 d = cross(gradient, c)*0.01;*/
    }

    pos = stp;

    for(i = 0; i < l; i++) {
        pos += d*vecScl;
        pos = clamp(pos, 0.0f, 1.0f);
        colLic += texture3D(randNoiseTex, pos*randTCScl).a;

        // resample vector
        if(rayMarchTex == 0) { // Density map
            v = normalize(texture3D(colorTex, pos).xyz);
        }
        else if(rayMarchTex == 1) {// Use vector magitude
            v = normalize(texture3D(uniGridTex, pos).xyz);
        }
        else if(rayMarchTex == 2) {// Use curl magitude
            v = normalize(texture3D(uniGridTex, pos).xyz);
        }

            // Compute gradient by central differences
            /*if(rayMarchTex == 0) { // Density map
                gradient.x = texture3D(densityTex, pos + vec3(gradOffset, 0, 0)).a
                           - texture3D(densityTex, pos + vec3(-gradOffset, 0, 0)).a;
                gradient.y = texture3D(densityTex, pos + vec3(0, gradOffset, 0)).a
                           - texture3D(densityTex, pos + vec3(0, -gradOffset, 0)).a;
                gradient.z = texture3D(densityTex, pos + vec3(0, 0, gradOffset)).a
                           - texture3D(densityTex, pos + vec3(0, 0, -gradOffset)).a;
            }
            else if(rayMarchTex == 1) {// Use vector magitude
                gradient.x = length(texture3D(uniGridTex, pos + vec3(gradOffset, 0, 0)).rgb)
                           - length(texture3D(uniGridTex, pos + vec3(-gradOffset, 0, 0)).rgb);
                gradient.y = length(texture3D(uniGridTex, pos + vec3(0, gradOffset, 0)).rgb)
                           - length(texture3D(uniGridTex, pos + vec3(0, -gradOffset, 0)).rgb);
                gradient.z = length(texture3D(uniGridTex, pos + vec3(0, 0, gradOffset)).rgb)
                           - length(texture3D(uniGridTex,pos  + vec3(0, 0, -gradOffset)).rgb);
            }
            else if(rayMarchTex == 2) {// Use curl magitude
                gradient.x = texture3D(curlMagTex, pos+ vec3(gradOffset, 0, 0)).a
                           - texture3D(curlMagTex, pos + vec3(-gradOffset, 0, 0)).a;
                gradient.y = texture3D(curlMagTex, pos + vec3(0, gradOffset, 0)).a
                           - texture3D(curlMagTex, pos + vec3(0, -gradOffset, 0)).a;
                gradient.z = texture3D(curlMagTex, pos + vec3(0, 0, gradOffset)).a
                           - texture3D(curlMagTex, pos + vec3(0, 0, -gradOffset)).a;
                // Let gradient point in the opposite direction
                gradient.x *= -1.0f;
                gradient.y *= -1.0f;
                gradient.z *= -1.0f;
            }
        gradient = normalize(gradient);

        vec3 c = cross(gradient, v);
        vec3 d = cross(gradient, c)*0.01;*/
    }

    colLic /= float(l+l+1);

    // Apply contrast stretching and to enhance the final image
    colLic -= licContrast;
    colLic /= (1.0 - 2.0*licContrast);
    colLic += licBright;

    return colLic;
}


/*
 * Calculate color gradient out of two colors
 */
vec4 getColorGrad(float val) {
    // blue - white - red
    vec4 colLow = vec4(0.23f, 0.29f, 0.75f, 1.0f);
    vec4 colHigh = vec4(0.75f, 0.01f, 0.15f, 1.0f);
    // Calc color
    vec4 colRes;
    colRes = val*colHigh + (1.0f - val)*colLow;
    return vec4(colRes.rgb, 1.0f);
}


void main (void) {

    vec2 texc = ((pos.xy/pos.w) + 1.0f)/2.0f; // Get normalized device coords

    vec3 viewES =  normalize(texture2D(posESBuff, texc).rgb - posES); // For local lighting

    vec3 rayEnd = texture2D(tcBuff, texc).rgb;
    vec3 rayStart = gl_TexCoord[0].stp;
    vec3 ray = rayEnd - rayStart;

    //vec3 rayNDCPosEnd = texture2D(posWinSBuff, texc).rgb;
    //vec3 rayNDCPosEnd = texture2D(posWinSBuff, gl_FragCoord.xy/viewportDim.xy).rgb;


//    vec3 rayNDCPosEnd = texelFetch2D(posWinSBuff, ivec2(gl_FragCoord.xy), 0).rgb;
    vec3 rayESPosEnd = texelFetch2D(posESBuff, ivec2(gl_FragCoord.xy), 0).rgb;

//    vec3 rayNDCPosStart = gl_FragCoord.xyz;
    vec3 rayESPosStart = posES;

    vec3 rayESPos = rayESPosEnd - rayESPosStart;

    float rayLen = length(ray);
    vec3 ray_norm = normalize(ray);
    vec3 ray_delta = ray_norm*delta;

//    float rayNDCLen = length(rayNDCPos);
    float rayESLen = length(rayESPos);
//    float ndcPosDelta = delta * (rayNDCLen / rayLen);
    float esPosDelta = delta * (rayESLen / rayLen);
//    vec3 rayNDCPos_norm = normalize(rayNDCPos);
    vec3 rayESPos_norm = normalize(rayESPos);
//    vec3 rayNDCPosDelta = rayNDCPos_norm*ndcPosDelta;
    vec3 rayESPosDelta = rayESPos_norm*esPosDelta;

    float ray_delta_len = length(ray_delta);

    vec3 vec = rayStart;
//    vec3 vecNDC = rayNDCPosStart;
    vec3 vecES = rayESPosStart;

    vec4 col_acc = vec4(0.0f, 0.0f, 0.0f, 0.0f);
    float alpha_acc = 0.0f;
    float length_acc = 0.0f;

    float tex_sample;
    vec4 color_sample, color_sample_old;
    float alpha_sample;
    vec3 gradient;

    float depthValComp = texelFetch2D(srcDepthBuff, ivec2(gl_FragCoord.xy), 0).r;

    //if(vecNDC.z >= depthValComp) discard; // 'Depth test'

    float isoDiff = 0;
    float isoDiffOld = 0;
    float ssDepth = 0.0;


//    viewportDim

    //while(!(length_acc > rayLen || alpha_acc >= 1.0 || vecNDC.z >= depthValComp)) {
    int MAXSTEP = 1000;
    int stepCnt = 0;
    while(!(length_acc > rayLen || alpha_acc >= 1.0 || ssDepth >= depthValComp || stepCnt > MAXSTEP)) {

        // Get sample
        if (rayMarchTex == 0) { // Use density map
            tex_sample = texture3D(densityTex, vec).a;
        }
        else if (rayMarchTex == 1) { // Use vector magnitude
            tex_sample = length(texture3D(uniGridTex, vec).rgb);
        }
        else if (rayMarchTex == 2) { // Use curl magitude
            tex_sample = texture3D(curlMagTex, vec).a;
        }

        isoDiff = tex_sample - isoVal;

        if(isoDiffOld*isoDiff < 0) { // We crossed the iso surface
            alpha_sample = alphaScl;

            float gradOffset = 0.01;
            vec3 vecSclOffs = ray_delta*(abs(isoDiff)/(abs(isoDiff)+abs(isoDiffOld)));
//            vec3 vecSclOffsNDS = rayNDCPosDelta*(abs(isoDiffOld)/(abs(isoDiff)+abs(isoDiffOld)));
            vec3 vecSclOffsES = rayESPosDelta*(abs(isoDiffOld)/(abs(isoDiff)+abs(isoDiffOld)));

            // Compute gradient by central differences
            if(rayMarchTex == 0) { // Density map
                gradient.x = texture3D(densityTex, (vec-vecSclOffs) + vec3(gradOffset, 0, 0)).a
                           - texture3D(densityTex, (vec-vecSclOffs) + vec3(-gradOffset, 0, 0)).a;
                gradient.y = texture3D(densityTex, (vec-vecSclOffs) + vec3(0, gradOffset, 0)).a
                           - texture3D(densityTex, (vec-vecSclOffs) + vec3(0, -gradOffset, 0)).a;
                gradient.z = texture3D(densityTex, (vec-vecSclOffs) + vec3(0, 0, gradOffset)).a
                           - texture3D(densityTex, (vec-vecSclOffs) + vec3(0, 0, -gradOffset)).a;
            }
            else if(rayMarchTex == 1) {// Use vector magitude
                gradient.x = length(texture3D(uniGridTex, (vec-vecSclOffs) + vec3(gradOffset, 0, 0)).rgb)
                           - length(texture3D(uniGridTex, (vec-vecSclOffs) + vec3(-gradOffset, 0, 0)).rgb);
                gradient.y = length(texture3D(uniGridTex, (vec-vecSclOffs) + vec3(0, gradOffset, 0)).rgb)
                           - length(texture3D(uniGridTex, (vec-vecSclOffs) + vec3(0, -gradOffset, 0)).rgb);
                gradient.z = length(texture3D(uniGridTex, (vec-vecSclOffs) + vec3(0, 0, gradOffset)).rgb)
                           - length(texture3D(uniGridTex, (vec-vecSclOffs) + vec3(0, 0, -gradOffset)).rgb);
            }
            else if(rayMarchTex == 2) {// Use curl magitude
                gradient.x = texture3D(curlMagTex, (vec-vecSclOffs) + vec3(gradOffset, 0, 0)).a
                           - texture3D(curlMagTex, (vec-vecSclOffs) + vec3(-gradOffset, 0, 0)).a;
                gradient.y = texture3D(curlMagTex, (vec-vecSclOffs) + vec3(0, gradOffset, 0)).a
                           - texture3D(curlMagTex, (vec-vecSclOffs) + vec3(0, -gradOffset, 0)).a;
                gradient.z = texture3D(curlMagTex, (vec-vecSclOffs) + vec3(0, 0, gradOffset)).a
                           - texture3D(curlMagTex, (vec-vecSclOffs) + vec3(0, 0, -gradOffset)).a;
                // Let gradient point in the opposite direction
                gradient.x *= -1.0f;
                gradient.y *= -1.0f;
                gradient.z *= -1.0f;
            }

            // Setup color
            vec4 color;
            if(vColorMode == 0) { // Uniform color
                color = vec4(0.0, 0.0, 1.0, 1.0);
            }
            else if(vColorMode == 1) { // Vector dir
                if(rayMarchTex == 0) {
                    color = vec4(texture3D(colorTex, vec-vecSclOffs).rgb, 1.0);
                    color.xyz = normalize(color.xyz);
                    color.xyz += 1.0;
                    color.xyz *= 0.5;
                }
                else {
                    color = vec4(((normalize(texture3D(uniGridTex, vec-vecSclOffs).rgb)+1.0f)*0.5), 1.0f);
                }
            }
            else if(vColorMode == 2) { // Vector mag
                if(rayMarchTex == 0) {
                    //color = vec4(normalize(texture3D(colorTex, vec-vecSclOffs).rgb), 1.0);
                    color = vec4(texture3D(colorTex, vec-vecSclOffs).rgb, 1.0);
                }
                else {
                    color = getColorGrad(length(texture3D(uniGridTex, vec-vecSclOffs).rgb));
                }
            }
            else if(vColorMode == 3) { // LIC
                //color = vec4(vec3(CalcLIC(licLen, vec-vecSclOffs)), 1.0);
                color = vec4(vec3(CalcLICProj(licLen, vec-vecSclOffs, normalize(gradient))), 1.0);
            }
            else { // Invalid coloring mode
                color = vec4(0.0, 1.0, 1.0, 1.0);
            }

            // Gradient is now in uvw-space -> transform it to world space
            // Note: This works only for texture coordinates, where the z axis is (0, 0, -1) in WS
            gradient.xyz *= -1.0;

            // ... and then eye space
            vec3 gradES = gl_NormalMatrix*gradient;

            // Calculate local lighting
            vec3 L = normalize(vec3(gl_LightSource[0].position.xyz));
            vec3 V = normalize(vec3(viewES.xyz));

            // Don't use lighting
            //color_sample = color;

            // Use lighting
            color_sample = vec4(LocalLighting(V, normalize(gradES), L, color.rgb), 1.0);
            //color_sample = color_sample*0.75 + color*0.25;

            // draw interior darker
            if(isoDiffOld > 0.0 )
                color_sample.xyz *= 0.7;

            //vecNDC += vecSclOffsNDS; // for z test
            vecES += vecSclOffsES; // for z test
        }
        else { // We crossed empty space
            alpha_sample = 0.0f;
            color_sample = vec4(0.0f, 0.0f, 0.0f, 0.0f);
//            vecNDC += rayNDCPosDelta; // for z test
            vecES += rayESPosDelta; // for z test
        }

        // Accumulate color and alpha sample
        col_acc   += ((1.0f - alpha_acc)*color_sample*alpha_sample);
        alpha_acc += ((1.0f - alpha_acc)*alpha_sample);

        vec += ray_delta;
        length_acc += ray_delta_len;

        isoDiffOld = isoDiff;

        float n = viewportDim.z;
        float f = viewportDim.w;

        ssDepth = (viewportDim.z*viewportDim.w)/vecES.z + viewportDim.z + viewportDim.w;
        // Map near ... far to 0 ... 1
        ssDepth -= n;
        ssDepth /= (f-n);


        stepCnt++;
    }

    gl_FragColor = vec4(col_acc.rgb/alpha_acc, alpha_acc);
}
