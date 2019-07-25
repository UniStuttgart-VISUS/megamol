uniform sampler2D densityTexture;

in vec2 vsUV;

layout(location = 0) out vec4 fsColor;

vec4 blur5(  vec2 uv,   vec2 direction) {
    vec2 resolution = textureSize(densityTexture, 0);
  vec4 color = vec4(0.0);
  vec2 off1 = vec2(1.3333333333333333) * direction;
  color += texture2D(densityTexture, uv) * 0.29411764705882354;
  color += texture2D(densityTexture, uv + (off1 / resolution)) * 0.35294117647058826;
  color += texture2D(densityTexture, uv - (off1 / resolution)) * 0.35294117647058826;
  return color; 
}

float blur(mat3 a) {
    // Gaussian 3x3 kernel.
    mat3 s = mat3(1.0/16.0, 1.0/8.0, 1.0/16.0, 
    1.0/8.0, 1.0/4.0, 1.0/8.0, 
    1.0/16.0, 1.0/8.0, 1.0/16.0);

    return s[0][0] * a[0][0] +
    s[0][1] * a[0][1] +
    s[0][2] * a[0][2] +
    s[1][0] * a[1][0] +
    s[1][1] * a[1][1] +
    s[1][2] * a[1][2] +
    s[2][0] * a[2][0] +
    s[2][1] * a[2][1] +
    s[2][2] * a[2][2] ;
}

mat3 density3x3(vec2 uv) {
    float dx = 1.0 / textureSize(densityTexture, 0).x;
    float dy = 1.0 / textureSize(densityTexture, 0).y;

    return mat3(
        texture(densityTexture,uv + vec2(-dx, dy)).r,
        texture(densityTexture,uv + vec2(-dx, 0.0)).r,
        texture(densityTexture,uv + vec2(-dx, -dy)).r,
        texture(densityTexture,uv + vec2(0.0, dy)).r,
        texture(densityTexture,uv).r,
        texture(densityTexture,uv + vec2(0.0, -dy)).r,
        texture(densityTexture,uv + vec2(dx, dy)).r,
        texture(densityTexture,uv + vec2(dx, 0.0)).r,
        texture(densityTexture,uv + vec2(dx, -dy)).r
     );
}

float density(vec2 uv) {
    //return texture(densityTexture, uv).r;
    return blur(density3x3(uv));
}

float densityGradient(vec2 uv) {
    float dx = 1.0 / textureSize(densityTexture, 0).x;
    float dy = 1.0 / textureSize(densityTexture, 0).y;

    float n[9];
    n[0] = density(uv + vec2(-dx, -dy));
    n[1] = density(uv + vec2(0.0, -dy));
    n[2] = density(uv + vec2(dx, -dy));
    n[3] = density(uv + vec2(-dx, 0.0));
    n[4] = density(uv);
    n[5] = density(uv + vec2(dx, 0.0));
    n[6] = density(uv + vec2(-dx, dy));
    n[7] = density(uv + vec2(0.0, dy));
    n[8] = density(uv + vec2(dx, dy));

    // Central differences
    float gx = (n[5] - n[3]) * 0.5;
    float gy = (n[7] - n[1]) * 0.5;

    // Sobel
    //float gx = n[2] + (2.0*n[5]) + n[8] - (n[0] + (2.0*n[3]) + n[6]);
    //float gy = n[0] + (2.0*n[1]) + n[2] - (n[6] + (2.0*n[7]) + n[8]);
    //gx /= 8.0;
    //gy /= 8.0;

    // Prewitt
    //float gx = n[2] + n[5] + n[8] - (n[0] + n[3] + n[6]);
    //float gy = n[0] + n[1] + n[2] - (n[6] + n[7] + n[8]);
    //gx /= 6.0;
    //gy /= 6.0;

    float magnitude = sqrt(gx * gx + gy * gy);

    return magnitude;
}

float contourLine() {
    const float magicIsoValue = 0.7; //TODO: array uniform
    const float contourSize = 0.5; // TODO: contour size uniform

    //TODO: smoothing kernel size >= contour size. (i.e. adjust kernel size depending on contour size)

    float delta = abs(density(vsUV) - magicIsoValue);
    float gradient = densityGradient(vsUV);
    float distanceToContour = delta / gradient;
  
    float contourness =  1.0 - smoothstep(contourSize - 0.5, contourSize + 0.5, distanceToContour);

    //return vec3(contourness, 0.0, gradient * 20.0);
     return contourness;
}

void main() {
    const vec4 contourColor = vec4(1.0, 0.0, 0.0, 1.0); //TODO: uniform

    vec4 densityAlpha = texture(densityTexture, vsUV);
    vec4 valueColor = vec4(vec3(densityAlpha.r), densityAlpha.a);
    float contour = contourLine();

    fsColor = mix(valueColor, contourColor, contour);
}
