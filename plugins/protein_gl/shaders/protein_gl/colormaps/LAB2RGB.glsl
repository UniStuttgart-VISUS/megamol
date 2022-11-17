// Converts a color from CIELAB RGB
//
// RGB values can be between 0.0 and 1.0
// Values for L are in the range [0,100] while a and b are roughly in the range
// [-110,110].
//
// This transform is based on ITU-R Recommendation BT.709 using the D65
// white point reference. The error in transforming RGB -> Lab -> RGB is
// approximately 10^-5.
//
// Based on MATLAB code by Mark Ruzon
// see http://robotics.stanford.edu/~ruzon/software/rgblab.html
vec3 LAB2RGB(float L, float a, float b) {

    /* 1. LAB to XYZ */

    vec3 xyz;

    // Define Thresholds
    float T1 = 0.008856;
    float T2 = 0.206893;

    // Compute Y
    float fY = pow(((L+16.0)/116.0), 3.0);
    bool yt = (fY > T1);
    fY = int(!yt)*(L/903.3) + int(yt)*fY;
    xyz.y = fY;

    // Alter fY slightly for further calculations
    fY = int(yt)*(pow(fY, 1.0/3.0)) + int(!yt)*(7.787*fY+0.137931034);

    // Compute X
    float fX = a / 500.0 + fY;
    bool xt = fX > T2;
    xyz.x = int(xt)*pow(fX, 3.0) + int(!xt)*((fX - 0.137931034) / 7.787);

    // Compute Z
    float fZ = fY - b / 200.0;
    bool zt = fZ > T2;
    xyz.z = int(zt)*pow(fZ, 3.0) + int(!zt)*((fZ - 0.137931034) / 7.787);

    // Normalize for D65 white point
    xyz.x = xyz.x * 0.950456;
    xyz.z = xyz.z * 1.088754;

    /* 2. XYZ to RGB */

    vec3 rgb;

    mat3 m = mat3( 3.240479, -0.969256,  0.055648,
                  -1.537150,  1.875992, -0.204043,
                  -0.498535,  0.041556,  1.057311);
    rgb = m*xyz;
    rgb.x = clamp(rgb.x, 0.0, 1.0);
    rgb.y = clamp(rgb.y, 0.0, 1.0);
    rgb.z = clamp(rgb.z, 0.0, 1.0);

    return rgb;
}
