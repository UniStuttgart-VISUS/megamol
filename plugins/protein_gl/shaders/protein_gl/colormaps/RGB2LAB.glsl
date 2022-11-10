// Converts a color from RGB to CIELAB
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
vec3 RGB2LAB(float r, float g, float b) {

    /* 1. RGB to XYZ */

    mat3 m = mat3(0.41242400, 0.21265600, 0.01933240,
                  0.35757900, 0.71515800, 0.11919300,
                  0.18046400, 0.07218560, 0.95044400);
    vec3 xyz = m * vec3(r, g, b);

    /* 2. XYZ to LAB */

    // Normalize for D65 white point
    xyz.x /= 0.950456;
    xyz.z /= 1.088754;

    // Set a threshold
    float T = 0.008856;
    bool xt = xyz.x > T;
    bool yt = xyz.y > T;
    bool zt = xyz.z > T;

    float y3 = pow(xyz.y, 1.0/3.0); // Precompute because it is used twice

    float fX = int(xt)*pow(xyz.x, 1.0/3.0) + int(!xt)*(7.787037037*xyz.x+0.137931034);
    float fY = int(yt)*y3                  + int(!yt)*(7.787037037*xyz.y+0.137931034);
    float fZ = int(zt)*pow(xyz.z, 1.0/3.0) + int(!zt)*(7.787037037*xyz.z+0.137931034);

    vec3 Lab;

    Lab.x = int(yt)*(116*y3-16.0) + int(!yt)*(903.3*xyz.y);
    Lab.y = 500*(fX-fY);
    Lab.z = 200*(fY-fZ);

    return Lab;
}
