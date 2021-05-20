vec3 hsvSpiralColor(int color_idx, int total_colors){

    float orbit_cnt = 1.0 + log(total_colors);

    float idx_normalized = float(color_idx)/float(total_colors);

    float h = mod(idx_normalized * orbit_cnt, 1.0);
    float s = 0.1 + 0.9 *idx_normalized;
    float v = 1.0 - 0.9 * idx_normalized;

    vec3 c = vec3(h,s,v);

    // http://lolengine.net/blog/2013/07/27/rgb-to-hsv-in-glsl
    // All components are in the range [0…1], including hue.
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
    //return c.yyy;
}
