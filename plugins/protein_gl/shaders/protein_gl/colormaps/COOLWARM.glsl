vec3 CoolWarmMsh(float val, float min, float thresh, float max) {

    val = clamp(val, min, max);

    vec3 redMsh = vec3(90, 1.08, 0.5);
    vec3 whiteMsh = vec3(90, 0, 1.061);
    vec3 blueMsh = vec3(90, 1.08, -1.1);

    vec3 resMsh;

    if (val < thresh) {
        //resMsh = redMsh;
        resMsh =  (val-min)/(thresh-min)*whiteMsh +
            (1.0f - (val-min)/(thresh-min))*redMsh;
    }
    else {
        //resMsh = blueMsh;
        whiteMsh.z *= -1.0;
        return ((val-thresh)/(max-thresh))*blueMsh +
            (1.0f - ((val-thresh)/(max-thresh)))*whiteMsh;
    }
    return resMsh;
}
