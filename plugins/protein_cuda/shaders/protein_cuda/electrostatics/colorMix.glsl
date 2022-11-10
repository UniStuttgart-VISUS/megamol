/* Mix three colors (RGBA) */
vec4 MixColors (float val,
                float min,
                float thresh,
                float max,
                vec4 colorMin,
                vec4 colorThresh,
                vec4 colorMax) {
    val = clamp(val, min, max);
    if (val < thresh) {
        return (val-min)/(thresh-min)*colorThresh +
            (1.0f - (val-min)/(thresh-min))*colorMin;
    }
    else {
        return ((val-thresh)/(max-thresh))*colorMax +
            (1.0f - ((val-thresh)/(max-thresh)))*colorThresh;
    }
}

/* Mix three colors (RGB) */
vec3 MixColors (float val,
                float min,
                float thresh,
                float max,
                vec3 colorMin,
                vec3 colorThresh,
                vec3 colorMax) {
    val = clamp(val, min, max);
    if (val < thresh) {
        return (val-min)/(thresh-min)*colorThresh +
            (1.0f - (val-min)/(thresh-min))*colorMin;
    }
    else {
        return ((val-thresh)/(max-thresh))*colorMax +
            (1.0f - ((val-thresh)/(max-thresh)))*colorThresh;
    }
}

/* Mix two colors (RGBA) */
vec4 MixColors(float val,
               float min,
               float max,
               vec4 colorMin,
               vec4 colorMax) {
    val = clamp(val, min, max);
    return colorMin*(1.0f - (abs(val-min)/abs(max-min)))+
        colorMax*(abs(val-min)/abs(max-min));
}

/* Mix two colors (RGB) */
vec3 MixColors(float val,
               float min,
               float max,
               vec3 colorMin,
               vec3 colorMax) {
    val = clamp(val, min, max);
    return colorMin*(1.0f - (abs(val-min)/abs(max-min)))+
        colorMax*(abs(val-min)/abs(max-min));
}
