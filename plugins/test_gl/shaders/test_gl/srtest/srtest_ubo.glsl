layout(std140, binding = 1) uniform SceneVars {
    mat4 MVP;
    mat4 MVPinv;
    vec4 viewAttr;
    vec3 camDir, camUp, camRight, camPos;
    vec3 lightDir;
    float near;
    float far;
    float p2_z;
    float p3_z;
    float frustum_ratio_x;
    float frustum_ratio_y;
    float frustum_ratio_w;
    float frustum_ratio_h;
};
