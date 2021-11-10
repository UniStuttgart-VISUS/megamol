layout(std140, binding = 1) uniform SceneVars {
    mat4 MVP;
    mat4 MVPinv;
    mat4 MVPtransp;
    vec4 viewAttr;
    vec3 camDir, camUp, camRight, camPos;
    float near;
    vec3 lightDir;
    float far;
};
