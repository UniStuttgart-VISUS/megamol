
// if you change the binding here
// make sure to change the binding in ShpereRenderer.cpp as well
// see #define AO_DIR_UBO_BINDING_POINT
layout(std140, binding = 0) uniform cone_buffer {
    vec4 coneDirs[NUM_CONEDIRS];
};
