
#extension GL_ARB_shader_storage_buffer_object : require   // glsl version 430
#extension GL_ARB_gpu_shader5 : require                    // glsl version 150
#extension GL_ARB_gpu_shader_fp64 : enable                 // glsl version 150

uniform int instanceOffset;

// Only used by SPLAT render mode:
uniform int attenuateSubpixel;
out float effectiveDiameter;

void main(void) {

    float inColIdx;
    vec4 inColor;
    vec4 inPosition;
