#version 120
#extension GL_EXT_geometry_shader4 : enable
#extension GL_EXT_gpu_shader4 : enable

varying vec3 fragTangent;
varying vec3 fragView;

void main(void) {
    gl_FrontColor = gl_FrontColorIn[1];
    gl_Position = gl_ModelViewProjectionMatrix*gl_PositionIn[1];
    fragTangent = normalize(gl_PositionIn[2].xyz-gl_PositionIn[0].xyz);
    fragView = (gl_ModelViewMatrix*gl_PositionIn[1]).xyz;
    EmitVertex();

    gl_FrontColor = gl_FrontColorIn[2];
    gl_Position = gl_ModelViewProjectionMatrix*gl_PositionIn[2];
    fragTangent = normalize(gl_PositionIn[3].xyz-gl_PositionIn[1].xyz);
    fragView = (gl_ModelViewMatrix*gl_PositionIn[2]).xyz;
    EmitVertex();
}
