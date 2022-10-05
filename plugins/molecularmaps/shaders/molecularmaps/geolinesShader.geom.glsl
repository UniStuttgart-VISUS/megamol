#version 130

#include "common_defines_btf.glsl"

#extension GL_EXT_geometry_shader4 : enable
#extension GL_EXT_gpu_shader4 : enable

void main(void)
{
    vec3 v0 = vec3(gl_PositionIn[0].xyz) / gl_PositionIn[0].w;
    vec3 v1 = vec3(gl_PositionIn[1].xyz) / gl_PositionIn[1].w;

    // shift positions that are outside the map
    if (v0.x > 1.0) {
        v0.x -= 2.0;
    }
    if (v0.x < -1.0) {
        v0.x += 2.0;
    }
    //if (v0.x > 0.75) {
    //    v0.x -= 2.0;
    //} else if (v0.x < -1.0) {
    //    v0.x += 2.0;
    //}
    if (v1.x > 1.0) {
        v1.x -= 2.0;
    }
    if (v1.x < -1.0) {
        v1.x += 2.0;
    }

    if( v0.x > 0.5 && v1.x < -0.5) {
        gl_FrontColor = gl_FrontColorIn[0];
        gl_Position = vec4( v0, 1.0);
        EmitVertex();

        gl_FrontColor = gl_FrontColorIn[1];
        gl_Position = vec4( v1 + vec3(2.0, 0.0, 0.0), 1.0);
        EmitVertex();
        EndPrimitive();

        gl_FrontColor = gl_FrontColorIn[0];
        gl_Position = vec4( v0 - vec3(2.0, 0.0, 0.0), 1.0);
        EmitVertex();

        gl_FrontColor = gl_FrontColorIn[1];
        gl_Position = vec4( v1, 1.0);
        EmitVertex();
        EndPrimitive();
    } else if( v0.x < -0.5 && v1.x > 0.5) {
        gl_FrontColor = gl_FrontColorIn[0];
        gl_Position = vec4( v0 + vec3(2.0, 0.0, 0.0), 1.0);
        EmitVertex();

        gl_FrontColor = gl_FrontColorIn[1];
        gl_Position = vec4( v1, 1.0);
        EmitVertex();
        EndPrimitive();

        gl_FrontColor = gl_FrontColorIn[0];
        gl_Position = vec4( v0, 1.0);
        EmitVertex();

        gl_FrontColor = gl_FrontColorIn[1];
        gl_Position = vec4( v1 - vec3(2.0, 0.0, 0.0), 1.0);
        EmitVertex();
        EndPrimitive();
    } else {
        gl_FrontColor = gl_FrontColorIn[0];
        gl_Position = vec4( v0, 1.0);
        EmitVertex();

        gl_FrontColor = gl_FrontColorIn[1];
        gl_Position = vec4( v1, 1.0);
        EmitVertex();
        EndPrimitive();
    }
}
