#version 430

#include "pc_common/pc_extensions.inc.glsl"
#include "pc_common/pc_buffers.inc.glsl"
#include "pc_common/pc_uniforms.inc.glsl"
#include "pc_common/pc_common.inc.glsl"

out vec4 actualColor;
uniform ivec2 pickedIndicator;
uniform float axesThickness;
uniform int width;
uniform int height;

void main()
{

    uint dimension = pc_dimension(gl_InstanceID / 2);
    vec4 bottom = axis_line(gl_InstanceID / 2, 0);
    vec4 top = axis_line(gl_InstanceID / 2, 1);

    int realID = int(gl_VertexID / 6) + gl_VertexID % 2;
    int side = gl_VertexID / 2 - gl_VertexID/3 - gl_VertexID / 6;

    float iAmTop = (gl_InstanceID % 2);
    float topOffset = iAmTop * 2.0 - 1.0; // top -> 1, bottom -> -1
    float val = (topOffset > 0) ? filters[dimension].upper : filters[dimension].lower;
    val = val - dataMinimum[dimension];
    val /= dataMaximum[dimension] - dataMinimum[dimension];
    vec4 vertex = vec4(
    bottom.x - axisHalfTick + realID * axisHalfTick,
    mix(bottom.y, top.y, val) + axisHalfTick * (-topOffset * (-1 + realID % 2)),
    bottom.z,
    bottom.w);



    if ((dimension == pickedIndicator.x) && (abs(iAmTop - pickedIndicator.y) < 0.1)) {
        actualColor = vec4(1.0, 0.0, 0.0, 1.0);
    } else {
        actualColor = color;
    }
    gl_Position = projection * modelView * vertex;
    gl_Position = gl_Position + axesThickness * (iAmTop * side * vec4(0, 16.0 / height, 0, 0) + side * vec4(0, -8.0 / height, 0, 0));
}
