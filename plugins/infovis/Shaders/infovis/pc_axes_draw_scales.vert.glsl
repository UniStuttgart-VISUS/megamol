#version 430

#include "pc_common/pc_extensions.inc.glsl"
#include "pc_common/pc_buffers.inc.glsl"
#include "pc_common/pc_uniforms.inc.glsl"
#include "pc_common/pc_common.inc.glsl"

out vec4 actualColor;
uniform float axesThickness;
uniform int width;
uniform int height;

void main()
{

    uint dimension = pc_dimension(gl_InstanceID / numTicks);
    vec4 bottom = axis_line(gl_InstanceID / numTicks, 0);
    vec4 top = axis_line(gl_InstanceID / numTicks, 1);

    int realID = gl_VertexID %2;

    vec4 vertex = vec4(
    bottom.x - axisHalfTick + 2 * realID * axisHalfTick,
    mix(bottom.y, top.y, (gl_InstanceID % numTicks) / float(numTicks - 1)),
    bottom.z,
    bottom.w);

    int side = gl_VertexID / 2 - gl_VertexID/3;

    if (dimension == pickedAxis) {
        actualColor = vec4(1.0, 0.0, 0.0, 1.0);
    } else {
        actualColor = color;
    }
    gl_Position = projection * modelView * vertex + axesThickness * (vec4(0, 2.0/height, 0, 0) + side * vec4(0, -4.0/height, 0, 0));
}
