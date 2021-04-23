#version 430

#include <snippet name="::pc::extensions" />
#include <snippet name="::pc::buffers" />
#include <snippet name="::pc::uniforms" />
#include <snippet name="::pc::common" />

out vec4 actualColor;
uniform float axesThickness;
uniform int width;
uniform int height;

void main()
{
    uint dimension = pc_dimension(gl_InstanceID);
    int realID = gl_VertexID % 2;
    int side = gl_VertexID / 2 - gl_VertexID/3;

    vec4 vertex = axis_line(gl_InstanceID, realID);

    if (dimension == pickedAxis) {
        actualColor = vec4(1.0, 0.0, 0.0, 1.0);
    } else {
        actualColor = color;
    }

    gl_Position = projection * modelView * vertex + axesThickness * (vec4(2.0 / width, 0, 0, 0) + side * vec4(-4.0 / width, 0, 0, 0));
}
