#version 430

#include "core/tflookup.inc.glsl"
#include "core/tfconvenience.inc.glsl"
#include "core/bitflags.inc.glsl"
#include "common/mapping.inc.glsl"
#include "common/plots.inc.glsl"
#include "common/data.inc.glsl"

uniform vec2 frustrumSize;
uniform ivec2 viewRes;
uniform mat4 modelViewProjection;

uniform int rowStride;

uniform float kernelWidth;
uniform int attenuateSubpixel;

out float vsKernelSize;
out vec2 vsPosition;
flat out vec2 vsPointPosition;
flat out int vsPointID;

// triangle with inner circle around center of gravity with radius 1
const vec2 vertices[3] = vec2[3](vec2( -1.74,-1.0 ),
                                 vec2( 0.0,2.0 ),
                                 vec2( 1.74,-1.0 ));

float rand(vec2 co){
    return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453);
}

void main() {
    const int pointID = int(floor(gl_VertexID / 3));
    vsPointID = pointID;
    const Plot plot = plots[gl_InstanceID];
    const int rowOffset = pointID * rowStride;

    // note: although its called width, it is here used as radius to be visually consistent
    //       with GL_POINTS draw mode
    vsKernelSize = kernelWidth;
    
    if (attenuateSubpixel==1) {
        // Compute pixel size (of view res) in world space.
        vec2 wsPixelSize = frustrumSize / vec2(viewRes.x,viewRes.y);
        // Ensure a minimum pixel size to attenuate alpha depending on subpixels.
        // (at least half a pixel for kernel radius)
        vsKernelSize = max(vsKernelSize, max(wsPixelSize.x,wsPixelSize.y) * 0.5);
        //vsKernelSize = max(vsKernelSize, length(wsPixelSize) * 0.5);
    }

    vec2 posValues = vec2(values[rowOffset + plot.indexX], values[rowOffset + plot.indexY]);
    vsPosition = valuesToPosition(plot,posValues).xy;
    
    vsPointPosition = vsPosition;
    vsPosition += vertices[gl_VertexID % 3].xy * (vsKernelSize);

    gl_Position = modelViewProjection * vec4(vsPosition,0.0,1.0);
}
