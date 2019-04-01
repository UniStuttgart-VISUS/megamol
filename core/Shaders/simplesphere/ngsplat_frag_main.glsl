#extension GL_ARB_explicit_attrib_location : enable
#extension GL_ARB_conservative_depth       : require
layout (depth_greater) out float gl_FragDepth; 

#ifdef BACKSIDE_ENABLED
uniform float hitsideFlag;
#endif // BACKSIDE_ENABLED^

//#define DISCARD_COLOR_MARKER

// clipping plane attributes
uniform vec4 clipDat;
uniform vec4 clipCol;
uniform float alphaScaling;

uniform vec4 viewAttr;

FLACH in vec4 objPos;
FLACH in vec4 camPos;
//FLACH in vec4 lightPos;
FLACH in float squarRad;
FLACH in float rad;
FLACH in float effectiveDiameter;

#ifdef RETICLE
FLACH in vec2 centerFragment;
#endif // RETICLE

in vec4 vertColor;
out layout(location = 0) vec4 outColor;

void main(void) {

    //gl_FragColor = vec4((gl_PointCoord.xy - vec2(0.5)) * vec2(2.0), 0.0, 1.0);
    //gl_FragColor = vec4(gl_PointCoord.xy, 0.5, 1.0);
    //gl_FragColor = vertColor;
    vec2 dist = gl_PointCoord.xy - vec2(0.5);
    float d = sqrt(dot(dist, dist));
    float alpha = 0.5-d;
    alpha *= effectiveDiameter * effectiveDiameter;
    alpha *= alphaScaling;
    //alpha = 0.5;
#if 0
    // blend against white!
    outColor = vec4(vertColor.rgb, alpha);
#else
    outColor = vec4(vertColor.rgb * alpha, alpha);
#endif
    //outColor = vec4(vertColor.rgb, 1.0);
}