#version 140

#define CLIP
#define DEPTH
#define SMALL_SPRITE_LIGHTING
//#define CALC_CAM_SYS

#ifdef DEBUG
#undef CLIP
#define RETICLE
#define AXISHINTS
#endif // DEBUG

//#define BULLSHIT

#ifndef FLACH
#define FLACH
#endif

// TODO: Implementation is wrong! Does positional Lighting instead of directional lighting!

// ray:      the eye to fragment ray vector
// normal:   the normal of this fragment
// lightPos: the position of the light source
// color:    the base material color
vec3 LocalLighting(const in vec3 ray, const in vec3 normal, const in vec3 lightPos, const in vec3 color) {
    // TODO: rewrite!
    vec3 lightDir = normalize(lightPos);

    vec4 lightparams = vec4(0.2, 0.8, 0.4, 10.0);
#define LIGHT_AMBIENT lightparams.x
#define LIGHT_DIFFUSE lightparams.y
#define LIGHT_SPECULAR lightparams.z
#define LIGHT_EXPONENT lightparams.w
    float nDOTl = dot(normal, lightDir);

    vec3 r = normalize(2.0 * vec3(nDOTl) * normal - lightDir);
    return LIGHT_AMBIENT * color 
        + LIGHT_DIFFUSE * color * max(nDOTl, 0.0) 
        + LIGHT_SPECULAR * vec3(pow(max(dot(r, -ray), 0.0), LIGHT_EXPONENT));
}

#ifdef BACKSIDE_ENABLED
uniform float hitsideFlag;
#endif // BACKSIDE_ENABLED
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

in vec4 vsColor;
out vec4 outColor;

void main(void) {

    //gl_FragColor = vec4((gl_PointCoord.xy - vec2(0.5)) * vec2(2.0), 0.0, 1.0);
    //gl_FragColor = vec4(gl_PointCoord.xy, 0.5, 1.0);
    //gl_FragColor = vsColor;
    vec2 dist = gl_PointCoord.xy - vec2(0.5);
    float d = sqrt(dot(dist, dist));
    float alpha = 0.5-d;
    alpha *= effectiveDiameter * effectiveDiameter;
    alpha *= alphaScaling;
    //alpha = 0.5;
#if 1
    // blend against white!
    outColor = vec4(vsColor.rgb, alpha);
#else
    outColor = vec4(vsColor.rgb * alpha, alpha);
#endif
    //gl_FragColor = vec4(vsColor.rgb, 1.0);
}