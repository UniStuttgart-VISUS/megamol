#version 430

#include "protein_gl/simplemolecule/sm_common_defines.glsl"
#include "lightdirectional.glsl"

uniform vec4 viewAttr;
uniform vec3 zValues;
uniform vec3 fogCol;
uniform float alpha = 0.5;

uniform mat4 view;
uniform mat4 proj;
uniform mat4 viewInverse;
uniform mat4 mvp;
uniform mat4 mvpinverse;
uniform mat4 mvptransposed;

in vec4 objPos;
in vec4 camPos;
in vec4 lightPos;
in float squarRad;
in float rad;
in vec3 move_color;

void main(void) {
    vec4 coord;
    vec3 ray;
    float lambda;

    // transform fragment coordinates from window coordinates to view coordinates.
    coord = gl_FragCoord 
        * vec4(viewAttr.z, viewAttr.w, 2.0, 0.0) 
        + vec4(-1.0, -1.0, -1.0, 1.0);


    // transform fragment coordinates from view coordinates to object coordinates.
    coord = mvpinverse * coord;
    coord /= coord.w;
    coord -= objPos; // ... and to glyph space

    // calc the viewing ray
    ray = normalize(coord.xyz - camPos.xyz);


    // calculate the geometry-ray-intersection
    float d1 = -dot(camPos.xyz, ray);                       // projected length of the cam-sphere-vector onto the ray
    float d2s = dot(camPos.xyz, camPos.xyz) - d1 * d1;      // off axis of cam-sphere-vector and ray
    float radicand = squarRad - d2s;                        // square of difference of projected length and lambda
    
    if (radicand < 0.0) { discard; }

    float lambdaSign = -1.0;

    lambda = d1 + (lambdaSign * sqrt(radicand));                           // lambda
    vec3 sphereintersection = lambda * ray + camPos.xyz;    // intersection point

    // "calc" normal at intersection point
    vec3 normal = sphereintersection / rad;
#ifdef SMALL_SPRITE_LIGHTING
    normal = mix(-ray, normal, lightPos.w);
#endif // SMALL_SPRITE_LIGHTING

    // chose color for lighting

    vec3 color = move_color.rgb;

    // phong lighting with directional light
    gl_FragColor = vec4(LocalLighting(ray, normal, lightPos.xyz, color), 1.0);
    gl_FragColor = vec4(color,1.0);

    // calculate depth
#ifdef DEPTH
    vec4 Ding = vec4(sphereintersection + objPos.xyz, 1.0);
    float depth = dot(mvptransposed[2], Ding);
    float depthW = dot(mvptransposed[3], Ding);
#ifdef OGL_DEPTH_SES
    gl_FragDepth = ((depth / depthW) + 1.0) * 0.5;
#else
    //gl_FragDepth = ( depth + zValues.y) / zValues.z;
    gl_FragDepth = (depth + zValues.y)/( zValues.z + zValues.y);
#endif // OGL_DEPTH_SES
#endif // DEPTH

#ifdef FOGGING_SES
    float f = clamp( ( 1.0 - gl_FragDepth)/( 1.0 - zValues.x ), 0.0, 1.0);
    gl_FragColor.rgb = mix( fogCol, gl_FragColor.rgb, f);
#endif // FOGGING_SES
    gl_FragColor.a = alpha;
}