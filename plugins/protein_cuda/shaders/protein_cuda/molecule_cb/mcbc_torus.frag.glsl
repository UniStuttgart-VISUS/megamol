#version 120

#include "protein_cuda/molecule_cb/mcbc_common.glsl"
#include "lightdirectional.glsl"
#include "protein_cuda/molecule_cb/mcbc_rootsolver.glsl"

uniform vec4 viewAttr;
uniform vec3 zValues;
uniform vec3 fogCol;
uniform float alpha = 0.5;
uniform mat4 mvpinv;
uniform mat4 mvptrans;

varying vec4 objPos;
varying vec4 camPos;
varying vec4 lightPos;
varying vec4 radii;
varying vec4 visibilitySphere;

varying vec3 rotMatT0;
varying vec3 rotMatT1; // rotation matrix from the quaternion
varying vec3 rotMatT2;

varying vec4 colors;

#include "protein_cuda/molecule_cb/mcbc_decodecolor.glsl"

void main(void) {
    vec4 coord;
    vec3 ray, tmp;

    // transform fragment coordinates from window coordinates to view coordinates.
    coord = gl_FragCoord 
        * vec4(viewAttr.z, viewAttr.w, 2.0, 0.0) 
        + vec4(-1.0, -1.0, -1.0, 1.0);

    // transform fragment coordinates from view coordinates to object coordinates.
    coord = mvpinv * coord;
    coord /= coord.w;
    coord -= objPos; // ... and move

    // calc the viewing ray
    ray = rotMatT0 * coord.x + rotMatT1 * coord.y + rotMatT2 * coord.z;
    ray = normalize( ray - camPos.xyz);

    #define r radii.x
    #define r2 radii.y
    #define R radii.z
    #define R2 radii.w

    // calculate the base point of the ray
    vec3 a = camPos.xyz + ( length( camPos.xyz) - (R + r)) * ray;
    
    // compute coefficients of the quartic equation for the ray-torus-intersection
    float K = dot( a, a) - ( R2 + r2);
    float A = 4.0 * dot( a, ray);
    float B = 2.0 * ( 2.0 * dot( a, ray) * dot( a, ray) + K + 2.0 * R2 * ray.z*ray.z);
    float C = 4.0 * ( K * dot( a, ray) + 2.0 * R2 * a.z * ray.z);
    float D = K*K + 4.0 * R2 * ( a.z*a.z - r2);
    
    vec4 lambdas = vec4( 0.0, 0.0, 0.0, 0.0);
    vec3 intersection;
    // ==> various quartic root solvers (only stabilized ferrari shows good results and performance)
    //int numRoots = quartic( A, B, C, D, lambdas);
    int numRoots = ferrari( A, B, C, D, lambdas);
    //int numRoots = neumark( A, B, C, D, lambdas);
    //int numRoots = simpleFerrari( A, B, C, D, lambdas);
    //int numRoots = descartes( A, B, C, D, lambdas);
    if( numRoots < 2 ) { discard; }
    // get lambda of the first intersection
    /*
    // ==> this test is unnecessary, lambdas.x always holds the value of the second intersection [kroneml 31.8.2008]
    float lambda = lambdas.x;
    if( lambdas.y < lambda && numRoots > 1 && lambdas.y > 0.0 ) { lambda = lambdas.y; }
    if( lambdas.z < lambda && numRoots > 2 && lambdas.z > 0.0 ) { lambda = lambdas.z; }
    if( lambdas.w < lambda && numRoots > 3 && lambdas.w > 0.0 ) { lambda = lambdas.w; }
    // get lambda of the second intersection
    float second;
    if( lambdas.x > lambda ) { second = lambdas.x; } else { second = lambdas.y; }
    if( lambdas.y > lambda && lambdas.y < second && numRoots > 1 ) { second = lambdas.y; }
    if( lambdas.z > lambda && lambdas.z < second && numRoots > 2 ) { second = lambdas.z; }
    if( lambdas.w > lambda && lambdas.w < second && numRoots > 3 ) { second = lambdas.w; }
    */
    float second = lambdas.x;
    intersection = a + ray * second;
    // handle singularity
    bool sing = false;
    if( r > R )
    {
        //float radius2 = r2 - 2.0 * ( r2 -r*R) + r2 - 2.0 * r * R + R2;
        //float cutRad2 = r2 - radius2;
        //float cutRad2 = 2.0 * ( r2 - r * R) - r2 + 2.0 * r * R - R2;
        float cutRad2 = r2 - R2;
        if( dot( intersection, intersection) < cutRad2 )
        {
            second = lambdas.x;
            if( lambdas.y > second && numRoots > 1 ) { second = lambdas.y; }
            if( lambdas.z > second && numRoots > 2 ) { second = lambdas.z; }
            if( lambdas.w > second && numRoots > 3 ) { second = lambdas.w; }
            intersection = a + ray * second;
        }
    }
    // discard fragment if the intersection point lies outside the sphere
    if( length( intersection - visibilitySphere.xyz) > visibilitySphere.w ) { discard; }
    
    // compute inward-facing normal
    vec3 normal;
    //normal = ( intersection - vec3( normalize( intersection.xy), 0.0));
    float factor01 = ( dot( intersection, intersection) - r2 - R2);
    normal.x = 4.0*intersection.x*factor01;
    normal.y = 4.0*intersection.y*factor01;
    normal.z = 4.0*intersection.z*factor01 + 8.0*R2*intersection.z;
    normal = -normalize( normal);
    
    vec3 color;
    /*
    float d = ( dot( intersection, vec3( 0.0, 0.0, 1.0)) - colors.z) / colors.w;
    // chose color for lighting
    color = mix( decodeColor( colors.x), decodeColor( colors.y), d);
#ifdef FLATSHADE_SES
    if( d <= 0.5 )
        color = decodeColor( colors.x);
    else
        color = decodeColor( colors.y);
#endif // FLATSHADE_SES
    */
    // uniform color
    color = vec3( 1.0, 0.75, 0.0);
    //color = vec3( 0.98, 0.82, 0.0 ); // for VIS
    //color = vec3( 0.02, 0.75, 0.02);
    //color = vec3( 0.19, 0.52, 0.82);

#ifdef COLOR_SES
    color = COLOR_GREEN;
#endif    
#ifdef SET_COLOR
    color = COLOR1;
#endif

#ifdef SFB_DEMO
    color = vec3(0.70f, 0.8f, 0.4f);
#endif

    // phong lighting with directional light
    gl_FragColor = vec4( LocalLighting( ray, normal, lightPos.xyz, color), 1.0);
    
    // calculate depth
#ifdef DEPTH
    tmp = intersection;
    intersection.x = dot( rotMatT0, tmp.xyz);
    intersection.y = dot( rotMatT1, tmp.xyz);
    intersection.z = dot( rotMatT2, tmp.xyz);

    intersection += objPos.xyz;

    vec4 Ding = vec4( intersection, 1.0);
    float depth = dot(mvptrans[2], Ding);
    float depthW = dot(mvptrans[3], Ding);
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
    gl_FragColor.rgb = color;
}
