#version 430

#include "protein_gl/simplemolecule/sm_common_defines.glsl"
#include "protein_gl/deferred/gbuffer_output.glsl"
#include "protein_gl/moleculeses/mses_common_defines.glsl"

in vec4 objPos;
in vec4 camPos;

// texture sampler
uniform sampler2D tex;
uniform vec2 texOffset;

in vec4 inVec1;
in vec4 inVec2;
in vec4 inVec3;
in vec3 inColors;

in vec3 texCoord1;
in vec3 texCoord2;
in vec3 texCoord3;

#include "protein_gl/moleculeses/mses_decodecolor.glsl"
#include "protein_gl/moleculeses/mses_dot1.glsl"

void main(void) {
    vec4 coord;
    vec3 ray;
    float lambda;
    float rad = inVec1.w;
    float squarRad = inVec2.w;

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
    
    lambda = d1 + sqrt(radicand);                           // lambda
    vec3 sphereintersection = lambda * ray + camPos.xyz;    // intersection point

    // compute the actual position of the intersection with the sphere
    vec3 pos1 = sphereintersection + objPos.xyz;
    // cut with plane 1
    vec3 planeNormal = normalize( cross( inVec1.xyz, inVec2.xyz));
    float d = dot( objPos.xyz, planeNormal);
    float dist1 = dot( pos1, planeNormal) - d;
    float dist2 = dot( inVec3.xyz + objPos.xyz, planeNormal) - d;
    if( ( dist2 < 0.0 && dist1 > 0.0 ) || ( dist2 > 0.0 && dist1 < 0.0 ) ) { discard; }
    // cut with plane 2
    planeNormal = normalize( cross( inVec2.xyz, inVec3.xyz));
    d = dot( objPos.xyz, planeNormal);
    dist1 = dot( pos1, planeNormal) - d;
    dist2 = dot( inVec1.xyz + objPos.xyz, planeNormal) - d;
    if( ( dist2 < 0.0 && dist1 > 0.0 ) || ( dist2 > 0.0 && dist1 < 0.0 ) ) { discard; }
    // cut with plane 3
    planeNormal = normalize( cross( inVec1.xyz, inVec3.xyz));
    d = dot( objPos.xyz, planeNormal);
    dist1 = dot( pos1, planeNormal) - d;
    dist2 = dot( inVec2.xyz + objPos.xyz, planeNormal) - d;
    if( ( dist2 < 0.0 && dist1 > 0.0 ) || ( dist2 > 0.0 && dist1 < 0.0 ) ) { discard; }
    // discard the point if it is eaten away by one of the neighbouring probes
    // ==> check first, if one of the probes is nearly dual to the object position
    /*
    if( ( dot1( inProbe1.xyz - objPos.xyz) > 0.1 && ( dot1( pos1 - inProbe1.xyz) < squarRad ) ) || 
        ( dot1( inProbe2.xyz - objPos.xyz) > 0.1 && ( dot1( pos1 - inProbe2.xyz) < squarRad ) ) || 
            ( dot1( inProbe3.xyz - objPos.xyz) > 0.1 && ( dot1( pos1 - inProbe3.xyz) < squarRad ) ) ||
            ( dot1( pos1 - dualProbe) < inVec3.w ) ) { discard; }
    */
    int i;
    vec3 probePos;
    int numProbes = min( int(texCoord1.x), 32);
    if( numProbes > 0 ) {
        for( i = 0; i < numProbes; i++ ) {
            probePos = texelFetch(tex, ivec2( texCoord1.yz) + ivec2( i, 0), 0).xyz;
            if( dot1( probePos - objPos.xyz) > 0.1 && ( dot1( pos1 - probePos) < squarRad ) ) { discard; }
        }
    }
    numProbes = min( int(texCoord2.x), 16);
    if( numProbes > 0 )
    {
        for( i = 0; i < numProbes; i++ )
        {
            //probePos = texture2D( tex, ( texCoord2.yz + vec2( 0.5, 0.5) + vec2( float( i), 0.0))*texOffset).xyz;
            probePos = texelFetch(tex, ivec2( texCoord2.yz) + ivec2( i, 0), 0).xyz;
            if( dot1( probePos - objPos.xyz) > 0.1 && ( dot1( pos1 - probePos) < squarRad ) ) { discard; }
        }
    }
    numProbes = min( int(texCoord3.x), 16);
    if( numProbes > 0 )
    {
        for( i = 0; i < numProbes; i++ )
        {
            //probePos = texture2D( tex, ( texCoord3.yz + vec2( 0.5, 0.5) + vec2( float( i), 0.0))*texOffset).xyz;
            probePos = texelFetch(tex, ivec2( texCoord3.yz) + ivec2( i, 0), 0).xyz;
            if( dot1( probePos - objPos.xyz) > 0.1 && ( dot1( pos1 - probePos) < squarRad ) ) { discard; }
        }
    }
    // "calc" normal at intersection point
    vec3 normal = -sphereintersection / rad;
#ifdef SMALL_SPRITE_LIGHTING
    normal = mix(-ray, normal, 1);
#endif // SMALL_SPRITE_LIGHTING

    // ========== START compute color ==========
    // compute auxiliary direction vectors
    vec3 u = inVec1.xyz - inVec2.xyz;
    vec3 v = inVec3.xyz - inVec2.xyz;
    // base point and direction of ray from the origin to the intersection point
    vec3 w = -inVec2.xyz;
    vec3 dRay = normalize( sphereintersection);
    // cross products for computing the determinant
    vec3 wXu = cross( w, u);
    vec3 dXv = cross( dRay, v);
    // compute interse determinant
    float invdet = 1.0 / dot( dXv, u);
    // compute weights
    float beta = dot( dXv, w) * invdet;
    float gamma = dot( wXu, dRay) * invdet;
    float alpha2 = 1.0 - ( beta + gamma);
    // compute color
    vec3 color = decodeColor( inColors.y) * alpha2 + decodeColor( inColors.x) * beta + decodeColor( inColors.z) * gamma;
#ifdef FLATSHADE_SES
    if( alpha2 > beta && alpha2 > gamma )
        color = decodeColor( inColors.y);
    else if( beta > alpha2 && beta > gamma )
        color = decodeColor( inColors.x);
    else
        color = decodeColor( inColors.z);
#endif // FLATSHADE_SES

    albedo_out = vec4(color,1.0);
    float depthval = gl_FragCoord.z;
    
    // calculate depth
#ifdef DEPTH
    vec4 Ding = vec4(sphereintersection + objPos.xyz, 1.0);
    float depth = dot(mvptransposed[2], Ding);
    float depthW = dot(mvptransposed[3], Ding);
#ifdef OGL_DEPTH_SES
    depthval = ((depth / depthW) + 1.0) * 0.5;
#else
    //gl_FragDepth = ( depth + zValues.x) / zValues.y;
    depthval = (depth + zValues.x)/( zValues.y + zValues.x);
#endif // OGL_DEPTH_SES
#endif // DEPTH

    normal_out = normal;
    gl_FragDepth = depthval;
    depth_out = depthval;
}
