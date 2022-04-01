#version 130

#include "protein_cuda/molecule_cb/mcbc_common.glsl"
#include "lightdirectional.glsl"

uniform vec4 viewAttr;
uniform vec3 zValues;
uniform vec3 fogCol;
uniform vec2 texOffset;
// texture sampler
uniform sampler2D tex;
uniform float alpha = 0.5;
uniform mat4 mvpinv;
uniform mat4 mvptrans;

varying vec4 objPos;
varying vec4 camPos;
varying vec4 lightPos;

varying vec4 inVec1;
varying vec4 inVec2;
varying vec4 inVec3;
varying vec3 inColors;

varying vec3 texCoord1;
varying vec3 texCoord2;
varying vec3 texCoord3;

#include "protein_cuda/molecule_cb/mcbc_decodecolor.glsl"
#include "protein_cuda/molecule_cb/mcbc_dot1.glsl"

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
    coord = mvpinv * coord;
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
            probePos = texelFetch( tex, ivec2( texCoord1.yz) + ivec2( i, 0), 0).xyz;
            if( dot1( probePos - objPos.xyz) > 0.1 && ( dot1( pos1 - probePos) < squarRad ) ) { discard; }
        }
    }
    numProbes = min( int(texCoord2.x), 16);
    if( numProbes > 0 )
    {
        for( i = 0; i < numProbes; i++ )
        {
            //probePos = texture2D( tex, ( texCoord2.yz + vec2( 0.5, 0.5) + vec2( float( i), 0.0))*texOffset).xyz;
            probePos = texelFetch( tex, ivec2( texCoord2.yz) + ivec2( i, 0), 0).xyz;
            if( dot1( probePos - objPos.xyz) > 0.1 && ( dot1( pos1 - probePos) < squarRad ) ) { discard; }
        }
    }
    numProbes = min( int(texCoord3.x), 16);
    if( numProbes > 0 )
    {
        for( i = 0; i < numProbes; i++ )
        {
            //probePos = texture2D( tex, ( texCoord3.yz + vec2( 0.5, 0.5) + vec2( float( i), 0.0))*texOffset).xyz;
            probePos = texelFetch( tex, ivec2( texCoord3.yz) + ivec2( i, 0), 0).xyz;
            if( dot1( probePos - objPos.xyz) > 0.1 && ( dot1( pos1 - probePos) < squarRad ) ) { discard; }
        }
    }
    // "calc" normal at intersection point
    vec3 normal = -sphereintersection / rad;
#ifdef SMALL_SPRITE_LIGHTING
    normal = mix(-ray, normal, lightPos.w);
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
    // ========== END compute color ==========
    // uniform color
    //color = vec3( 0.0, 0.75, 1.0);
    //color = vec3( 0.0, 0.6, 0.6 ); // for VIS
    //color = vec3( texCoord2.xy, 0.0);
    //color = vec3( 0.8, 0.0, 0.2);
    //color = vec3( 0.19, 0.52, 0.82);

#ifdef COLOR_SES
    color = COLOR_YELLOW;
#endif    
#ifdef SET_COLOR
    color = COLOR1;
#endif

#ifdef SFB_DEMO
    color = vec3(0.70f, 0.8f, 0.4f);
#endif

    // phong lighting with directional light
    gl_FragColor = vec4( LocalLighting(ray, normal, lightPos.xyz, color), 1.0);
    gl_FragDepth = gl_FragCoord.z;
    
    // calculate depth
#ifdef DEPTH
    vec4 Ding = vec4(sphereintersection + objPos.xyz, 1.0);
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
    
#ifdef AXISHINTS
    // debug-axis-hints
    vec3 colorX = vec3( 1.0, 1.0, 0.0);
    float mc = min(abs(normal.x), min(abs(normal.y), abs(normal.z)));
    if( abs(normal.x) - 0.05 < abs(normal.z) && abs(normal.x) + 0.05 > abs(normal.z) )
    { gl_FragColor = vec4( LocalLighting( ray, normal, lightPos.xyz, colorX), 1.0); }
    if( abs(normal.x) - 0.05 < abs(normal.y) && abs(normal.x) + 0.05 > abs(normal.y) )
    { gl_FragColor = vec4( LocalLighting( ray, normal, lightPos.xyz, colorX), 1.0); }
    if( abs(normal.z) - 0.05 < abs(normal.y) && abs(normal.z) + 0.05 > abs(normal.y) )
    { gl_FragColor = vec4( LocalLighting( ray, normal, lightPos.xyz, colorX), 1.0); }
    if (mc < 0.05) { gl_FragColor = vec4( LocalLighting( ray, normal, lightPos.xyz, colorX), 1.0); }
#endif // AXISHINTS

#ifdef PUXELS
if(puxels_use != 0)
    puxels_store(makePuxel(packUnorm4x8(gl_FragColor), normal, gl_FragDepth));
#endif

    gl_FragColor.rgb = vec3(1,0,0);

}
