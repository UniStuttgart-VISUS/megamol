#version 120

// Copyright (c) 2009  Martin Falk <falk@vis.uni-stuttgart.de>
//                     Visualization Research Center (VISUS),
//                     Universitaet Stuttgart, Germany
//                     http://www.vis.uni-stuttgart.de/~falkmn/
//      modified 2010  Michael Krone <kroneml@vis.uni-stuttgart.de>
// This program may be distributed, modified, and used free of charge
// as long as this copyright notice is included in its original form.
// Commercial use is strictly prohibited.

#define ISO_SURFACE

#if !defined(ISO_SURFACE) && !defined(VOLUME_RENDERING)
#  define VOLUME_RENDERING
#endif


// scale factors of the volume
uniform vec4 scaleVol;
uniform vec4 scaleVolInv;

uniform float stepSize;

uniform int numIterations;
uniform float alphaCorrection;

uniform vec2 screenResInv;
uniform float isoValue;
uniform float isoOpacity = 0.4;
uniform float clipPlaneOpacity = 0.4;

//uniform vec4 clipPlane0 = vec4( 0.0);

// textures
uniform sampler3D volumeSampler;
uniform sampler1D transferRGBASampler;

uniform sampler2D rayStartSampler;   // ray start pos, a=0: no ray
uniform sampler2D rayLengthSampler;  // ray direction, ray length

varying vec3 lightPos;
varying vec3 fillLightPos;


vec3 calcIllum(vec3 illumPos, vec3 normal, vec3 dir, vec3 srcColor) {
    // main light source
    vec3 lightDir = normalize(lightPos - illumPos);
    float ndotl = (dot(normal, lightDir));
    vec3 r = normalize(2.0 * ndotl * normal - lightDir);
    float spec = pow(max(dot(r, -dir), 0.0), 10.0) * 0.5;
    float diff = max(abs(ndotl), 0.0);
    // fill light
    lightDir = normalize(fillLightPos - illumPos);
    ndotl = dot(normal, lightDir);
    r = normalize(2.0 * ndotl * normal - lightDir);
    float specFill = pow(max(dot(r, -dir), 0.0), 20.0) * 0.24;
    float diffuseFill = 0.5*max(abs(ndotl), 0.0);

    vec3 color = (diff + diffuseFill + 0.3)*srcColor.rgb + ( spec + specFill) * 0.5;

    return color;
}


void main(void) {
    vec4 dest = vec4(0.0);
    vec4 src;
    float w;

    vec2 texCoord = gl_FragCoord.xy * screenResInv;

    vec4 rayStart  = texture2D(rayStartSampler, texCoord);
    vec4 rayLength = texture2D(rayLengthSampler, texCoord);

    vec3 center = vec3(0.0);

    // DEBUG
    //gl_FragColor = vec4(rayLength.www, 1.0);
    //gl_FragColor = vec4(rayLength.www*0.1, 1.0);
    //gl_FragColor = vec4(rayLength.xyz, 1.0);
    //gl_FragColor = vec4(rayStart.xyz*scaleVol.xyz, 0.8);
    //gl_FragColor = vec4(rayStart.aaa, 0.8);
    //return;

    // ray starting position
    vec3 pos = rayStart.xyz * scaleVol.xyz;
    // ray direction
    vec3 dir = rayLength.xyz * scaleVol.xyz;
    // ray distance to traverse
    float rayDist = rayLength.w;

    float scalarData;
    vec4 data;
    vec3 volColor; // DEBUG

    bool outside = false;

    // move one step forward
    vec3 stepDir = dir * stepSize;
    pos += stepDir;
    rayDist -= stepSize;

    if (rayDist <= 0.0) {
        gl_FragColor = dest;
        return;
    }

        #if defined(ISO_SURFACE)
    //const float isoValue = 60.0/256.0;
    float isoDiff = 0;
    float isoDiffOld = texture3D(volumeSampler, pos).w - isoValue;

    if( isoDiffOld > 0.0 ) {
        volColor = clamp( texture3D(volumeSampler, pos).rgb, 0.0, 1.0);
        // higher opacity for surfaces orthogonal to view dir
        dest = vec4( volColor, clipPlaneOpacity);
        // perform blending
        dest.rgb *= dest.a;
    }
        #endif

    for (int j=0; (!outside && (j<numIterations)); ++j)
    {
        for (int i=0; i<numIterations; ++i)
        {
            /*
            // lookup scalar value
            scalarData = texture3D(volumeSampler, pos).w;
            // DEBUG lookup color
            volColor = texture3D(volumeSampler, pos).rgb;
            */
            data = texture3D(volumeSampler, pos);
            scalarData = data.w;
            volColor = clamp( data.rgb, 0.0, 1.0);

            #if defined(VOLUME_RENDERING)
            // lookup in transfer function
            //src = texture1D(transferRGBASampler, log(scalarData*1.0+1.0));
            //src = texture1D(transferRGBASampler, scalarData/1.2-0.18);
            src = texture1D(transferRGBASampler, scalarData);
            src = vec4( vec3( scalarData), 0.05); // DEBUG
            // DEBUG ...
            src = vec4( clamp( abs( scalarData), 0.0, 1.0) );
            if( scalarData > 0.0 ) src *= vec4( 1.0, 0.0, 0.0, 1.0);
            else if( scalarData < 0.0 ) src *= vec4( 0.0, 1.0, 0.0, 1.0);
            // ... DEBUG

            //src.rgb = vec3(0.6-2.5*scalarData) * vec3(0.2, 1.0, 0.1);

            // opacity and color correction
            src.a = 1.0 - pow(1.0 - src.a, alphaCorrection);

            src.rgb *= src.a;
            // perform blending
            dest = (1.0-dest.a)*src + dest;

            #endif
            #if defined(ISO_SURFACE)
            isoDiff = scalarData - isoValue;

            if (isoDiff*isoDiffOld <= 0.0) {
                // improve position
                vec3 isoPos = mix(pos - stepDir, pos, isoDiffOld/(isoDiffOld - isoDiff));

                // lookup color
                volColor = texture3D( volumeSampler, isoPos).rgb;
                volColor = clamp( volColor, 0.0, 1.0);

                // compute gradient by central differences
                vec3 gradient;
                // TODO: schrittweite skalieren (halbe voxel-laenge)
                //float gradOffset = 0.05; // 0.0038
                float gradOffset = 0.075;
                // 0.015 ==> huebsch.dat

                gradient.x = texture3D(volumeSampler, isoPos + vec3(gradOffset*scaleVol.x, 0, 0)).w
                - texture3D(volumeSampler, isoPos + vec3(-gradOffset*scaleVol.x, 0, 0)).w;
                gradient.y = texture3D(volumeSampler, isoPos + vec3(0, gradOffset*scaleVol.y, 0)).w
                - texture3D(volumeSampler, isoPos + vec3(0, -gradOffset*scaleVol.y, 0)).w;
                gradient.z = texture3D(volumeSampler, isoPos + vec3(0, 0, gradOffset*scaleVol.z)).w
                - texture3D(volumeSampler, isoPos + vec3(0, 0, -gradOffset*scaleVol.z)).w;
                gradient = normalize(gradient);

                // illumination
                vec3 posWorld = isoPos * scaleVolInv.xyz;
                vec3 dirWorld = dir * scaleVolInv.xyz;

                float tmp = sqrt(abs(dot(rayLength.xyz, gradient)));
                // DEBUG src = vec4(0.45, 0.55, 0.8, min( 1.0, isoOpacity + 1.5*(1.0-tmp))); // higher opacity for surfaces orthogonal to view dir
                //src = vec4(0.45, 0.55, 0.8, 0.6);
                src = vec4( volColor, min( 1.0, isoOpacity + 1.5*(1.0-tmp))); // DEBUG
                src.rgb = calcIllum( posWorld, gradient, dirWorld, src.rgb);
                //src.rgb = -gradient*0.5 + 0.5;

                // draw interior darker
                if( isoDiffOld > 0.0 )
                src.xyz *= 0.5;

                // perform blending
                src.rgb *= src.a;
                dest = (1.0-dest.a)*src + dest;

                // rotate clip plane normal
                //vec3 cp = gl_ClipPlane[0].xyz * gl_NormalMatrix;
                // if( dot( posWorld - scaleVol.xyz, clipPlane0.xyz ) + clipPlane0.w < 0.05 )
                //  //dest.xyz *= 0.2;
                //  dest.xyz += 0.5 * vec3( 1.0, 1.0, 0.0);
                // dest.xyz = clamp( dest.xyz, vec3( 0.0), vec3( 1.0));
            }
            isoDiffOld = isoDiff;
            #endif // ISO_SURFACE

            // move one step forward
            pos += stepDir;
            rayDist -= stepSize;

            outside = (dest.a > 0.98) || (rayDist <= 0);
            if (outside)
            break;
        }
    }

    gl_FragColor = dest;
}
