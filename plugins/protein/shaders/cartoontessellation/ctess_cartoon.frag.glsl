#version 430

#include "simplemolecule/sm_common_defines.glsl"
#include "simplemolecule/sm_common_gbuffer_output.glsl"

layout (depth_greater) out float gl_FragDepth; // we think this is right
// this should be wrong //layout (depth_less) out float gl_FragDepth;
#extension GL_ARB_explicit_attrib_location : enable

uniform mat4 MVP;
uniform mat4 MV;
uniform mat4 MVPinv;
uniform mat4 MVPtransp;
uniform mat4 ProjInv;

uniform vec4 lightPos;
uniform vec4 phong = vec4(0.4, 0.8, 0.4, 10.0);

in vec4 mycol;
in vec3 normal;
in vec3 rawnormal;

void main(void) {
    vec4 pos = ProjInv * gl_FragCoord.xyzw;
    vec4 light = MV * lightPos;
    vec3 lightDir = normalize(light.xyz - pos.xyz);
    
    vec3 n = normal;
    vec3 e = pos.xyz;
    vec3 h = normalize(lightDir + e);
    float d = dot(n, lightDir);
    vec3 r = normalize(2.0 * d * n - lightDir);
    vec3 c = phong.x * mycol.xyz +
        phong.y * mycol.xyz * max(d, 0.0);
        
    vec4 eyenew = vec4(0.0, 0.0, 0.0, 1.0);
    
    if(dot(lightDir, n) < 0.0)
        c += vec3(0.0);
    else
        c += phong.z * vec3(pow(max(dot(r, normalize(eyenew.xyz - e)), 0.0), phong.w));

    albedo_out = mycol;    
    albedo_out = vec4(c, 1); // TODO delete lighting here
    normal_out = normal;
    depth_out = gl_FragCoord.z;
    gl_FragDepth = depth_out;
}